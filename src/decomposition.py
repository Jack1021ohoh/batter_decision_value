"""The decomposed swing model (Track B).

A swing ends one of three ways, and the target is the run value of
(outcome, count), so the value of swinging is exactly

    Q_swing(s) = P(whiff | s) * RE(whiff, c)
               + P(foul  | s) * RE(foul, c)
               + P(BIP   | s) * E[target | BIP, s]

The direct twin estimates the same conditional mean in one regression. The
decomposition estimates it in pieces: a three-class classifier for what the
swing produces, and a model for what a ball in play is worth -- either a
regression of its run value, or (`in_play='classifier'`) the probability of
each in-play event weighted by its run value in the count. Whiffs are
far more predictable than run value, which is the case for splitting them out.

`DecomposedModels.predict(df, action)` has the same signature as
`ActionModels.predict`, so everything downstream -- `predict_both`,
`predict_chosen`, the decision scores and the whole harness -- runs on either
twin unchanged. The take model is fitted by the same `fit_take` call as the
direct twin's, so both twins share one `Q_take` and every difference in the
decision score comes from the swing branch.

Hitter priors are routed by kind: contact priors (whiff and foul tendencies)
to the classifier, damage priors (the hot zone) to the ball-in-play regressor,
none to the take model. The direct twin gets all of them in its one swing
regression.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from .baselines import (REGRESSION, SWING_ONLY_FEATURES, _dmatrix, fit_take,
                        train_early_stopped)
from .features import BIP_OUTCOMES

#: The three things a swing can produce, in classifier column order.
SWING_CLASSES = ('whiff', 'foul', 'bip')

#: Priors that describe making contact, and priors that describe what contact
#: is worth. Each goes only to the sub-model it describes.
CONTACT_PRIORS = {'prior_whiff', 'prior_foul'}
DAMAGE_PRIORS = {'hot_zone'}

CLASSIFIER = {'objective': 'multi:softprob', 'num_class': len(SWING_CLASSES),
              'eval_metric': 'mlogloss'}

#: What a ball in play can become, in in-play classifier column order.
IN_PLAY_EVENTS = tuple(BIP_OUTCOMES)
IN_PLAY_CLASSIFIER = {'objective': 'multi:softprob', 'num_class': len(IN_PLAY_EVENTS),
                      'eval_metric': 'mlogloss'}


def swing_class(outcome: pd.Series) -> np.ndarray:
    """0 = whiff (incl. foul tips), 1 = foul, 2 = ball in play; -1 if not a swing outcome."""
    out = np.full(len(outcome), -1)
    o = outcome.astype(str).to_numpy()
    out[o == 'swinging_strike'] = 0
    out[o == 'foul'] = 1
    out[np.isin(o, BIP_OUTCOMES)] = 2
    return out


def in_play_class(outcome: pd.Series) -> np.ndarray:
    """Index of each ball-in-play outcome in `IN_PLAY_EVENTS`; -1 if not in play."""
    lookup = {e: i for i, e in enumerate(IN_PLAY_EVENTS)}
    return outcome.astype(str).map(lookup).fillna(-1).astype(int).to_numpy()


@dataclass
class DecomposedModels:
    take: xgb.Booster
    outcome: xgb.Booster
    bip_value: xgb.Booster
    take_features: list[str]
    outcome_features: list[str]
    value_features: list[str]
    re_whiff: pd.Series                      # RE(swinging_strike, count), indexed by count
    re_foul: pd.Series                       # RE(foul, count)
    rounds: dict
    in_play: str = 'regression'              # how a ball in play is valued
    re_in_play: pd.DataFrame | None = None   # RE(event, count): counts x IN_PLAY_EVENTS

    def components(self, df: pd.DataFrame) -> pd.DataFrame:
        """The pieces of `Q_swing` for every row: class probabilities and ball-in-play value.

        With `in_play='classifier'` the in-play value is the probability of each
        event times its run value in the pitch's count, and the event
        probabilities (`p_single`, ...) are returned too.
        """
        p = self.outcome.predict(_dmatrix(df, self.outcome_features))
        out = pd.DataFrame({'p_whiff': p[:, 0], 'p_foul': p[:, 1], 'p_bip': p[:, 2]}, index=df.index)
        v = self.bip_value.predict(_dmatrix(df, self.value_features))
        if self.in_play == 'regression':
            out['bip_value'] = v
            return out
        re = self.re_in_play.loc[df['count'].astype(str), list(IN_PLAY_EVENTS)].to_numpy()
        for k, event in enumerate(IN_PLAY_EVENTS):
            out[f'p_{event}'] = v[:, k]
        out['bip_value'] = (v * re).sum(axis=1)
        return out

    def predict(self, df: pd.DataFrame, action: str) -> np.ndarray:
        if action == 'take':
            return self.take.predict(_dmatrix(df, self.take_features))
        c = self.components(df)
        count = df['count'].astype(str)
        return (c['p_whiff'].to_numpy() * count.map(self.re_whiff).to_numpy()
                + c['p_foul'].to_numpy() * count.map(self.re_foul).to_numpy()
                + c['p_bip'].to_numpy() * c['bip_value'].to_numpy())


def fit_decomposed(train: pd.DataFrame, features: list[str], rv: pd.Series,
                   target: str = 'target', in_play: str = 'regression', **_) -> DecomposedModels:
    """Fit the decomposed twin on one fold's training rows.

    `features` is the same list the direct twin gets. The take model sees the
    pitch features only; the classifier adds any contact priors; the
    ball-in-play model adds any damage priors. `rv` is the fold's
    (outcome, count) run-value table -- the one that defines `target`.

    `in_play` chooses how a ball in play is valued:

    * `'regression'` -- regress its run value directly;
    * `'classifier'` -- predict the probability of each of `IN_PLAY_EVENTS`
      and weight each by its run value in the count, from `rv`. The count is
      still a feature: it changes the odds of each event (hitters change their
      approach), while the table supplies what each event is worth in that
      count. Rare events (triples, errors) are expected to be predicted near
      their base rates.

    The take model and the whiff / foul / in-play classifier are fitted
    identically either way, so two fits that differ only in `in_play` differ
    only in the value of a ball in play.
    """
    if in_play not in ('regression', 'classifier'):
        raise ValueError(f"in_play must be 'regression' or 'classifier', not {in_play!r}")
    features = list(features)
    unknown = [f for f in features if f in SWING_ONLY_FEATURES - CONTACT_PRIORS - DAMAGE_PRIORS]
    assert not unknown, f'no routing rule for {unknown}'
    take_features = [f for f in features if f not in SWING_ONLY_FEATURES]
    outcome_features = [f for f in features if f not in DAMAGE_PRIORS]
    value_features = [f for f in features if f not in CONTACT_PRIORS]

    take = fit_take(train, take_features, target)

    sw = train[train['swing']]
    y = swing_class(sw['outcome'])
    assert (y >= 0).all(), 'a swing row has no swing outcome class'
    outcome = train_early_stopped(sw, outcome_features, y, CLASSIFIER)

    bip = sw[y == 2]
    rv = rv.copy()
    rv.index = rv.index.set_levels(rv.index.levels[1].astype(str), level=1)
    re_in_play = None
    if in_play == 'regression':
        bip_value = train_early_stopped(bip, value_features, bip[target], REGRESSION)
    else:
        bip_value = train_early_stopped(bip, value_features, in_play_class(bip['outcome']),
                                        IN_PLAY_CLASSIFIER)
        re_in_play = rv.unstack('outcome').reindex(columns=list(IN_PLAY_EVENTS))
        # An event that never happened in some count in the training seasons
        # (a triple on 3-0, say) takes its average value over all counts; the
        # classifier gives it a tiny probability there anyway.
        re_in_play = re_in_play.fillna(re_in_play.mean())
        assert re_in_play.notna().all().all()

    return DecomposedModels(
        take=take, outcome=outcome, bip_value=bip_value,
        take_features=take_features, outcome_features=outcome_features,
        value_features=value_features,
        re_whiff=rv.xs('swinging_strike', level=0), re_foul=rv.xs('foul', level=0),
        rounds={'take': take.num_boosted_rounds(), 'outcome': outcome.num_boosted_rounds(),
                'bip_value': bip_value.num_boosted_rounds()},
        in_play=in_play, re_in_play=re_in_play)


# --------------------------------------------------------------------------
# Saving and loading a fitted model
# --------------------------------------------------------------------------

_BOOSTERS = ('take', 'outcome', 'bip_value')


def save_models(models: DecomposedModels, path, meta: dict | None = None) -> Path:
    """Write a fitted decomposed model to `path`: one XGBoost file per booster,
    the run-value lookups as parquet, and a `meta.json` describing the rest.

    `meta` is stored alongside (training seasons, date, ...); it is not needed
    to load the model.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    for name in _BOOSTERS:
        getattr(models, name).save_model(path / f'{name}.ubj')
    pd.DataFrame({'whiff': models.re_whiff, 'foul': models.re_foul}).to_parquet(path / 're_swing.parquet')
    if models.re_in_play is not None:
        models.re_in_play.to_parquet(path / 're_in_play.parquet')
    info = {'take_features': models.take_features, 'outcome_features': models.outcome_features,
            'value_features': models.value_features, 'in_play': models.in_play,
            'rounds': models.rounds, 'meta': meta or {}}
    (path / 'meta.json').write_text(json.dumps(info, indent=2, default=str))
    return path


def load_models(path) -> DecomposedModels:
    """Rebuild a `DecomposedModels` written by `save_models`."""
    path = Path(path)
    info = json.loads((path / 'meta.json').read_text())
    boosters = {}
    for name in _BOOSTERS:
        b = xgb.Booster()
        b.load_model(path / f'{name}.ubj')
        boosters[name] = b
    re_swing = pd.read_parquet(path / 're_swing.parquet')
    re_in_play = (pd.read_parquet(path / 're_in_play.parquet')
                  if (path / 're_in_play.parquet').exists() else None)
    return DecomposedModels(
        take=boosters['take'], outcome=boosters['outcome'], bip_value=boosters['bip_value'],
        take_features=info['take_features'], outcome_features=info['outcome_features'],
        value_features=info['value_features'],
        re_whiff=re_swing['whiff'], re_foul=re_swing['foul'], rounds=info['rounds'],
        in_play=info['in_play'], re_in_play=re_in_play)
