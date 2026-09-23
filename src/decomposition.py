"""The decomposed swing model (Track B).

A swing ends one of three ways, and the target is the run value of
(outcome, count), so the value of swinging is exactly

    Q_swing(s) = P(whiff | s) * RE(whiff, c)
               + P(foul  | s) * RE(foul, c)
               + P(BIP   | s) * E[target | BIP, s]

The direct twin estimates the same conditional mean in one regression. The
decomposition estimates it in pieces: a three-class classifier for what the
swing produces, and a regressor for what a ball in play is worth. Whiffs are
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

from dataclasses import dataclass

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


def swing_class(outcome: pd.Series) -> np.ndarray:
    """0 = whiff (incl. foul tips), 1 = foul, 2 = ball in play; -1 if not a swing outcome."""
    out = np.full(len(outcome), -1)
    o = outcome.astype(str).to_numpy()
    out[o == 'swinging_strike'] = 0
    out[o == 'foul'] = 1
    out[np.isin(o, BIP_OUTCOMES)] = 2
    return out


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

    def components(self, df: pd.DataFrame) -> pd.DataFrame:
        """The pieces of `Q_swing` for every row: class probabilities and ball-in-play value."""
        p = self.outcome.predict(_dmatrix(df, self.outcome_features))
        return pd.DataFrame({'p_whiff': p[:, 0], 'p_foul': p[:, 1], 'p_bip': p[:, 2],
                             'bip_value': self.bip_value.predict(_dmatrix(df, self.value_features))},
                            index=df.index)

    def predict(self, df: pd.DataFrame, action: str) -> np.ndarray:
        if action == 'take':
            return self.take.predict(_dmatrix(df, self.take_features))
        c = self.components(df)
        count = df['count'].astype(str)
        return (c['p_whiff'].to_numpy() * count.map(self.re_whiff).to_numpy()
                + c['p_foul'].to_numpy() * count.map(self.re_foul).to_numpy()
                + c['p_bip'].to_numpy() * c['bip_value'].to_numpy())


def fit_decomposed(train: pd.DataFrame, features: list[str], rv: pd.Series,
                   target: str = 'target', **_) -> DecomposedModels:
    """Fit the decomposed twin on one fold's training rows.

    `features` is the same list the direct twin gets. The take model sees the
    pitch features only; the classifier adds any contact priors; the
    ball-in-play regressor adds any damage priors. `rv` is the fold's
    (outcome, count) run-value table -- the one that defines `target`.
    """
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
    bip_value = train_early_stopped(bip, value_features, bip[target], REGRESSION)

    rv = rv.copy()
    rv.index = rv.index.set_levels(rv.index.levels[1].astype(str), level=1)
    return DecomposedModels(
        take=take, outcome=outcome, bip_value=bip_value,
        take_features=take_features, outcome_features=outcome_features,
        value_features=value_features,
        re_whiff=rv.xs('swinging_strike', level=0), re_foul=rv.xs('foul', level=0),
        rounds={'take': take.num_boosted_rounds(), 'outcome': outcome.num_boosted_rounds(),
                'bip_value': bip_value.num_boosted_rounds()})
