"""Evaluation harness for swing-decision metrics.

Every variant is scored by these functions, so the results table in
FINDINGS.md compares like with like. Three of the choices here are not obvious:

* **Held-out scoring only.** Every variant is fitted on rolling-origin folds
  (`GENERIC_FOLDS`, `PERSONALIZED_FOLDS`) and each validation season is scored
  by the one fold model that did not train on it. The player-metric checks see
  those scores alone: split-half within 2023, 2024 and 2025, year-over-year and
  next-season validity on 2023->24 and 2024->25. `run_folds` refuses 2026:
  it is kept out of model selection. (It is not a clean test -- earlier
  versions scored it -- so it is reported at the end as an out-of-regime
  evaluation. The confirmatory test is 2027; once it is complete, 2026 becomes
  ordinary data -- a training season and a held-out fold -- and this guard
  moves to 2027.)
* **Models are selected by paired held-out accuracy.** `paired_accuracy`
  compares two variants' squared error on the same pitches. Outcome luck adds
  the same amount to both, so the difference is the difference in how far
  each model is from the true expected run value -- even though a swing's MSE
  level is dominated by that luck. Calibration (`prediction_calibration`,
  with a game-resampled interval) then checks the chosen model, since the
  score uses each edge's magnitude; a miscalibrated winner is recalibrated,
  never rejected. The player-metric checks are guardrails: they show the
  score is stable and plausible, not that a model is right.
* Zone% correlation is a first-class output. Salorio retired SOTO after finding
  Zone% explained ~23% of its variance, i.e. it was substantially measuring the
  pitches a hitter was thrown rather than his decisions. Reliability cannot
  catch that: a metric of the wrong quantity can be perfectly stable.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xgboost as xgb

from . import data as D
from . import decision as DEC
from .features import BIP_OUTCOMES
from .baselines import (SEED, ActionModels, _dmatrix, fit_v1, predict_both, predict_chosen,
                        train_early_stopped)

QUALIFY_PITCHES = 500

# --------------------------------------------------------------------------
# Folds
# --------------------------------------------------------------------------

#: (training seasons, held-out season). Generic variants can train from 2021.
GENERIC_FOLDS = (((2021, 2022), 2023),
                 ((2021, 2022, 2023), 2024),
                 ((2021, 2022, 2023, 2024), 2025))

#: Personalized variants need a prior season to build hitter features from, so
#: 2021 is prior data only. Whenever a generic and a personalized variant are
#: compared, both use these folds, or the comparison varies the training data
#: along with the feature.
PERSONALIZED_FOLDS = (((2022,), 2023),
                      ((2022, 2023), 2024),
                      ((2022, 2023, 2024), 2025))

HELD_OUT = (2023, 2024, 2025)

#: Kept out of model selection. The first ABS season: scored at the end as an
#: out-of-regime evaluation. Not a clean test -- earlier versions of the
#: project scored and inspected it -- so a confirmatory test needs a later
#: season.
EXCLUDED_SEASON = 2026


@dataclass
class FoldRun:
    """Held-out scores from one variant, plus what produced them."""
    held: pd.DataFrame                       # every held-out pitch, scored by its fold
    models: dict[int, ActionModels]          # keyed by held-out season
    folds: tuple
    in_sample: pd.DataFrame | None = field(default=None, repr=False)


def run_folds(df: pd.DataFrame, features: list[str], folds=GENERIC_FOLDS,
              in_sample: bool = False, fit=fit_v1, **fit_kwargs) -> FoldRun:
    """Fit a variant on each fold and score only the season it held out.

    Per fold, everything that is learned is learned from the training seasons:
    the (outcome, count) run-value table that defines the target, both action
    models, and the count-only reference predictor used by `bin_calibration`.

    Returns the held-out seasons concatenated, with `q_take`, `q_swing`,
    `edge`, `y_pred` (the chosen action's value) and every per-pitch score in
    `decision.SCORES`. With `in_sample=True` the last fold's model also scores
    its own training seasons, kept apart in `.in_sample` so it can be reported
    beside the held-out figures but never averaged into them.

    `fit` is the model-fitting function, `fit_v1` by default. Any function
    returning an object with `.predict(df, action)` works; if it takes an `rv`
    argument it receives the fold's run-value table, and if the fitted object
    has `.components(df)` those columns are attached to the scored rows.
    """
    need = [f for f in features if f != 'count']
    wants_rv = 'rv' in inspect.signature(fit).parameters

    def scored(frame, model, count_only):
        out = DEC.add_scores(predict_both(predict_chosen(frame, model), model))
        idx = pd.MultiIndex.from_arrays([out['swing'], out['count']])
        out['count_only'] = count_only.reindex(idx).to_numpy()
        if hasattr(model, 'components'):
            out = out.join(model.components(frame))
        return out

    frames, models = [], {}
    for train_seasons, valid in folds:
        train_seasons = tuple(train_seasons)
        if EXCLUDED_SEASON in (*train_seasons, valid):
            raise ValueError(f'{EXCLUDED_SEASON} is kept out of model selection; it has no place in a fold')
        assert valid not in train_seasons and max(train_seasons) < valid

        rv = D.run_value_table(df, train_seasons)
        part = df[df['season'].isin([*train_seasons, valid])].dropna(subset=need)
        part = D.apply_run_value(part, rv, name='target').dropna(subset=['target'])
        train = part[part['season'].isin(train_seasons)]

        model = fit(train, features=features, **({'rv': rv} if wants_rv else {}), **fit_kwargs)
        count_only = train.groupby(['swing', 'count'], observed=True)['target'].mean()

        held = scored(part[part['season'] == valid], model, count_only)
        held['fold'] = valid
        frames.append(held)
        models[valid] = model

    run = FoldRun(held=pd.concat(frames, ignore_index=True), models=models, folds=tuple(folds))
    if in_sample:
        train_seasons, valid = folds[-1]
        rv = D.run_value_table(df, train_seasons)
        part = df[df['season'].isin(train_seasons)].dropna(subset=need)
        part = D.apply_run_value(part, rv, name='target').dropna(subset=['target'])
        count_only = part.groupby(['swing', 'count'], observed=True)['target'].mean()
        run.in_sample = scored(part, models[valid], count_only)
    return run


# --------------------------------------------------------------------------
# Sub-model performance
# --------------------------------------------------------------------------

def rmse_vs_count_baseline(held: pd.DataFrame) -> pd.DataFrame:
    """Pitch-level RMSE per action against the fold's count-only lookup.

    Reported for the take model, where it is informative. For the swing model it
    is the wrong instrument -- see `bin_calibration`.
    """
    rows = []
    for action, mask in (('take', ~held['swing']), ('swing', held['swing'])):
        g = held.loc[mask]
        y = g['target'].to_numpy()
        pred = g['q_swing' if action == 'swing' else 'q_take'].to_numpy()
        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
        rmse_base = float(np.sqrt(np.mean((g['count_only'].to_numpy() - y) ** 2)))
        rows.append({'action': action, 'n': len(g), 'rmse': rmse,
                     'rmse_count_only': rmse_base,
                     'improvement_%': (1 - rmse / rmse_base) * 100})
    return pd.DataFrame(rows).set_index('action')


#: Grid for the conditional-mean check, in the batter frame.
BIN_X = np.linspace(-1.2, 1.2, 13)
BIN_Z = np.linspace(-0.4, 1.6, 13)
MIN_BIN = 200


#: Game-resampling draws for the calibration-slope intervals.
BOOT_DRAWS = 200


def _slope_interval(s: pd.DataFrame, bin_key: np.ndarray, obs: str, pred: str,
                    min_n: int, draws: int = BOOT_DRAWS, seed: int = SEED) -> tuple[float, float]:
    """95% interval for the binned calibration slope, resampling whole games.

    Games, not pitches, are the resampling unit, since pitches within a game
    share a park, a starter and conditions. Per-(game, bin) sums are built once;
    a draw is then a vector of game weights, so each resample is a matrix
    product rather than a regrouping of every pitch.
    """
    game = pd.factorize(s['game_pk'])[0]
    ng, nb = game.max() + 1, bin_key.max() + 1

    def sums(v):
        m = np.zeros((ng, nb)); np.add.at(m, (game, bin_key), v); return m

    N, O, P = sums(np.ones(len(s))), sums(s[obs].to_numpy()), sums(s[pred].to_numpy())
    rng = np.random.default_rng(seed)
    out = []
    for w in rng.multinomial(ng, np.full(ng, 1 / ng), size=draws):
        n, o, p = w @ N, w @ O, w @ P
        ok = n >= min_n
        out.append(np.polyfit(p[ok] / n[ok], o[ok] / n[ok], 1, w=np.sqrt(n[ok]))[0])
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(lo), float(hi)


def bin_calibration(held: pd.DataFrame, action: str = 'swing',
                    min_n: int = MIN_BIN, ci: bool = False) -> pd.DataFrame:
    """Score an action model on the conditional means it estimates, per held-out season.

    Pitches are binned by (batter-frame location x count). In each bin with at
    least `min_n` pitches the observed mean target is compared with the mean
    prediction, and with the count-only predictor from the fold's training
    seasons. Columns:

    * `err`, `err_count_only` -- pitch-weighted RMS error of the bin means.
    * `noise` -- the error a perfect predictor would still show, because each
      observed bin mean is itself an estimate (sqrt of mean within-bin
      variance / n). Neither error can go below it.
    * `signal_recovered` -- `1 - (err^2 - noise^2) / (err_count_only^2 - noise^2)`:
      of the between-bin structure the count-only predictor misses, the share
      the model captures. Tied to this one binning, so compare variants on it,
      don't quote it as a ceiling.
    * `slope` -- observed bin mean regressed on predicted, weighted by n. A
      model can correlate well with the bin means and still be systematically
      over- or under-confident; 1 is calibrated, below 1 overstates contrasts.
      With `ci=True`, `slope_lo`/`slope_hi` give a 95% interval from resampling
      games.

    It groups by location and count only, so it cannot credit a feature that
    varies within a bin (pitch characteristics, hitter priors).
    `prediction_calibration` is the check that covers the whole model.
    """
    swung = action == 'swing'
    col = 'q_swing' if swung else 'q_take'
    g = held[held['swing'] == swung]
    g = g.assign(cx=pd.cut(g['plate_x_bat'], BIN_X, labels=False),
                 cz=pd.cut(g['plate_z_norm'], BIN_Z, labels=False)).dropna(subset=['cx', 'cz'])
    rows = []
    for season, s in g.groupby('season', observed=True):
        agg = (s.groupby(['cx', 'cz', 'count'], observed=True)
                .agg(obs=('target', 'mean'), pred=(col, 'mean'), base=('count_only', 'mean'),
                     var=('target', 'var'), n=('target', 'size')))
        agg = agg[agg['n'] >= min_n]
        w = agg['n'].to_numpy()

        def wms(a):
            return float(np.average(a ** 2, weights=w))

        mse, mse_base = wms(agg['pred'] - agg['obs']), wms(agg['base'] - agg['obs'])
        noise = float(np.average(agg['var'] / agg['n'], weights=w))
        slope = float(np.polyfit(agg['pred'], agg['obs'], 1, w=np.sqrt(w))[0])
        row = {'season': season, 'bins': len(agg), 'pitches': int(w.sum()),
               'err': np.sqrt(mse), 'err_count_only': np.sqrt(mse_base),
               'noise': np.sqrt(noise),
               'signal_recovered': 1 - (mse - noise) / (mse_base - noise),
               'slope': slope}
        if ci:
            key = s.groupby(['cx', 'cz', 'count'], observed=True).ngroup().to_numpy()
            row['slope_lo'], row['slope_hi'] = _slope_interval(s, key, 'target', col, min_n)
        rows.append(row)
    return pd.DataFrame(rows).set_index('season')


def prediction_calibration(held: pd.DataFrame, action: str = 'swing', bins: int = 20,
                           ci: bool = True) -> pd.DataFrame:
    """The standard calibration check: group by *predicted* value, per held-out season.

    Pitches land in the same bin because the model rates them alike, for
    whatever reason -- location, count, pitch type or hitter -- so unlike
    `bin_calibration` it tests the whole model. `slope` regresses the observed
    mean target on the mean prediction across the `bins` quantile bins,
    weighted by n: 1 is calibrated, above 1 means the predictions are too
    compressed, below 1 too spread. With `ci`, a 95% interval from resampling
    games.
    """
    swung = action == 'swing'
    col = 'q_swing' if swung else 'q_take'
    g = held[held['swing'] == swung]
    rows = []
    for season, s in g.groupby('season', observed=True):
        key = pd.qcut(s[col].rank(method='first'), bins, labels=False).to_numpy()
        agg = s.groupby(key).agg(obs=('target', 'mean'), pred=(col, 'mean'), n=('target', 'size'))
        slope, icpt = np.polyfit(agg['pred'], agg['obs'], 1, w=np.sqrt(agg['n']))
        row = {'season': season, 'bins': len(agg), 'pitches': int(agg['n'].sum()),
               'pred_range': float(agg['pred'].max() - agg['pred'].min()),
               'obs_range': float(agg['obs'].max() - agg['obs'].min()),
               'slope': float(slope), 'intercept': float(icpt)}
        if ci:
            row['slope_lo'], row['slope_hi'] = _slope_interval(s, key, 'target', col, min_n=1)
        rows.append(row)
    return pd.DataFrame(rows).set_index('season')


def outcome_diagnostics(held: pd.DataFrame, bins: int = 20) -> pd.DataFrame:
    """How well the decomposed classifier predicts what a swing produces, per held-out season.

    Needs `p_whiff`/`p_foul`/`p_bip` from `DecomposedModels.components`. Log
    loss is against each season's own class frequencies, a base rate slightly
    better informed than the model's training rates, so the comparison is
    conservative. Per-class slopes bin swings by predicted probability and
    regress the observed rate on it; 1 is calibrated.
    """
    from sklearn.metrics import log_loss, roc_auc_score
    from .decomposition import SWING_CLASSES, swing_class

    sw = held[held['swing']]
    rows = []
    for season, s in sw.groupby('season', observed=True):
        y = swing_class(s['outcome'])
        p = s[['p_whiff', 'p_foul', 'p_bip']].to_numpy()
        base = np.bincount(y, minlength=3) / len(y)
        row = {'season': season, 'swings': len(s),
               'logloss': log_loss(y, p, labels=[0, 1, 2]),
               'logloss_base_rate': log_loss(y, np.tile(base, (len(y), 1)), labels=[0, 1, 2]),
               'whiff_auc': roc_auc_score(y == 0, p[:, 0])}
        row['improvement_%'] = (1 - row['logloss'] / row['logloss_base_rate']) * 100
        for k, name in enumerate(SWING_CLASSES):
            key = pd.qcut(pd.Series(p[:, k]).rank(method='first'), bins, labels=False).to_numpy()
            obs = np.bincount(key, weights=(y == k)) / np.bincount(key)
            pred = np.bincount(key, weights=p[:, k]) / np.bincount(key)
            row[f'{name}_slope'] = float(np.polyfit(pred, obs, 1)[0])
        rows.append(row)
    return pd.DataFrame(rows).set_index('season')


def in_play_diagnostics(held: pd.DataFrame, bins: int = 20) -> pd.DataFrame:
    """How well the in-play classifier predicts what a ball in play becomes.

    Needs the `p_<event>` columns from a decomposed model fitted with
    `in_play='classifier'`. Per held-out season, on balls in play: multiclass
    log loss against that season's own event frequencies, and for each event
    the mean predicted against the observed rate and a calibration slope
    (observed rate on predicted, over `bins` quantile bins; 1 is calibrated).
    Rare events are expected to sit near their base rate, with little spread
    to calibrate.
    """
    from sklearn.metrics import log_loss
    from .decomposition import IN_PLAY_EVENTS, in_play_class

    bip = held[held['outcome'].astype(str).isin(BIP_OUTCOMES)]
    cols = [f'p_{e}' for e in IN_PLAY_EVENTS]
    rows = []
    for season, s in bip.groupby('season', observed=True):
        y = in_play_class(s['outcome'])
        p = s[cols].to_numpy()
        base = np.bincount(y, minlength=len(IN_PLAY_EVENTS)) / len(y)
        labels = list(range(len(IN_PLAY_EVENTS)))
        ll = log_loss(y, p, labels=labels)
        ll0 = log_loss(y, np.tile(base, (len(y), 1)), labels=labels)
        for k, event in enumerate(IN_PLAY_EVENTS):
            key = pd.qcut(pd.Series(p[:, k]).rank(method='first'), bins, labels=False).to_numpy()
            obs = np.bincount(key, weights=(y == k)) / np.bincount(key)
            pr = np.bincount(key, weights=p[:, k]) / np.bincount(key)
            rows.append({'season': season, 'event': event, 'balls in play': len(s),
                         'observed rate': (y == k).mean(), 'mean predicted': p[:, k].mean(),
                         'predicted spread (sd)': p[:, k].std(),
                         'calibration slope': float(np.polyfit(pr, obs, 1)[0]),
                         'logloss improvement % (all events)': (1 - ll / ll0) * 100})
    return pd.DataFrame(rows).set_index(['season', 'event'])


BINARY = {'objective': 'binary:logistic', 'eval_metric': 'logloss'}


def whiff_auc(df: pd.DataFrame, features: list[str], folds=GENERIC_FOLDS) -> pd.DataFrame:
    """Held-out whiff prediction on swings: AUC and log loss against the base rate.

    Whether a swing misses is far more predictable than what it is worth, which
    is the premise of decomposing the swing into events. Same folds and the
    same early-stopping learner as everything else.
    """
    from sklearn.metrics import log_loss, roc_auc_score

    need = [f for f in features if f != 'count']
    sw = df[df['swing']].dropna(subset=need)
    whiff = (sw['outcome'] == 'swinging_strike').astype(int)
    rows = []
    for train_seasons, valid in folds:
        if EXCLUDED_SEASON in (*train_seasons, valid):
            raise ValueError(f'{EXCLUDED_SEASON} is kept out of model selection; it has no place in a fold')
        tr, va = sw['season'].isin(train_seasons), sw['season'] == valid
        booster = train_early_stopped(sw[tr], features, whiff[tr], BINARY)
        p = booster.predict(_dmatrix(sw[va], features))
        base = log_loss(whiff[va], np.full(va.sum(), whiff[tr].mean()))
        ll = log_loss(whiff[va], p)
        rows.append({'season': valid, 'swings': int(va.sum()),
                     'auc': roc_auc_score(whiff[va], p), 'logloss': ll,
                     'logloss_base_rate': base, 'improvement_%': (1 - ll / base) * 100})
    return pd.DataFrame(rows).set_index('season')


# --------------------------------------------------------------------------
# Accuracy -- the step that selects a model
# --------------------------------------------------------------------------

#: A pitch's identity across runs.
PITCH_KEY = ['game_pk', 'at_bat_number', 'pitch_number']


def _game_mean_interval(games: pd.Series, values: np.ndarray, draws: int = BOOT_DRAWS,
                        seed: int = SEED) -> tuple[float, float]:
    """95% interval for the mean of `values`, resampling whole games."""
    g = pd.factorize(games)[0]
    ng = g.max() + 1
    total = np.bincount(g, weights=values, minlength=ng)
    n = np.bincount(g, minlength=ng).astype(float)
    w = np.random.default_rng(seed).multinomial(ng, np.full(ng, 1 / ng), size=draws)
    lo, hi = np.percentile((w @ total) / (w @ n), [2.5, 97.5])
    return float(lo), float(hi)


def _q(action: str) -> str:
    return 'q_swing' if action == 'swing' else 'q_take'


def accuracy(run: FoldRun) -> pd.DataFrame:
    """Held-out MSE per action against the fold's count-only predictor.

    `diff` is model MSE minus count-only MSE (negative = the model is closer
    to the truth), with a game-resampled interval. Per held-out season and
    pooled. The level of swing MSE is dominated by outcome luck; the
    difference is not, because that luck adds the same amount to both.
    """
    rows = []
    for action in ('take', 'swing'):
        g = run.held[run.held['swing'] == (action == 'swing')]
        for season, s in [*g.groupby('season', observed=True), ('pooled', g)]:
            y = s['target'].to_numpy()
            se_m, se_c = (s[_q(action)].to_numpy() - y) ** 2, (s['count_only'].to_numpy() - y) ** 2
            lo, hi = _game_mean_interval(s['game_pk'], se_m - se_c)
            rows.append({'action': action, 'season': season, 'pitches': len(s),
                         'mse': se_m.mean(), 'mse_count_only': se_c.mean(),
                         'improvement_%': (1 - se_m.mean() / se_c.mean()) * 100,
                         'diff': se_m.mean() - se_c.mean(), 'diff_lo': lo, 'diff_hi': hi})
    return pd.DataFrame(rows).set_index(['action', 'season'])


def paired_accuracy(run_a: FoldRun, run_b: FoldRun, labels=('a', 'b'),
                    rows: str = 'all') -> pd.DataFrame:
    """Held-out MSE of two variants on the same pitches: the model-selection test.

    Outcome noise adds the same amount to both models' squared error, so the
    paired difference `mse_a - mse_b` is the difference in how far each model
    is from the true expected run value. Negative = `a` is better. The
    interval resamples games. `diff_%` expresses the difference (and its
    interval) as a percentage of `b`'s MSE, which is easier to read than
    squared runs. Both runs must use the same folds, so their
    targets are the same.

    `rows='in_play'` scores only the decomposed model's in-play branch: its
    `bip_value` against the target, on balls in play. That isolates a change
    to how a ball in play is valued, where the swing-level comparison dilutes
    it with whiffs and fouls.
    """
    if rows not in ('all', 'in_play'):
        raise ValueError(f"rows must be 'all' or 'in_play', not {rows!r}")
    pred = {'take': 'q_take', 'swing': 'q_swing', 'in play': 'bip_value'}
    cols = PITCH_KEY + ['season', 'swing', 'target', 'outcome'] + (
        ['q_take', 'q_swing'] if rows == 'all' else ['bip_value'])
    m = run_a.held[cols].merge(run_b.held[cols].drop(columns='outcome'), on=PITCH_KEY, suffixes=('_a', '_b'))
    assert np.allclose(m['target_a'], m['target_b']), 'runs use different folds or targets'
    la, lb = labels
    out = []
    groups = ([('take', ~m['swing_a']), ('swing', m['swing_a'])] if rows == 'all'
              else [('in play', m['outcome'].astype(str).isin(BIP_OUTCOMES))])
    for action, mask in groups:
        g = m[mask]
        for season, s in [*g.groupby('season_a', observed=True), ('pooled', g)]:
            y = s['target_a'].to_numpy()
            se_a = (s[f'{pred[action]}_a'].to_numpy() - y) ** 2
            se_b = (s[f'{pred[action]}_b'].to_numpy() - y) ** 2
            lo, hi = _game_mean_interval(s['game_pk'], se_a - se_b)
            d = se_a.mean() - se_b.mean()
            out.append({'action': action, 'season': season, 'pitches': len(s),
                         f'mse {la}': se_a.mean(), f'mse {lb}': se_b.mean(),
                         'diff': d, 'diff_lo': lo, 'diff_hi': hi,
                         'diff_%': d / se_b.mean() * 100,
                         'diff_%_lo': lo / se_b.mean() * 100, 'diff_%_hi': hi / se_b.mean() * 100,
                         'verdict': (f'{la} better' if hi < 0 else f'{lb} better' if lo > 0 else 'tie')})
    return pd.DataFrame(out).set_index(['action', 'season'])


def calibration_table(run: FoldRun) -> pd.DataFrame:
    """Prediction-grouped calibration slope with interval, both actions, per season."""
    return pd.concat({a: prediction_calibration(run.held, a)[['slope', 'slope_lo', 'slope_hi']]
                      for a in ('take', 'swing')}, names=['action'])


def recalibrate_last_season(run: FoldRun) -> tuple[FoldRun, pd.DataFrame]:
    """Recalibrate each held-out season with a linear map learned one season earlier.

    For season s the map `q -> a + b*q` (per action) is fitted on the previous
    fold's held-out season s-1: predictions made one season ahead, compared
    with what happened. That is the same kind of shift the map has to correct,
    and it uses no data the fold for s did not already train on. The first
    held-out season has no earlier one and is left as fitted.

    Returns the recalibrated run (edges and every decision score recomputed)
    and the table of maps. Maps fitted within the training seasons -- e.g. on
    early-stopping games -- do not work here, because the residual
    miscalibration is a between-season shift they cannot see.
    """
    held = run.held.copy()
    seasons = sorted(held['season'].unique())
    maps = []
    for prev, cur in zip(seasons[:-1], seasons[1:]):
        p = held[held['season'] == prev]
        for action in ('take', 'swing'):
            g = p[p['swing'] == (action == 'swing')]
            b, a = np.polyfit(g[_q(action)], g['target'], 1)
            maps.append({'season': cur, 'action': action, 'fitted on': prev, 'a': a, 'b': b})
    maps = pd.DataFrame(maps)
    for m in maps.itertuples():
        # Each action's Q is recalibrated on every pitch of the season, not only
        # where that action was taken: the score uses both on every pitch.
        col = _q(m.action)
        season_rows = held['season'] == m.season
        held.loc[season_rows, col] = m.a + m.b * held.loc[season_rows, col]
    held['edge'] = held['q_swing'] - held['q_take']
    held['y_pred'] = np.where(held['swing'], held['q_swing'], held['q_take'])
    held = DEC.add_scores(held)
    return FoldRun(held=held, models=run.models, folds=run.folds), maps.set_index(['season', 'action'])


def swing_propensity(df: pd.DataFrame, features: list[str], folds) -> pd.Series:
    """League `P(swing | pitch)` for every held-out pitch, fitted per fold on its
    training seasons. Indexed by `PITCH_KEY`."""
    need = [f for f in features if f != 'count']
    d = df.dropna(subset=need)
    parts = []
    for train_seasons, valid in folds:
        if EXCLUDED_SEASON in (*train_seasons, valid):
            raise ValueError(f'{EXCLUDED_SEASON} is kept out of model selection; it has no place in a fold')
        tr, va = d[d['season'].isin(train_seasons)], d[d['season'] == valid]
        booster = train_early_stopped(tr, features, tr['swing'].astype(int), BINARY)
        parts.append(pd.Series(booster.predict(_dmatrix(va, features)),
                               index=pd.MultiIndex.from_frame(va[PITCH_KEY])))
    return pd.concat(parts).rename('p_swing')


PROPENSITY_BANDS = [0, 0.1, 0.3, 0.7, 0.9, 1.0]


def propensity_calibration(run: FoldRun, p_swing: pd.Series, bins: int = 10) -> pd.DataFrame:
    """Calibration within bands of league swing propensity, pooled over held-out seasons.

    `Q_swing` is used on every pitch, including the many a hitter took, but it
    can only be checked on pitches someone swung at. Swings at pitches the
    league usually takes (low propensity) are the closest available evidence
    that it holds up where it is extrapolated. Takes are checked the same way
    at high propensity.
    """
    h = run.held.join(p_swing, on=PITCH_KEY)
    h['band'] = pd.cut(h['p_swing'], PROPENSITY_BANDS, include_lowest=True)
    rows = []
    for action in ('swing', 'take'):
        g = h[h['swing'] == (action == 'swing')]
        for band, s in g.groupby('band', observed=True):
            if len(s) < 2000:
                continue
            key = pd.qcut(s[_q(action)].rank(method='first'), bins, labels=False).to_numpy()
            agg = s.groupby(key).agg(obs=('target', 'mean'), pred=(_q(action), 'mean'), n=('target', 'size'))
            slope = float(np.polyfit(agg['pred'], agg['obs'], 1, w=np.sqrt(agg['n']))[0])
            lo, hi = _slope_interval(s, key, 'target', _q(action), min_n=1)
            rows.append({'action': action, 'league swing propensity': str(band), 'pitches': len(s),
                         'mean pred': s[_q(action)].mean(), 'mean obs': s['target'].mean(),
                         'slope': slope, 'slope_lo': lo, 'slope_hi': hi})
    return pd.DataFrame(rows).set_index(['action', 'league swing propensity'])


# --------------------------------------------------------------------------
# Player metric
# --------------------------------------------------------------------------

def player_metric(df: pd.DataFrame, value_col: str = 'y_pred',
                  min_pitches: int = QUALIFY_PITCHES,
                  scale: bool = True) -> pd.DataFrame:
    """Per (season, batter) decision value: mean `value_col`, z-scored to 100 +- 10.

    Z-scoring is within season, so a score is always relative to that season's
    qualified field.
    """
    g = (df.groupby(['season', 'batter'], observed=True)[value_col]
           .agg(pitches='size', raw='mean').reset_index())
    g = g[g.pitches >= min_pitches].copy()
    g['raw'] = g['raw'].astype('float64')      # XGBoost predicts float32
    if scale:
        z = g.groupby('season')['raw'].transform(lambda s: (s - s.mean()) / s.std(ddof=0))
        g['decision_value'] = 100 + 10 * z
    else:
        g['decision_value'] = g['raw']
    return g


# --------------------------------------------------------------------------
# Reliability
# --------------------------------------------------------------------------

def yoy_reliability(scores: pd.DataFrame) -> pd.DataFrame:
    """R^2 between consecutive seasons' scores, over hitters qualified in both."""
    seasons = sorted(scores.season.unique())
    rows = []
    for a, b in zip(seasons[:-1], seasons[1:]):
        m = (scores[scores.season == a][['batter', 'decision_value']]
             .merge(scores[scores.season == b][['batter', 'decision_value']],
                    on='batter', suffixes=('_a', '_b')))
        if len(m) < 30:
            continue
        r = np.corrcoef(m.decision_value_a, m.decision_value_b)[0, 1]
        rows.append({'pair': f'{a}->{b}', 'hitters': len(m), 'r': r, 'r2': r ** 2})
    return pd.DataFrame(rows).set_index('pair')


def split_half(df: pd.DataFrame, value_col: str = 'y_pred',
               min_pitches: int = QUALIFY_PITCHES) -> pd.DataFrame:
    """Odd/even plate-appearance split-half reliability, Spearman-Brown corrected.

    Cleaner than year-over-year because it holds real skill change fixed: both
    halves come from the same season.
    """
    d = df.copy()
    d['half'] = (d['at_bat_number'] % 2).map({0: 'even', 1: 'odd'})
    rows = []
    for season, g in d.groupby('season', observed=True):
        halves = {h: player_metric(gg, value_col, min_pitches=min_pitches // 2, scale=False)
                  for h, gg in g.groupby('half', observed=True)}
        if len(halves) != 2:
            continue
        m = halves['odd'].merge(halves['even'], on='batter', suffixes=('_o', '_e'))
        keep = g.groupby('batter').size()
        m = m[m.batter.isin(keep[keep >= min_pitches].index)]
        if len(m) < 30:
            continue
        r = np.corrcoef(m.raw_o, m.raw_e)[0, 1]
        rows.append({'season': season, 'hitters': len(m), 'r_half': r,
                     'r_spearman_brown': 2 * r / (1 + r)})
    return pd.DataFrame(rows).set_index('season')


# --------------------------------------------------------------------------
# Validity
# --------------------------------------------------------------------------

def zone_pct_correlation(df: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """Correlation between the metric and Zone% -- the failure that retired SOTO.

    A metric substantially explained by the share of pitches a hitter sees in
    the zone is measuring his opportunities, not his decisions.
    """
    z = (df.assign(_in=D.in_rulebook_zone(df, ball_edge=True))
           .groupby(['season', 'batter'], observed=True)['_in'].mean()
           .rename('zone_pct').reset_index())
    m = scores.merge(z, on=['season', 'batter'])
    rows = []
    for season, g in m.groupby('season', observed=True):
        r = np.corrcoef(g.decision_value, g.zone_pct)[0, 1]
        rows.append({'season': season, 'hitters': len(g), 'r': r,
                     'variance_explained_%': r ** 2 * 100,
                     'zone_pct_std': g.zone_pct.std()})
    return pd.DataFrame(rows).set_index('season')


def _partial_corr(y: np.ndarray, x: np.ndarray, z: np.ndarray) -> float:
    """Correlation of y with x, holding z fixed, by residualising both on z."""
    ry = y - np.polyval(np.polyfit(z, y, 1), z)
    rx = x - np.polyval(np.polyfit(z, x, 1), z)
    return float(np.corrcoef(ry, rx)[0, 1])


def construct_validity(df: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """Does the metric behave the way swing-decision quality must?

    Two requirements, and a metric has to satisfy both: punish chasing, and
    reward attacking hittable pitches.

    **Report the partials.** Chase rate and zone-swing rate correlate about
    +0.5 across hitters, because aggression is a single dimension -- an
    aggressive hitter swings more at everything. Since the chase penalty is
    roughly three times the zone-swing bonus, a raw correlation against
    zone-swing is swamped by it and comes out near zero, which reads as though
    the metric ignores half the construct. It does not; holding chase rate
    fixed recovers a strong positive relationship. The raw columns are returned
    alongside so the confound stays visible, but the partials are the numbers
    to judge on.

    This is the check that actually discriminates between candidate scores.
    Reliability cannot: it rewards a metric that is stably wrong exactly as much
    as one that is stably right.
    """
    in_zone = D.in_rulebook_zone(df, ball_edge=True)
    keys = ['season', 'batter']
    behaviour = pd.DataFrame({
        'chase': df[~in_zone].groupby(keys, observed=True)['swing'].mean(),
        'zone_swing': df[in_zone].groupby(keys, observed=True)['swing'].mean(),
    }).reset_index()

    m = scores.merge(behaviour, on=keys).dropna(subset=['chase', 'zone_swing'])
    v, chase, zsw = (m['decision_value'].to_numpy(), m['chase'].to_numpy(),
                     m['zone_swing'].to_numpy())
    return pd.DataFrame([{
        'hitters': len(m),
        'chase (raw)': float(np.corrcoef(v, chase)[0, 1]),
        'zone_swing (raw)': float(np.corrcoef(v, zsw)[0, 1]),
        'chase | zone_swing': _partial_corr(v, chase, zsw),
        'zone_swing | chase': _partial_corr(v, zsw, chase),
        'corr(chase, zone_swing)': float(np.corrcoef(chase, zsw)[0, 1]),
    }])


def _production(df: pd.DataFrame) -> pd.DataFrame:
    """Offensive production per batter-season: delta run expectancy per PA.

    Taken from our own data rather than a linear-weights wOBA, so the measure
    is internally consistent with the modelling target and needs no external
    weight table maintained per season.

    A plate appearance is a distinct (game_pk, at_bat_number) pair.
    `at_bat_number` alone is the at-bat index *within a game*, so counting its
    distinct values over a season counts batting slots -- about 82 for a
    regular, against roughly 650 actual plate appearances -- and because that
    saturates near the number of available slots, the resulting rate would
    track playing time rather than production.
    """
    grouped = df.groupby(['season', 'batter'], observed=True)
    pa = grouped.apply(lambda g: g.groupby(['game_pk', 'at_bat_number']).ngroups,
                       include_groups=False).rename('pa')
    re = grouped['delta_run_exp'].sum()
    return (re / pa).rename('re_per_pa').reset_index()


def predictive_validity(df: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """Does season-t decision value predict season-t+1 production?

    Reported as a partial correlation controlling for season-t production: the
    question is whether the metric adds anything beyond knowing how well the
    hitter already hit.
    """
    prod = _production(df)
    s = scores.merge(prod, on=['season', 'batter'])
    seasons = sorted(s.season.unique())
    rows = []
    for a, b in zip(seasons[:-1], seasons[1:]):
        m = (s[s.season == a][['batter', 'decision_value', 're_per_pa']]
             .merge(s[s.season == b][['batter', 're_per_pa']], on='batter',
                    suffixes=('_t', '_t1')))
        if len(m) < 30:
            continue
        raw = np.corrcoef(m.decision_value, m.re_per_pa_t1)[0, 1]
        # partial correlation of (metric, next-season production) given this-season production
        def resid(y, x):
            return y - np.polyval(np.polyfit(x, y, 1), x)
        partial = np.corrcoef(resid(m.decision_value.to_numpy(), m.re_per_pa_t.to_numpy()),
                              resid(m.re_per_pa_t1.to_numpy(), m.re_per_pa_t.to_numpy()))[0, 1]
        rows.append({'pair': f'{a}->{b}', 'hitters': len(m),
                     'r_raw': raw, 'r_partial': partial})
    return pd.DataFrame(rows).set_index('pair')


# --------------------------------------------------------------------------
# Guardrail differences with intervals
# --------------------------------------------------------------------------

def _wcorr(x, y, w):
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    cov = np.average((x - mx) * (y - my), weights=w)
    return cov / np.sqrt(np.average((x - mx) ** 2, weights=w) * np.average((y - my) ** 2, weights=w))


def _wresid(y, controls, w):
    Z = np.column_stack([np.ones(len(y)), *controls])
    sw = np.sqrt(w)
    beta = np.linalg.lstsq(Z * sw[:, None], y * sw, rcond=None)[0]
    return y - Z @ beta


def _wpartial(y, x, controls, w):
    return _wcorr(_wresid(y, controls, w), _wresid(x, controls, w), w)


def metric_differences(run_a: FoldRun, run_b: FoldRun, value_col: str = DEC.DEFAULT_SCORE,
                       labels=('a', 'b'), draws: int = BOOT_DRAWS, seed: int = SEED) -> pd.DataFrame:
    """Paired differences in the guardrail checks, with hitter-resampled intervals.

    The same statistics as `metric_checks` -- construct partials, YoY R^2,
    Zone% |r|, next-season partial r -- for both variants on the same hitters,
    and the 95% interval of `a - b` from resampling hitters. Guardrails, not
    selectors: they flag a change for discussion, they do not choose a model.
    """
    key = ['season', 'batter']
    sa = player_metric(run_a.held, value_col)[key + ['decision_value']]
    sb = player_metric(run_b.held, value_col)[key + ['decision_value']]
    h = run_a.held.assign(_in=D.in_rulebook_zone(run_a.held, ball_edge=True))
    beh = pd.DataFrame({'chase': h[~h['_in']].groupby(key, observed=True)['swing'].mean(),
                        'zone_swing': h[h['_in']].groupby(key, observed=True)['swing'].mean(),
                        'zone_pct': h.groupby(key, observed=True)['_in'].mean()}).reset_index()
    t = (sa.merge(sb, on=key, suffixes=('_a', '_b')).merge(beh, on=key)
           .merge(_production(run_a.held), on=key).dropna())
    seasons = sorted(t['season'].unique())
    pairs = []
    for s0, s1 in zip(seasons[:-1], seasons[1:]):
        pairs.append(t[t['season'] == s0].merge(t[t['season'] == s1], on='batter', suffixes=('_t', '_t1')))
    batters = pd.Index(t['batter'].unique())

    def stats(weights: pd.Series) -> dict:
        out = {}
        w = t['batter'].map(weights).to_numpy()
        for v in ('a', 'b'):
            dv = t[f'decision_value_{v}'].to_numpy()
            out[('chase | zone_swing', v)] = _wpartial(dv, t['chase'].to_numpy(), [t['zone_swing'].to_numpy()], w)
            out[('zone_swing | chase', v)] = _wpartial(dv, t['zone_swing'].to_numpy(), [t['chase'].to_numpy()], w)
            out[('Zone% |r|', v)] = np.mean([abs(_wcorr(g[f'decision_value_{v}'].to_numpy(), g['zone_pct'].to_numpy(),
                                                        g['batter'].map(weights).to_numpy()))
                                            for _, g in t.groupby('season')])
            yoy, nxt = [], []
            for p in pairs:
                pw = p['batter'].map(weights).to_numpy()
                yoy.append(_wcorr(p[f'decision_value_{v}_t'].to_numpy(), p[f'decision_value_{v}_t1'].to_numpy(), pw) ** 2)
                nxt.append(_wpartial(p[f'decision_value_{v}_t'].to_numpy(), p['re_per_pa_t1'].to_numpy(),
                                     [p['re_per_pa_t'].to_numpy()], pw))
            out[('YoY R2 (mean)', v)] = np.mean(yoy)
            out[('next-season r (partial)', v)] = np.mean(nxt)
        return out

    base = stats(pd.Series(1.0, index=batters))
    rng = np.random.default_rng(seed)
    boots = [stats(pd.Series(rng.multinomial(len(batters), np.full(len(batters), 1 / len(batters))).astype(float),
                             index=batters)) for _ in range(draws)]
    la, lb = labels
    rows = []
    for stat in ['chase | zone_swing', 'zone_swing | chase', 'YoY R2 (mean)', 'Zone% |r|', 'next-season r (partial)']:
        d = [b[(stat, 'a')] - b[(stat, 'b')] for b in boots]
        lo, hi = np.percentile(d, [2.5, 97.5])
        rows.append({'check': stat, la: base[(stat, 'a')], lb: base[(stat, 'b')],
                     'diff': base[(stat, 'a')] - base[(stat, 'b')], 'diff_lo': lo, 'diff_hi': hi})
    return pd.DataFrame(rows).set_index('check')


def zone_contamination(held: pd.DataFrame, value_col: str = DEC.DEFAULT_SCORE) -> dict:
    """Zone% correlation with the score under four controls, averaged over seasons.

    Raw Zone% |r| is a lenient screen: better hitters are thrown fewer
    strikes, so a score that tracks hitter quality picks up a negative pull
    that can mask a positive bias. So the correlation is also reported holding
    fixed the correct-decision rate (from the same model's edge -- and
    circular for `correct_decision` itself), chase and zone-swing rates
    (model-free, but they vary with how hard the pitches a hitter sees are),
    and production (removing the hitter-quality pull). Every control is a
    screen, not proof.
    """
    key = ['season', 'batter']
    h = held.assign(_in=D.in_rulebook_zone(held, ball_edge=True), _right=held['correct_decision'] > 0)
    g = h.groupby(key, observed=True)
    hit = pd.DataFrame({'zone_pct': g['_in'].mean(), 'cd_rate': g['_right'].mean(),
                        'chase': h[~h['_in']].groupby(key, observed=True)['swing'].mean(),
                        'zone_swing': h[h['_in']].groupby(key, observed=True)['swing'].mean()}).reset_index()
    hit = hit.merge(_production(held), on=key)
    m = player_metric(held, value_col).merge(hit, on=key).dropna()
    controls = {'raw': [], '| correct-decision rate': ['cd_rate'],
                '| chase, zone-swing': ['chase', 'zone_swing'], '| production': ['re_per_pa']}
    out = {}
    for label, ctrl in controls.items():
        rs = []
        for _, t in m.groupby('season'):
            w = np.ones(len(t))
            rs.append(_wpartial(t['decision_value'].to_numpy(), t['zone_pct'].to_numpy(),
                                [t[c].to_numpy() for c in ctrl], w))
        out[f'Zone% r {label}'] = float(np.mean(rs))
    out['corr(Zone%, production)'] = float(np.corrcoef(hit['zone_pct'], hit['re_per_pa'])[0, 1])
    return out


# --------------------------------------------------------------------------
# One row for the scorecard
# --------------------------------------------------------------------------

#: The player-metric columns, in the order they should be weighed.
SHOW = ['chase | zone_swing', 'zone_swing | chase', 'split-half r', 'YoY R2 (mean)',
        'Zone% |r|', 'next-season r (partial)']

#: The model columns of the harness summary.
MODEL_COLS = ['swing MSE vs count-only %', 'swing pred slope', 'take MSE vs count-only %', 'take pred slope']


def metric_checks(df: pd.DataFrame, value_col: str) -> dict:
    """The player-metric checks alone, on whatever frame is passed."""
    scores = player_metric(df, value_col)
    cv = construct_validity(df, scores)
    sh = split_half(df, value_col)
    yoy = yoy_reliability(scores)
    zp = zone_pct_correlation(df, scores)
    pv = predictive_validity(df, scores)
    summary = {
        'chase | zone_swing': round(cv['chase | zone_swing'].iloc[0], 3),
        'zone_swing | chase': round(cv['zone_swing | chase'].iloc[0], 3),
        'split-half r': round(sh.r_spearman_brown.mean(), 3),
        'YoY R2 (mean)': round(yoy.r2.mean(), 3),
        'Zone% |r|': round(zp.r.abs().mean(), 3),
        'next-season r (partial)': round(pv.r_partial.mean(), 3),
    }
    return {'summary': summary, 'scores': scores, 'construct': cv, 'split_half': sh,
            'yoy': yoy, 'zone_pct': zp, 'predictive': pv}


def harness(run: FoldRun, label: str, value_col: str = DEC.DEFAULT_SCORE) -> dict:
    """Run every check on the held-out seasons and return the parts plus a summary row.

    The summary leads with the model columns -- held-out MSE improvement over
    the count-only predictor and the prediction-grouped calibration slope, per
    action -- then the player-metric checks. Only the model columns can say a
    model estimates well; the player checks say the score is stable and
    plausible, and serve as guardrails. To *choose* between two variants use
    `paired_accuracy`, which tests the difference on the same pitches.
    """
    held = run.held
    assert set(held['season'].unique()) <= set(HELD_OUT), 'harness scores held-out seasons only'
    out = metric_checks(held, value_col)
    acc = accuracy(run)
    cal = {a: prediction_calibration(held, a, ci=False) for a in ('take', 'swing')}

    summary = {'variant': label}
    for a in ('swing', 'take'):
        summary[f'{a} MSE vs count-only %'] = round(acc.loc[(a, 'pooled'), 'improvement_%'], 2)
        summary[f'{a} pred slope'] = round(cal[a].slope.mean(), 3)
    summary.update(out['summary'])
    return {**out, 'summary': summary, 'accuracy': acc, 'calibration': cal}


def in_sample_checks(run: FoldRun, label: str, value_col: str = DEC.DEFAULT_SCORE) -> dict:
    """The same player-metric checks on the last fold's own training seasons.

    Reported beside the held-out row so the optimism is visible, never averaged
    into it.
    """
    if run.in_sample is None:
        raise ValueError('run_folds(..., in_sample=True) is needed for in-sample checks')
    out = metric_checks(run.in_sample, value_col)
    return {**out, 'summary': {'variant': f'{label} (in-sample)', **out['summary']}}
