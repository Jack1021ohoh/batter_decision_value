"""Evaluation harness for swing-decision metrics.

Every variant is scored by these functions, so the results table in
FINDINGS.md compares like with like. Three of the choices here are not obvious:

* **Held-out scoring only.** Every variant is fitted on rolling-origin folds
  (`GENERIC_FOLDS`, `PERSONALIZED_FOLDS`) and each validation season is scored
  by the one fold model that did not train on it. The player-metric checks see
  those scores alone: split-half within 2023, 2024 and 2025, year-over-year and
  next-season validity on 2023->24 and 2024->25. 2026 is the final test and
  `run_folds` refuses to touch it.
* **Sub-models are judged on conditional means, not pitch-level RMSE.** An
  individual swing outcome is close to irreducible, so pitch-level error is
  dominated by noise the model is not trying to predict. `bin_calibration`
  scores what the model estimates -- the mean run value in a (location x count)
  cell -- on the held-out season, against a count-only predictor fitted on the
  same training seasons, and reports a calibration slope.
* Zone% correlation is a first-class output. Salorio retired SOTO after finding
  Zone% explained ~23% of its variance, i.e. it was substantially measuring the
  pitches a hitter was thrown rather than his decisions. Reliability cannot
  catch that: a metric of the wrong quantity can be perfectly stable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xgboost as xgb

from . import data as D
from . import decision as DEC
from .baselines import SEED, ActionModels, _dmatrix, fit_v1, predict_both, predict_chosen

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

#: The final test season. Scored once per model after every decision is
#: locked, never during model selection.
FINAL_SEASON = 2026


@dataclass
class FoldRun:
    """Held-out scores from one variant, plus what produced them."""
    held: pd.DataFrame                       # every held-out pitch, scored by its fold
    models: dict[int, ActionModels]          # keyed by held-out season
    folds: tuple
    in_sample: pd.DataFrame | None = field(default=None, repr=False)


def run_folds(df: pd.DataFrame, features: list[str], folds=GENERIC_FOLDS,
              in_sample: bool = False, **fit_kwargs) -> FoldRun:
    """Fit a variant on each fold and score only the season it held out.

    Per fold, everything that is learned is learned from the training seasons:
    the (outcome, count) run-value table that defines the target, both action
    models, and the count-only reference predictor used by `bin_calibration`.

    Returns the held-out seasons concatenated, with `q_take`, `q_swing`,
    `edge`, `y_pred` (the chosen action's value) and every per-pitch score in
    `decision.SCORES`. With `in_sample=True` the last fold's model also scores
    its own training seasons, kept apart in `.in_sample` so it can be reported
    beside the held-out figures but never averaged into them.
    """
    need = [f for f in features if f != 'count']
    frames, models = [], {}
    for train_seasons, valid in folds:
        train_seasons = tuple(train_seasons)
        if FINAL_SEASON in (*train_seasons, valid):
            raise ValueError(f'{FINAL_SEASON} is the final test season; it has no place in a fold')
        assert valid not in train_seasons and max(train_seasons) < valid

        rv = D.run_value_table(df, train_seasons)
        part = df[df['season'].isin([*train_seasons, valid])].dropna(subset=need)
        part = D.apply_run_value(part, rv, name='target').dropna(subset=['target'])
        train = part[part['season'].isin(train_seasons)]

        model = fit_v1(train, features=features, **fit_kwargs)
        count_only = train.groupby(['swing', 'count'], observed=True)['target'].mean()

        def score(frame):
            out = DEC.add_scores(predict_both(predict_chosen(frame, model), model))
            idx = pd.MultiIndex.from_arrays([out['swing'], out['count']])
            out['count_only'] = count_only.reindex(idx).to_numpy()
            return out

        held = score(part[part['season'] == valid])
        held['fold'] = valid
        frames.append(held)
        models[valid] = model

    run = FoldRun(held=pd.concat(frames, ignore_index=True), models=models, folds=tuple(folds))
    if in_sample:
        train_seasons, valid = folds[-1]
        rv = D.run_value_table(df, train_seasons)
        part = df[df['season'].isin(train_seasons)].dropna(subset=need)
        part = D.apply_run_value(part, rv, name='target').dropna(subset=['target'])
        model = models[valid]
        count_only = part.groupby(['swing', 'count'], observed=True)['target'].mean()
        ins = DEC.add_scores(predict_both(predict_chosen(part, model), model))
        ins['count_only'] = count_only.reindex(
            pd.MultiIndex.from_arrays([ins['swing'], ins['count']])).to_numpy()
        run.in_sample = ins
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


def bin_calibration(held: pd.DataFrame, action: str = 'swing',
                    min_n: int = MIN_BIN) -> pd.DataFrame:
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
        rows.append({'season': season, 'bins': len(agg), 'pitches': int(w.sum()),
                     'err': np.sqrt(mse), 'err_count_only': np.sqrt(mse_base),
                     'noise': np.sqrt(noise),
                     'signal_recovered': 1 - (mse - noise) / (mse_base - noise),
                     'slope': slope})
    return pd.DataFrame(rows).set_index('season')


WHIFF_PARAMS = {'objective': 'binary:logistic', 'eval_metric': 'auc', 'max_depth': 8,
                'learning_rate': 0.05, 'tree_method': 'hist', 'random_state': SEED}
WHIFF_ROUNDS = 300


def whiff_auc(df: pd.DataFrame, features: list[str], folds=GENERIC_FOLDS,
              params: dict | None = None, rounds: int = WHIFF_ROUNDS) -> pd.DataFrame:
    """Held-out whiff prediction on swings: AUC and log loss against the base rate.

    Whether a swing misses is far more predictable than what it is worth, which
    is the premise of decomposing the swing into events. Same folds as
    everything else, so the figure is comparable to the harness rows.
    """
    from sklearn.metrics import log_loss, roc_auc_score

    need = [f for f in features if f != 'count']
    sw = df[df['swing']].dropna(subset=need)
    whiff = (sw['outcome'] == 'swinging_strike').astype(int)
    rows = []
    for train_seasons, valid in folds:
        if FINAL_SEASON in (*train_seasons, valid):
            raise ValueError(f'{FINAL_SEASON} is the final test season; it has no place in a fold')
        tr, va = sw['season'].isin(train_seasons), sw['season'] == valid
        booster = xgb.train(params or WHIFF_PARAMS, _dmatrix(sw[tr], features, whiff[tr]), rounds)
        p = booster.predict(_dmatrix(sw[va], features))
        base = log_loss(whiff[va], np.full(va.sum(), whiff[tr].mean()))
        ll = log_loss(whiff[va], p)
        rows.append({'season': valid, 'swings': int(va.sum()),
                     'auc': roc_auc_score(whiff[va], p), 'logloss': ll,
                     'logloss_base_rate': base, 'improvement_%': (1 - ll / base) * 100})
    return pd.DataFrame(rows).set_index('season')


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
# One row for the scorecard
# --------------------------------------------------------------------------

#: The player-metric columns, in the order they should be weighed.
SHOW = ['chase | zone_swing', 'zone_swing | chase', 'split-half r', 'YoY R2 (mean)',
        'Zone% |r|', 'next-season r (partial)']


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

    The summary orders the columns the way the checks should be weighed:
    construct validity first, since it is the only one that tests whether the
    metric measures swing decisions at all; then reliability as a floor; then
    Zone% as a contamination veto. Predictive validity is reported last and is
    not decisive -- it asks whether the metric predicts future *production*,
    which is mostly hitting ability, so a clean decision metric can legitimately
    score low on it. The sub-model columns follow: held-out bin-mean error
    against the count-only predictor, and calibration slope.
    """
    held = run.held
    assert set(held['season'].unique()) <= set(HELD_OUT), 'harness scores held-out seasons only'
    out = metric_checks(held, value_col)
    cal = {a: bin_calibration(held, a) for a in ('take', 'swing')}

    summary = {'variant': label, **out['summary']}
    for a in ('swing', 'take'):
        c = cal[a]
        summary[f'{a} bin err / count-only'] = f"{c.err.mean():.4f} / {c.err_count_only.mean():.4f}"
        summary[f'{a} signal recovered'] = round(c.signal_recovered.mean(), 3)
        summary[f'{a} calib slope'] = round(c.slope.mean(), 3)
    return {**out, 'summary': summary, 'calibration': cal,
            'sub_models': rmse_vs_count_baseline(held)}


def in_sample_checks(run: FoldRun, label: str, value_col: str = DEC.DEFAULT_SCORE) -> dict:
    """The same player-metric checks on the last fold's own training seasons.

    Reported beside the held-out row so the optimism is visible, never averaged
    into it.
    """
    if run.in_sample is None:
        raise ValueError('run_folds(..., in_sample=True) is needed for in-sample checks')
    out = metric_checks(run.in_sample, value_col)
    return {**out, 'summary': {'variant': f'{label} (in-sample)', **out['summary']}}
