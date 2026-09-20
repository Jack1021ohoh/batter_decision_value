"""Evaluation harness for swing-decision metrics.

Every variant is scored by these functions, so the scorecard in
IMPROVEMENT_PLAN.md compares like with like. Two of the choices here are not
obvious:

* RMSE is always reported against a count-only baseline, never alone. The
  target is the mean run value of an (outcome, count) pair, so a model given
  only the count already reproduces most of it; the gap above that floor is
  what the other features buy. Absolute RMSE mostly reflects which action is
  being scored -- a take has three possible outcomes whose values sit within a
  few hundredths of a run at a fixed count, a swing has seven spanning nearly
  two runs -- so the level says little and the gap says everything.
* Zone% correlation is a first-class output. Salorio retired SOTO after finding
  Zone% explained ~23% of its variance, i.e. it was substantially measuring the
  pitches a hitter was thrown rather than his decisions. Reliability cannot
  catch that: a metric of the wrong quantity can be perfectly stable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as D
from .baselines import ActionModels, _dmatrix

QUALIFY_PITCHES = 500


# --------------------------------------------------------------------------
# Sub-model performance
# --------------------------------------------------------------------------

def rmse_vs_count_baseline(df: pd.DataFrame, models: ActionModels,
                           target: str = 'target') -> pd.DataFrame:
    """Per-action RMSE against a count-only lookup.

    The gap is what location actually buys; the level on its own says little.
    """
    rows = []
    for action, mask in (('take', ~df['swing']), ('swing', df['swing'])):
        g = df.loc[mask]
        y = g[target].to_numpy()
        pred = models.predict(g, action)
        baseline = g.groupby('count', observed=True)[target].transform('mean').to_numpy()
        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
        rmse_base = float(np.sqrt(np.mean((baseline - y) ** 2)))
        rows.append({'action': action, 'n': len(g), 'rmse': rmse,
                     'rmse_count_only': rmse_base,
                     'improvement_%': (1 - rmse / rmse_base) * 100})
    return pd.DataFrame(rows).set_index('action')


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
    if scale:
        z = g.groupby('season')['raw'].transform(lambda s: (s - s.mean()) / s.std(ddof=0))
        g['decision_value'] = 100 + 10 * z
    else:
        g['decision_value'] = g['raw']
    return g


# --------------------------------------------------------------------------
# Reliability
# --------------------------------------------------------------------------

def yoy_reliability(scores: pd.DataFrame, train_seasons=None) -> pd.DataFrame:
    """R^2 between consecutive seasons' scores, over hitters qualified in both."""
    train = set(train_seasons or [])
    seasons = sorted(scores.season.unique())
    rows = []
    for a, b in zip(seasons[:-1], seasons[1:]):
        m = (scores[scores.season == a][['batter', 'decision_value']]
             .merge(scores[scores.season == b][['batter', 'decision_value']],
                    on='batter', suffixes=('_a', '_b')))
        if len(m) < 30:
            continue
        r = np.corrcoef(m.decision_value_a, m.decision_value_b)[0, 1]
        rows.append({'pair': f'{a}->{b}', 'hitters': len(m), 'r': r, 'r2': r ** 2,
                     'in_sample': a in train and b in train})
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


def _production(df: pd.DataFrame) -> pd.DataFrame:
    """Offensive production per batter-season: delta run expectancy per PA.

    Taken from our own data rather than a linear-weights wOBA, so the measure
    is internally consistent with the modelling target and needs no external
    weight table maintained per season.
    """
    pa = df.groupby(['season', 'batter'], observed=True)['at_bat_number'].nunique()
    re = df.groupby(['season', 'batter'], observed=True)['delta_run_exp'].sum()
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

def harness(df: pd.DataFrame, models: ActionModels, label: str,
            value_col: str = 'y_pred', train_seasons=None) -> dict:
    """Run every check and return the parts, plus a one-line scorecard summary."""
    sub = rmse_vs_count_baseline(df, models)
    scores = player_metric(df, value_col)
    yoy = yoy_reliability(scores, train_seasons)
    sh = split_half(df, value_col)
    zp = zone_pct_correlation(df, scores)
    pv = predictive_validity(df, scores)

    summary = {
        'variant': label,
        'take RMSE vs count-only': f"{sub.loc['take','rmse']:.4f} / {sub.loc['take','rmse_count_only']:.4f}",
        'swing RMSE vs count-only': f"{sub.loc['swing','rmse']:.4f} / {sub.loc['swing','rmse_count_only']:.4f}",
        'split-half r': round(sh.r_spearman_brown.mean(), 3),
        'YoY R2 (mean)': round(yoy.r2.mean(), 3),
        'Zone% |r|': round(zp.r.abs().mean(), 3),
        'next-season r (partial)': round(pv.r_partial.mean(), 3),
    }
    return {'summary': summary, 'sub_models': sub, 'scores': scores,
            'yoy': yoy, 'split_half': sh, 'zone_pct': zp, 'predictive': pv}
