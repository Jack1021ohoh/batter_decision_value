"""Hitter-specific features.

Currently the nitro zone: the convex hull of the locations where a hitter
produced his hardest-hit balls in play. v2 introduced it and it is reproduced
here as designed -- binary, convex hull, top-5% exit velocity, 60 balls in play
minimum -- with two defects corrected.

**Leakage.** v2 built each season's hull from that same season's balls in play,
so `in_nitro` partly encoded the outcome it was used to predict, and on
held-out seasons it used the future. Hulls here are built from prior seasons
only; `season_in_nitro()` enforces it and asserts it.

**Silent deletion.** v2's `add_nitro_zone` inner-merged on `batter`, so a hitter
without a hull disappeared from the data entirely rather than scoring
`in_nitro = False`. With a prior-season hull only 79-88% of qualified hitters
have one, so that deletion would be large and non-random.

Better estimators of the same signal exist -- EDA §6 measured a kernel-smoothed,
shrunk surface at year-over-year r ~ 0.68 against ~ 0.30 for raw bins -- but
those are a change to v2's design rather than a correction of it, and belong to
the work that follows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError

#: Outcomes that are balls in play, i.e. rows that can carry an exit velocity.
BIP_OUTCOMES = ('single', 'double', 'triple', 'home_run', 'field_out', 'field_error')

#: v2's settings, kept as-is.
MIN_BIP = 60
EV_PERCENTILE = 95

X, Z = 'plate_x_mid', 'plate_z_mid'


def balls_in_play(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with a tracked exit velocity and location."""
    return df[df['outcome'].isin(BIP_OUTCOMES) & df['launch_speed'].notna()
              & df[X].notna() & df[Z].notna()]


def prior_seasons(df: pd.DataFrame, season: int) -> pd.DataFrame:
    """Every row from a season strictly before `season`.

    Named rather than inlined so the no-leak rule is visible in the code.
    """
    return df[df['season'] < season]


def batter_hulls(bip: pd.DataFrame, min_bip: int = MIN_BIP,
                 pct: int = EV_PERCENTILE) -> dict[int, np.ndarray]:
    """Convex hull per batter of his top `100-pct`% exit-velocity locations.

    Returns each hull's half-space equations, which is all the membership test
    needs. A batter with fewer than `min_bip` balls in play gets no hull; so
    does one whose points are degenerate (collinear), which `ConvexHull` raises
    `QhullError` for.
    """
    hulls: dict[int, np.ndarray] = {}
    for batter, g in bip.groupby('batter', observed=True):
        if len(g) < min_bip:
            continue
        threshold = np.percentile(g['launch_speed'].to_numpy(), pct)
        top = g.loc[g['launch_speed'] >= threshold, [X, Z]].to_numpy()
        try:
            hulls[batter] = ConvexHull(top).equations
        except QhullError:
            continue
    return hulls


def add_in_nitro(df: pd.DataFrame, hulls: dict[int, np.ndarray],
                 col: str = 'in_nitro') -> pd.DataFrame:
    """Flag pitches falling inside their batter's hull.

    Left-join semantics: a batter with no hull scores False on every pitch and
    is never dropped. Row count is preserved exactly.

    The membership test is `A @ p + b <= 0` for every face of the hull. v2
    applied that per row; here it runs once per batter over all of his pitches
    as a single matrix product.
    """
    df = df.copy()
    flag = np.zeros(len(df), dtype=bool)
    batters = df['batter'].to_numpy()
    points = df[[X, Z]].to_numpy()
    tracked = ~np.isnan(points).any(axis=1)

    for batter, equations in hulls.items():
        mask = (batters == batter) & tracked
        if not mask.any():
            continue
        p = points[mask]
        flag[mask] = np.all(p @ equations[:, :-1].T + equations[:, -1] <= 0, axis=1)

    df[col] = flag
    return df


def season_in_nitro(df: pd.DataFrame, min_bip: int = MIN_BIP,
                    col: str = 'in_nitro') -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add `in_nitro` season by season, each hull built from prior seasons only.

    Returns the frame (seasons with no prior are dropped, since they cannot be
    scored without leaking) and a coverage table, which matters: at 79-88%
    coverage the hitters without a hull are a large enough group that reporting
    them is part of the result, not a footnote.
    """
    bip = balls_in_play(df)
    seasons = sorted(df['season'].unique())
    frames, coverage = [], []

    for season in seasons[1:]:                      # the first has no prior
        history = prior_seasons(bip, season)
        assert (history['season'] < season).all(), f'leak building {season}'

        hulls = batter_hulls(history, min_bip=min_bip)
        current = df[df['season'] == season]
        scored = add_in_nitro(current, hulls, col=col)
        assert len(scored) == len(current), 'add_in_nitro dropped rows'
        frames.append(scored)

        pitches = current.groupby('batter').size()
        qualified = pitches[pitches >= 500].index
        coverage.append({
            'season': season,
            'prior_seasons': f'{seasons[0]}-{season - 1}',
            'hulls': len(hulls),
            'qualified': len(qualified),
            'qualified_with_hull': int(qualified.isin(hulls).sum()),
            'coverage': qualified.isin(hulls).mean(),
            'in_nitro_rate': float(scored[col].mean()),
        })

    return pd.concat(frames, ignore_index=True), pd.DataFrame(coverage).set_index('season')


def is_inside_hull_rowwise(plate_x: float, plate_z: float,
                           equations: np.ndarray) -> bool:
    """v2's original per-row membership test, kept to verify the vectorized one."""
    point = np.array([plate_x, plate_z])
    return bool(np.all(np.dot(equations[:, :-1], point) + equations[:, -1] <= 0))


# --------------------------------------------------------------------------
# Continuous hot zone
# --------------------------------------------------------------------------

#: Kernel bandwidth in feet, in the batter frame. A hot zone is spatially
#: smooth, so each ball in play informs its neighbourhood rather than one bin.
HOT_BANDWIDTH = 0.35

#: Empirical-Bayes shrinkage strength, in effective balls in play. A hitter with
#: little history reverts to the league surface instead of to noise.
HOT_SHRINKAGE = 60.0

#: Fixed prior window. Hull area grows with accumulated history -- the binary
#: flag fires on 14.5% of pitches with one prior season and 26.1% with five --
#: so a fixed window is what keeps the feature meaning the same thing each year.
PRIOR_WINDOW = 2


def hot_zone_surface(bip: pd.DataFrame, bandwidth: float = HOT_BANDWIDTH,
                     shrinkage: float = HOT_SHRINKAGE) -> tuple[dict, float]:
    """Per-batter expected exit velocity as a function of location.

    Returns `({batter: (points, values)}, league_mean)`, enough to evaluate the
    shrunk surface at any location later.

    This replaces the convex hull. The hull thresholds at the top 5% of one
    sample and so discards 95% of the balls in play, leaving a polygon defined
    by a handful of points; measured year over year a hull-class estimator
    reproduces itself at r ~ 0.30 against ~ 0.68 for this one.
    """
    league = float(bip['launch_speed'].mean())
    per_batter = {}
    for batter, g in bip.groupby('batter', observed=True):
        pts = g[['plate_x_bat', 'plate_z_norm']].to_numpy()
        vals = g['launch_speed'].to_numpy()
        ok = ~np.isnan(pts).any(axis=1) & ~np.isnan(vals)
        if ok.sum() >= 20:
            per_batter[batter] = (pts[ok], vals[ok])
    return per_batter, league


def add_hot_zone(df: pd.DataFrame, surface: tuple[dict, float],
                 bandwidth: float = HOT_BANDWIDTH, shrinkage: float = HOT_SHRINKAGE,
                 col: str = 'hot_zone') -> pd.DataFrame:
    """Evaluate the shrunk hot-zone surface at each pitch's location.

    Continuous, unlike the hull: a pitch just outside a hitter's best region is
    worth slightly less rather than nothing. Hitters with no history get the
    league mean, so they are neither dropped nor marked False.
    """
    per_batter, league = surface
    df = df.copy()
    out = np.full(len(df), league)
    batters = df['batter'].to_numpy()
    query = df[['plate_x_bat', 'plate_z_norm']].to_numpy()
    tracked = ~np.isnan(query).any(axis=1)
    h2 = 2 * bandwidth * bandwidth

    for batter, (pts, vals) in per_batter.items():
        mask = (batters == batter) & tracked
        if not mask.any():
            continue
        d2 = ((query[mask][:, None, :] - pts[None, :, :]) ** 2).sum(-1)
        w = np.exp(-d2 / h2)
        n_eff = w.sum(1)
        raw = (w * vals).sum(1) / np.maximum(n_eff, 1e-9)
        out[mask] = (n_eff * raw + shrinkage * league) / (n_eff + shrinkage)

    df[col] = out
    return df


def season_hot_zone(df: pd.DataFrame, window: int = PRIOR_WINDOW,
                    col: str = 'hot_zone') -> pd.DataFrame:
    """Add the hot-zone feature season by season, from a fixed prior window."""
    bip = balls_in_play(df)
    seasons = sorted(df['season'].unique())
    frames = []
    for season in seasons[1:]:
        history = bip[(bip['season'] < season) & (bip['season'] >= season - window)]
        assert (history['season'] < season).all(), f'leak building {season}'
        scored = add_hot_zone(df[df['season'] == season], hot_zone_surface(history), col=col)
        frames.append(scored)
    return pd.concat(frames, ignore_index=True)
