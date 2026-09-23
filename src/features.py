"""Hitter-specific features.

The nitro zone: the convex hull of the locations where a hitter
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

**Location surfaces** are the better estimator. A kernel-smoothed surface,
shrunk toward the league by effective sample size, reproduces itself year over
year at r ~ 0.68 against ~ 0.30 for raw bins (EDA section 6). `season_surface`
builds one for any per-row quantity from a fixed prior window:

* `season_hot_zone` -- expected exit velocity on balls in play (v3's damage
  prior);
* `season_contact_priors` -- whiff and foul rates per swing (Track B's contact
  priors).
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


def season_in_nitro(df: pd.DataFrame, min_bip: int = MIN_BIP, col: str = 'in_nitro',
                    window: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add `in_nitro` season by season, each hull built from prior seasons only.

    `window=None` uses every prior season, as the v2 rebuild does. That drifts:
    hull area grows with accumulated history, so the flag fires on 14.5% of
    pitches with one prior season and 25.1% with four. `window=2` fixes the
    prior to the two seasons before, matching the continuous surface, so the
    two can be compared on equal footing.

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
        if window is not None:
            history = history[history['season'] >= season - window]
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
            'prior_seasons': f'{history["season"].min()}-{season - 1}',
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
#: flag fires on 14.5% of pitches with one prior season and 25.1% with four --
#: so a fixed window is what keeps the feature meaning the same thing each year.
#: Two seasons also beats one as a predictor of the next season's surface
#: (EDA section 6).
PRIOR_WINDOW = 2


def value_surface(rows: pd.DataFrame, value_cols: list[str],
                  min_rows: int = 20) -> tuple[dict, np.ndarray]:
    """Per-batter points and values for a location surface of any per-row quantity.

    Returns `({batter: (points, values)}, league_means)`, `values` with one
    column per entry of `value_cols`. A batter with fewer than `min_rows`
    usable rows gets no surface and later takes the league value.
    """
    league = np.array([float(rows[c].mean()) for c in value_cols])
    per_batter = {}
    for batter, g in rows.groupby('batter', observed=True):
        pts = g[['plate_x_bat', 'plate_z_norm']].to_numpy()
        vals = g[value_cols].to_numpy(dtype=float)
        ok = ~np.isnan(pts).any(axis=1) & ~np.isnan(vals).any(axis=1)
        if ok.sum() >= min_rows:
            per_batter[batter] = (pts[ok], vals[ok])
    return per_batter, league


def hot_zone_surface(bip: pd.DataFrame, bandwidth: float = HOT_BANDWIDTH,
                     shrinkage: float = HOT_SHRINKAGE) -> tuple[dict, np.ndarray]:
    """Per-batter expected exit velocity as a function of location.

    This replaces the convex hull. The hull thresholds at the top 5% of one
    sample and so discards 95% of the balls in play, leaving a polygon defined
    by a handful of points; measured year over year a hull-class estimator
    reproduces itself at r ~ 0.30 against ~ 0.68 for this one.
    """
    return value_surface(bip, ['launch_speed'])


#: Grid the surface is evaluated on before lookup, in the batter frame. The
#: surface varies on the scale of the 0.35 ft bandwidth, more than three times
#: the spacing, so discretising costs nothing measurable.
GRID_X = np.arange(-1.6, 1.65, 0.10)
GRID_Z = np.arange(-0.6, 2.05, 0.10)
_GX, _GZ = np.meshgrid(GRID_X, GRID_Z, indexing='ij')
GRID_NODES = np.column_stack([_GX.ravel(), _GZ.ravel()])


def _shrunk_on_grid(pts: np.ndarray, vals: np.ndarray, league: np.ndarray,
                    bandwidth: float, shrinkage: float) -> np.ndarray:
    """Kernel-smoothed, shrunk surface at every grid node: (nodes, value columns)."""
    d2 = ((GRID_NODES[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
    w = np.exp(-d2 / (2 * bandwidth * bandwidth))
    n_eff = w.sum(1)
    out = np.empty((len(GRID_NODES), vals.shape[1]))
    for k in range(vals.shape[1]):
        raw = (w * vals[:, k]).sum(1) / np.maximum(n_eff, 1e-9)
        out[:, k] = (n_eff * raw + shrinkage * league[k]) / (n_eff + shrinkage)
    return out


def surface_on_grid(surface: tuple[dict, np.ndarray], bandwidth: float = HOT_BANDWIDTH,
                    shrinkage: float = HOT_SHRINKAGE) -> dict[int, np.ndarray]:
    """Every batter's shrunk surface on the grid, for comparing surfaces directly."""
    per_batter, league = surface
    return {b: _shrunk_on_grid(p, v, league, bandwidth, shrinkage)
            for b, (p, v) in per_batter.items()}


def add_surface(df: pd.DataFrame, surface: tuple[dict, np.ndarray], cols: list[str],
                bandwidth: float = HOT_BANDWIDTH, shrinkage: float = HOT_SHRINKAGE) -> pd.DataFrame:
    """Evaluate a shrunk surface at each pitch's location, one column per value.

    Continuous: a pitch just outside a hitter's best region is worth slightly
    less rather than nothing. Hitters with no history get the league value, so
    they are neither dropped nor marked False.

    Evaluated on a fixed grid and then looked up per pitch. Computing the kernel
    directly at every pitch location costs one distance per (pitch, row of
    history) pair; on the grid it costs one per (grid node, row of history)
    and the per-pitch step becomes an array index.
    """
    per_batter, league = surface
    df = df.copy()
    out = np.tile(league, (len(df), 1))
    query = df[['plate_x_bat', 'plate_z_norm']].to_numpy()
    tracked = ~np.isnan(query).any(axis=1)

    # Nearest grid node, not the one to the left: rounding rather than
    # searchsorted avoids a systematic half-cell bias in the looked-up value.
    ix = np.clip(np.rint((query[:, 0] - GRID_X[0]) / 0.10).astype(int), 0, len(GRID_X) - 1)
    iz = np.clip(np.rint((query[:, 1] - GRID_Z[0]) / 0.10).astype(int), 0, len(GRID_Z) - 1)
    flat = ix * len(GRID_Z) + iz

    # Index rows by batter once. Scanning `batters == batter` inside the loop
    # costs one full pass per hitter -- with ~650 hitters over 3.5M rows that
    # dominates everything else the function does.
    row_index = df.reset_index(drop=True).groupby('batter', observed=True).indices

    for batter, (pts, vals) in per_batter.items():
        rows = row_index.get(batter)
        if rows is None:
            continue
        rows = rows[tracked[rows]]
        if not len(rows):
            continue
        out[rows] = _shrunk_on_grid(pts, vals, league, bandwidth, shrinkage)[flat[rows]]

    for k, col in enumerate(cols):
        df[col] = out[:, k]
    return df


def add_hot_zone(df: pd.DataFrame, surface: tuple[dict, np.ndarray],
                 bandwidth: float = HOT_BANDWIDTH, shrinkage: float = HOT_SHRINKAGE,
                 col: str = 'hot_zone') -> pd.DataFrame:
    """Evaluate the shrunk hot-zone surface at each pitch's location."""
    return add_surface(df, surface, [col], bandwidth, shrinkage)


def season_surface(df: pd.DataFrame, rows: pd.DataFrame, value_cols: list[str],
                   cols: list[str], window: int = PRIOR_WINDOW) -> pd.DataFrame:
    """Add a surface feature season by season, each from a fixed prior window of `rows`.

    `rows` is the history the surface is built from (balls in play, swings, ...);
    every season of `df` after the first is scored from the `window` seasons
    before it, never its own.
    """
    seasons = sorted(df['season'].unique())
    frames = []
    for season in seasons[1:]:
        history = rows[(rows['season'] < season) & (rows['season'] >= season - window)]
        assert (history['season'] < season).all(), f'leak building {season}'
        frames.append(add_surface(df[df['season'] == season],
                                  value_surface(history, value_cols), cols))
    return pd.concat(frames, ignore_index=True)


def season_hot_zone(df: pd.DataFrame, window: int = PRIOR_WINDOW,
                    col: str = 'hot_zone') -> pd.DataFrame:
    """Add the hot-zone feature season by season, from a fixed prior window."""
    return season_surface(df, balls_in_play(df), ['launch_speed'], [col], window)


def swings_with_outcome(df: pd.DataFrame) -> pd.DataFrame:
    """Tracked swings, with 0/1 whiff and foul indicators to build surfaces from."""
    sw = df[df['swing'] & df['plate_x_bat'].notna() & df['plate_z_norm'].notna()]
    o = sw['outcome'].astype(str)
    return sw.assign(_whiff=(o == 'swinging_strike').astype(float),
                     _foul=(o == 'foul').astype(float))


def season_contact_priors(df: pd.DataFrame, window: int = PRIOR_WINDOW) -> pd.DataFrame:
    """Add `prior_whiff` / `prior_foul`: the hitter's location-smoothed, shrunk
    whiff and foul rates per swing, from a fixed prior window.

    Same estimator and settings as the hot zone, applied to swings instead of
    balls in play. Contact skill varies by location too, which is why this is a
    surface rather than one rate per hitter.
    """
    return season_surface(df, swings_with_outcome(df), ['_whiff', '_foul'],
                          ['prior_whiff', 'prior_foul'], window)
