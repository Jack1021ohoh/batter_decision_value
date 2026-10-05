"""Pre-aggregate the saved 2026 scores into the small tables the dashboard reads.

Reads `models/hitter_scores_2026.parquet` and `models/pitch_values_2026.parquet`
(written by `notebooks/final_models.ipynb`) and writes `dashboard/data/*.parquet`.
Re-run whenever the final models are refitted:

    uv run python dashboard/build_data.py

Every value is per pitch in runs, from `signed_edge` -- the value of the chosen
action minus the alternative -- for both outputs. The location grid is in the
batter's frame (positive x = inside) with height normalised to his common zone
(0 = bottom, 1 = top), so maps are comparable across hitters.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import data as D  # noqa: E402

SEASON = 2026
MODEL_DIR = ROOT / 'models'
OUT = Path(__file__).resolve().parent / 'data'
OUTPUTS = ('generic', 'personalized')

#: Location grid, batter frame: x in feet, height as a fraction of the zone.
X_EDGES = np.round(np.arange(-2.2, 2.2001, 0.4), 3)
Z_EDGES = np.round(np.arange(-1.25, 2.0001, 0.25), 3)
TOP_N = 10


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    hitters = pd.read_parquet(MODEL_DIR / f'hitter_scores_{SEASON}.parquet')
    p = pd.read_parquet(MODEL_DIR / f'pitch_values_{SEASON}.parquet')
    p = D.add_zone_frame(p)                       # plate_x_bat, plate_z_norm on the common zone
    p['in_zone'] = D.in_rulebook_zone(p, ball_edge=True)
    p['count'] = p['count'].astype(str)
    return hitters, p


def percentile(s: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Percentile rank 1-100 among qualified hitters, as Savant shows it; 100 is best."""
    return np.ceil(s.rank(ascending=higher_is_better, pct=True) * 100)


def hitter_table(hitters: pd.DataFrame, p: pd.DataFrame) -> pd.DataFrame:
    """One row per qualified hitter: both scores, ranks, percentiles, swing rates."""
    h = hitters.copy()
    for out in OUTPUTS:
        h[f'rank_{out}'] = h[out].rank(ascending=False, method='min').astype(int)
        h[f'pct_{out}'] = percentile(h[out])
    h['gap'] = h['personalized'] - h['generic']
    g = p.groupby('batter')
    swings = p[p['swing']]
    two_strike = p[p['count'].str.endswith('-2')]
    rates = pd.DataFrame({
        'chase_rate': p[~p['in_zone']].groupby('batter')['swing'].mean(),
        'zone_swing_rate': p[p['in_zone']].groupby('batter')['swing'].mean(),
        'swing_rate': g['swing'].mean(),
        'whiff_rate': swings.groupby('batter')['outcome'].apply(lambda o: o.eq('swinging_strike').mean()),
        'zone_pct': g['in_zone'].mean(),
    })
    for out in OUTPUTS:
        v = f'signed_edge_{out}'
        rates[f'zone_value_{out}'] = p[p['in_zone']].groupby('batter')[v].mean()
        rates[f'chase_value_{out}'] = p[~p['in_zone']].groupby('batter')[v].mean()
        rates[f'two_strike_value_{out}'] = two_strike.groupby('batter')[v].mean()
    h = h.merge(rates, left_on='batter', right_index=True, how='left')
    for col in [c for c in h if c.startswith(('zone_value_', 'chase_value_', 'two_strike_value_'))]:
        h[f'pct_{col}'] = percentile(h[col])
    for col in ('chase_rate', 'whiff_rate'):                  # lower is better
        h[f'pct_{col}'] = percentile(h[col], higher_is_better=False)
    return h


def location_grid(p: pd.DataFrame) -> pd.DataFrame:
    """Hitter x location cell x action: pitches and mean signed_edge, with the league's."""
    q = p.assign(cx=pd.cut(p['plate_x_bat'], X_EDGES, labels=False),
                 cz=pd.cut(p['plate_z_norm'], Z_EDGES, labels=False)).dropna(subset=['cx', 'cz'])
    q['cx'], q['cz'] = q['cx'].astype(int), q['cz'].astype(int)
    q['action'] = np.where(q['swing'], 'swing', 'take')
    vals = {f'value_{o}': (f'signed_edge_{o}', 'mean') for o in OUTPUTS}
    cell = q.groupby(['batter', 'action', 'cx', 'cz']).agg(n=('swing', 'size'), **vals).reset_index()
    league = (q.groupby(['action', 'cx', 'cz'])
                .agg(league_n=('swing', 'size'), **{f'league_{o}': (f'signed_edge_{o}', 'mean') for o in OUTPUTS})
                .reset_index())
    cell = cell.merge(league, on=['action', 'cx', 'cz'], how='left')
    cell['x'] = (X_EDGES[cell['cx']] + X_EDGES[cell['cx'] + 1]) / 2
    cell['z'] = (Z_EDGES[cell['cz']] + Z_EDGES[cell['cz'] + 1]) / 2
    # how often hitters swing at this location, for context
    swing_share = q.groupby(['cx', 'cz'])['swing'].mean().rename('league_swing_rate').reset_index()
    return cell.merge(swing_share, on=['cx', 'cz'], how='left')


def count_table(p: pd.DataFrame) -> pd.DataFrame:
    """Hitter x count: pitches, swing rate and mean signed_edge, with the league's."""
    vals = {f'value_{o}': (f'signed_edge_{o}', 'mean') for o in OUTPUTS}
    c = p.groupby(['batter', 'count']).agg(n=('swing', 'size'), swing_rate=('swing', 'mean'), **vals).reset_index()
    league = (p.groupby('count').agg(league_swing_rate=('swing', 'mean'),
                                     **{f'league_{o}': (f'signed_edge_{o}', 'mean') for o in OUTPUTS})
                .reset_index())
    return c.merge(league, on='count', how='left')


def date_table(p: pd.DataFrame) -> pd.DataFrame:
    """Hitter x game date: pitches and summed signed_edge, for rolling trends."""
    sums = {f'sum_{o}': (f'signed_edge_{o}', 'sum') for o in OUTPUTS}
    return p.groupby(['batter', 'game_date']).agg(n=('swing', 'size'), **sums).reset_index()


def top_decisions(p: pd.DataFrame, qualified) -> pd.DataFrame:
    """Each qualified hitter's best and worst decisions, per output."""
    cols = ['batter', 'game_date', 'count', 'pitch_type', 'plate_x_bat', 'plate_z_norm',
            'in_zone', 'swing', 'outcome']
    q = p[p['batter'].isin(qualified)]
    frames = []
    for out in OUTPUTS:
        v = f'signed_edge_{out}'
        keep = cols + [f'q_swing_{out}', f'q_take_{out}', v]
        best = q.sort_values(v, ascending=False).groupby('batter').head(TOP_N)[keep].assign(kind='best')
        worst = q.sort_values(v).groupby('batter').head(TOP_N)[keep].assign(kind='worst')
        frames.append(pd.concat([best, worst]).rename(columns={
            v: 'signed_edge', f'q_swing_{out}': 'q_swing', f'q_take_{out}': 'q_take'}).assign(output=out))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    hitters, p = load()
    OUT.mkdir(exist_ok=True)
    tables = {
        'hitters': hitter_table(hitters, p),
        'location': location_grid(p),
        'counts': count_table(p),
        'dates': date_table(p),
        'top_decisions': top_decisions(p, hitters['batter']),
    }
    # League averages per pitch, for reference lines.
    tables['league'] = pd.DataFrame([{f'value_{o}': p[f'signed_edge_{o}'].mean() for o in OUTPUTS}
                                     | {'season': SEASON, 'pitches': len(p)}])
    for name, t in tables.items():
        t.to_parquet(OUT / f'{name}.parquet', index=False)
        print(f'{name:14s} {len(t):>8,} rows')


if __name__ == '__main__':
    main()
