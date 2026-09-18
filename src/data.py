"""Load, cache, clean and harmonize Statcast data for the swing-decision model.

The raw pulls written by `data_fetch.ipynb` are ~400 MB per season across 119
columns. `build_cache()` trims them to the fields this project uses and writes
parquet, cutting a season load from ~40s to ~2s.

Everything downstream -- EDA and both modelling tracks -- should load through
`load_seasons()` so the cleaning rules and the 2026 harmonization are applied
in exactly one place.

See IMPROVEMENT_PLAN.md §0 for why each rule is what it is.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
CACHE_DIR = DATA_DIR / 'cache'

STATS_API = 'https://statsapi.mlb.com/api/v1'

# Rogers Centre is a permanent MLB park with a permanent Hawk-Eye installation,
# so Canada stays. Excluded are the overseas neutral-site series (Mexico City,
# London, Seoul, Tokyo), where tracking is a temporary rig.
KEEP_COUNTRIES = {'USA', 'Canada'}

# Reference planes, in feet from the back tip of home plate. Statcast reported
# location at the front of the plate through 2025 and at the middle from 2026
# on, to match ABS. vx0..az are specified at y = 50 ft.
Y_ANCHOR = 50.0
Y_FRONT = 17 / 12
Y_MIDDLE = 8.5 / 12

#: First season reported at middle-of-plate. Earlier seasons need conversion.
MIDDLE_OF_PLATE_FROM = 2026

CACHE_COLUMNS = [
    # identity
    'game_pk', 'game_date', 'game_type', 'at_bat_number', 'pitch_number',
    'batter', 'pitcher', 'stand', 'p_throws',
    # outcome
    'description', 'events', 'type', 'delta_run_exp',
    # location and the trajectory needed to move it between planes
    'plate_x', 'plate_z', 'sz_top', 'sz_bot', 'zone',
    'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az',
    'release_pos_x', 'release_pos_y', 'release_pos_z', 'release_extension',
    # pitch characteristics
    'release_speed', 'pfx_x', 'pfx_z', 'pitch_type',
    # state
    'balls', 'strikes', 'outs_when_up', 'inning', 'on_1b', 'on_2b', 'on_3b',
    # batted ball (contact quality; execution, so never a decision feature)
    'launch_speed', 'launch_angle', 'estimated_woba_using_speedangle', 'bat_speed',
]

#: description -> grouped outcome. Pitches whose description is absent here are
#: not decisions and are excluded by `clean()`; see NON_DECISION.
DESCRIPTION_MAP = {
    'ball': 'ball',
    'blocked_ball': 'ball',
    'pitchout': 'ball',
    'called_strike': 'called_strike',
    'foul': 'foul',
    'foul_bunt': 'foul',
    'foul_pitchout': 'foul',
    'swinging_strike': 'swinging_strike',
    'swinging_strike_blocked': 'swinging_strike',
    'foul_tip': 'swinging_strike',
    'missed_bunt': 'swinging_strike',
    'bunt_foul_tip': 'swinging_strike',
    'swinging_pitchout': 'swinging_strike',
    'hit_by_pitch': 'hit_by_pitch',
    'hit_into_play': 'hit_into_play',
}

#: Descriptions that involve no swing/take decision at all. `automatic_ball`
#: and `automatic_strike` are intentional walks (pre-2023) and pitch-clock
#: violations (2023+): no pitch is thrown, or none the batter could judge.
NON_DECISION = {'automatic_ball', 'automatic_strike'}

#: events -> grouped event, for balls in play. `field_error` is deliberately
#: kept separate from `field_out`: an error is not an out and carries a very
#: different run value.
EVENT_MAP = {
    'single': 'single',
    'double': 'double',
    'triple': 'triple',
    'home_run': 'home_run',
    'field_error': 'field_error',
    'field_out': 'field_out',
    'force_out': 'field_out',
    'grounded_into_double_play': 'field_out',
    'fielders_choice_out': 'field_out',
    'fielders_choice': 'field_out',
    'double_play': 'field_out',
    'triple_play': 'field_out',
    'sac_fly': 'field_out',
    'sac_fly_double_play': 'field_out',
    'sac_bunt': 'field_out',
    'sac_bunt_double_play': 'field_out',
    'other_out': 'field_out',
}

#: descriptions that mean the batter swung.
SWING_DESCRIPTIONS = {
    'foul', 'foul_bunt', 'foul_pitchout', 'foul_tip', 'hit_into_play',
    'swinging_strike', 'swinging_strike_blocked', 'swinging_pitchout',
    'missed_bunt', 'bunt_foul_tip',
}

BALL_RADIUS_FT = 0.12
PLATE_HALF_WIDTH_FT = 17 / 24  # 0.708


# --------------------------------------------------------------------------
# Fetch-side helpers (shared with data_fetch.ipynb)
# --------------------------------------------------------------------------

def season_bounds(year: int) -> tuple[str, str]:
    """Regular-season start/end dates from the MLB Stats API."""
    r = requests.get(f'{STATS_API}/seasons', params={'sportId': 1, 'season': year}, timeout=30)
    r.raise_for_status()
    s = r.json()['seasons'][0]
    return s['regularSeasonStartDate'], s['regularSeasonEndDate']


def game_venues(year: int) -> pd.DataFrame:
    """game_pk -> venue / city / country for every regular-season game in `year`."""
    start, end = season_bounds(year)
    r = requests.get(f'{STATS_API}/schedule',
                     params={'sportId': 1, 'gameType': 'R', 'startDate': start,
                             'endDate': end, 'hydrate': 'venue(location)'}, timeout=120)
    r.raise_for_status()
    rows = [{'game_pk': g['gamePk'],
             'venue': g['venue']['name'],
             'city': g['venue'].get('location', {}).get('city'),
             'country': g['venue'].get('location', {}).get('country')}
            for date in r.json()['dates'] for g in date['games']]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------

def build_cache(years, overwrite: bool = False) -> list[Path]:
    """Trim each raw season CSV to CACHE_COLUMNS and write parquet."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for year in years:
        out = CACHE_DIR / f'{year}.parquet'
        if out.exists() and not overwrite:
            print(f'{year}: cached already ({out.stat().st_size / 1e6:,.0f} MB)')
            written.append(out)
            continue

        src = DATA_DIR / f'{year}_data.csv'
        df = pd.read_csv(src, usecols=CACHE_COLUMNS, low_memory=False)
        df['season'] = year
        df['game_date'] = pd.to_datetime(df['game_date'])
        for col in ('stand', 'p_throws', 'description', 'events', 'type', 'pitch_type'):
            df[col] = df[col].astype('category')
        df.to_parquet(out, index=False)
        written.append(out)
        print(f'{year}: {len(df):,} rows, '
              f'{src.stat().st_size / 1e6:,.0f} MB CSV -> {out.stat().st_size / 1e6:,.0f} MB parquet')
    return written


def load_seasons(years, harmonize: bool = True, clean_rows: bool = True,
                 verbose: bool = False) -> pd.DataFrame:
    """Load cached seasons, optionally harmonized to middle-of-plate and cleaned.

    This is the entry point every notebook should use.
    """
    frames = []
    for year in years:
        path = CACHE_DIR / f'{year}.parquet'
        if not path.exists():
            raise FileNotFoundError(f'{path} missing -- run build_cache([{year}]) first')
        frames.append(pd.read_parquet(path))
    df = pd.concat(frames, ignore_index=True)

    if harmonize:
        df = to_middle_of_plate(df)
    if clean_rows:
        df = clean(df, verbose=verbose)
    return df


# --------------------------------------------------------------------------
# Harmonization
# --------------------------------------------------------------------------

def _time_to_plane(df: pd.DataFrame, y_ref: float) -> pd.Series:
    """Time from the y = 50 ft anchor until the ball reaches `y_ref`."""
    disc = df['vy0'] ** 2 - 2 * df['ay'] * (Y_ANCHOR - y_ref)
    return (-df['vy0'] - np.sqrt(disc)) / df['ay']


def to_middle_of_plate(df: pd.DataFrame) -> pd.DataFrame:
    """Put every season's location on the middle-of-plate plane.

    Statcast reported `plate_x`/`plate_z` at the front of the plate through
    2025 and at the middle from 2026 on. The ball sits ~1 inch lower at the
    middle, and the gap depends on the pitch (~0.7 in for a four-seamer, ~1.5
    in for a curveball), so it is not a constant offset.

    The plane-to-plane shift depends only on velocity and acceleration, not on
    absolute position, so it is added to the reported location directly. Do not
    anchor this at `release_pos_*`: vx0..az are specified at y = 50 ft, not at
    the release point (~54 ft).

    Adds `plate_x_mid` / `plate_z_mid`; 2026+ rows pass through unchanged.
    """
    df = df.copy()
    df['plate_x_mid'] = df['plate_x']
    df['plate_z_mid'] = df['plate_z']

    needs_shift = df['season'] < MIDDLE_OF_PLATE_FROM
    if not needs_shift.any():
        return df

    sub = df.loc[needs_shift]
    ta = _time_to_plane(sub, Y_FRONT)
    tb = _time_to_plane(sub, Y_MIDDLE)
    dt_, dt2 = tb - ta, tb ** 2 - ta ** 2

    df.loc[needs_shift, 'plate_x_mid'] = sub['plate_x'] + sub['vx0'] * dt_ + 0.5 * sub['ax'] * dt2
    df.loc[needs_shift, 'plate_z_mid'] = sub['plate_z'] + sub['vz0'] * dt_ + 0.5 * sub['az'] * dt2
    return df


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------

def clean(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Apply the outcome taxonomy and drop rows that are not swing decisions.

    Every dropped row is attributed to a reason; with `verbose` the tally is
    printed and the reasons plus the kept count sum to the input count.
    """
    n_in = len(df)
    reasons: dict[str, int] = {}

    def drop(mask: pd.Series, reason: str, frame: pd.DataFrame) -> pd.DataFrame:
        n = int(mask.sum())
        if n:
            reasons[reason] = reasons.get(reason, 0) + n
        return frame.loc[~mask]

    desc = df['description'].astype(str)
    df = drop(desc.isin(NON_DECISION), 'non-decision (auto ball/strike)', df)

    desc = df['description'].astype(str)
    df = drop(~desc.isin(DESCRIPTION_MAP), 'unmapped description', df)

    df = drop(df['plate_x'].isna() | df['plate_z'].isna(), 'no tracked location', df)
    df = drop(df[['vx0', 'vy0', 'vz0', 'ax', 'ay', 'az']].isna().any(axis=1), 'no trajectory', df)
    df = drop(df['sz_top'].isna() | df['sz_bot'].isna(), 'no strike-zone bounds', df)
    df = drop(df['delta_run_exp'].isna(), 'no delta_run_exp', df)
    df = drop((df['balls'] > 3) | (df['strikes'] > 2), 'impossible count', df)

    df = df.copy()
    df['outcome'] = df['description'].astype(str).map(DESCRIPTION_MAP)

    in_play = df['outcome'] == 'hit_into_play'
    df.loc[in_play, 'outcome'] = df.loc[in_play, 'events'].astype(str).map(EVENT_MAP)
    df = drop(in_play & df['outcome'].isna(), 'in-play with unmapped event', df)

    df = df.copy()
    df['swing'] = df['description'].astype(str).isin(SWING_DESCRIPTIONS)
    df['count'] = (df['balls'].astype(int).astype(str) + '-'
                   + df['strikes'].astype(int).astype(str)).astype('category')

    if verbose:
        print(f'{n_in:,} rows in')
        for reason, n in reasons.items():
            print(f'  -{n:>9,}  {reason}')
        print(f'{len(df):>10,}  kept')
        assert len(df) + sum(reasons.values()) == n_in, 'rows unaccounted for'
    return df


def drop_pitchers_batting(df: pd.DataFrame, min_pa: int = 50) -> pd.DataFrame:
    """Remove plate appearances by pitchers (2021 NL, before the universal DH).

    A batter is treated as a pitcher if he also appears as a pitcher that
    season and takes fewer than `min_pa` plate appearances -- which keeps
    two-way players such as Ohtani.
    """
    out = []
    for season, g in df.groupby('season', observed=True):
        pitchers = set(g['pitcher'].unique())
        pa = g.groupby('batter')['at_bat_number'].nunique()
        suspect = {b for b in pa.index if b in pitchers and pa[b] < min_pa}
        out.append(g[~g['batter'].isin(suspect)])
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------------
# Derived features
# --------------------------------------------------------------------------

def add_zone_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Put location in the batter's frame.

    `plate_x` is catcher-relative, so +0.7 ft is outside to a right-handed
    batter and inside to a left-handed one. Mirroring makes positive mean
    inside for everyone and doubles the data available per location.
    `plate_z_n` normalizes height to the batter's own zone: 0 is the bottom,
    1 the top.
    """
    df = df.copy()
    x = df['plate_x_mid'] if 'plate_x_mid' in df else df['plate_x']
    z = df['plate_z_mid'] if 'plate_z_mid' in df else df['plate_z']
    df['plate_x_bat'] = np.where(df['stand'].astype(str) == 'R', -x, x)
    df['plate_z_norm'] = (z - df['sz_bot']) / (df['sz_top'] - df['sz_bot'])
    return df


def in_rulebook_zone(df: pd.DataFrame, ball_edge: bool = True) -> pd.Series:
    """Whether the pitch is in the strike zone.

    `ball_edge=True` uses the human convention -- any part of the ball over the
    plate. ABS judges the ball's centre, so pass `ball_edge=False` for 2026+.
    """
    half = PLATE_HALF_WIDTH_FT + (BALL_RADIUS_FT if ball_edge else 0.0)
    x = df['plate_x_mid'] if 'plate_x_mid' in df else df['plate_x']
    z = df['plate_z_mid'] if 'plate_z_mid' in df else df['plate_z']
    return x.abs().le(half) & z.between(df['sz_bot'], df['sz_top'])


def run_value_table(df: pd.DataFrame, years) -> pd.Series:
    """Mean `delta_run_exp` by (outcome, count), computed on `years` only.

    The old pipeline recomputed this per season inside the cleaning step, so
    each season's targets were built from its own run environment. Fit it once
    on the training seasons and apply it everywhere.
    """
    train = df[df['season'].isin(list(years))]
    return train.groupby(['outcome', 'count'], observed=True)['delta_run_exp'].mean()


def apply_run_value(df: pd.DataFrame, table: pd.Series,
                    name: str = 'run_value') -> pd.DataFrame:
    """Attach the (outcome, count) run value from `table` as `name`."""
    df = df.copy()
    idx = pd.MultiIndex.from_arrays([df['outcome'], df['count'].astype(str)])
    df[name] = table.reindex(idx).to_numpy()
    return df
