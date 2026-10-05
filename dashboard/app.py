"""Batter decision value -- 2026 dashboard.

    uv sync --group dashboard
    uv run python dashboard/build_data.py      # after notebooks/final_models.ipynb
    uv run streamlit run dashboard/app.py

Reads the tables in dashboard/data/ (built from models/). Values are
`signed_edge`, the run value of the action a hitter chose minus the
alternative, shown per 100 pitches.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

DATA = Path(__file__).resolve().parent / 'data'
SEASON = 2026
OUTPUT_LABELS = {'personalized': 'Personalized — for this hitter', 'generic': 'Generic — for a typical hitter'}
PLATE_HALF_WIDTH = 17 / 24
MIN_CELL = 3          # hide map cells with fewer pitches than this
ROLL_GAMES = 15

st.set_page_config(page_title='Batter Decision Value', page_icon='⚾', layout='wide')


@st.cache_data
def load() -> dict[str, pd.DataFrame]:
    missing = [n for n in ('hitters', 'location', 'counts', 'dates', 'top_decisions', 'league')
               if not (DATA / f'{n}.parquet').exists()]
    if missing:
        st.error(f'Missing {missing} in {DATA}. Run `uv run python dashboard/build_data.py` first.')
        st.stop()
    return {n: pd.read_parquet(DATA / f'{n}.parquet')
            for n in ('hitters', 'location', 'counts', 'dates', 'top_decisions', 'league')}


T = load()
H = T['hitters'].sort_values('name').reset_index(drop=True)
NAMES = dict(zip(H['batter'], H['name'].fillna(H['batter'].astype(str))))


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

st.sidebar.title('Batter Decision Value')
st.sidebar.caption(f'{SEASON} regular season · {len(H)} qualified hitters (≥500 pitches)')
page = st.sidebar.radio('Page', ['Leaderboard', 'Hitter', 'Compare', 'About'])
out = st.sidebar.radio('Output', list(OUTPUT_LABELS), format_func=OUTPUT_LABELS.get)
st.sidebar.caption('Scores: 100 = average qualified hitter, 10 points = one standard deviation.')


def hitter_picker(label: str, key: str, default: str | None = None) -> int:
    ids = list(NAMES)
    idx = ids.index(next((b for b, n in NAMES.items() if n == default), ids[0])) if default else 0
    return st.selectbox(label, ids, index=idx, format_func=NAMES.get, key=key)


def score_cards(b: int, cols=None) -> None:
    r = H.set_index('batter').loc[b]
    cols = cols or st.columns(5)
    cols[0].metric(f'{out.capitalize()} score', f"{r[out]:.1f}",
                   help='100 = average qualified hitter; 10 points = one standard deviation')
    cols[1].metric('Rank', f"{int(r[f'rank_{out}'])} / {len(H)}", f"{r[f'pct_{out}']:.0f}th percentile",
                   delta_color='off')
    other = 'generic' if out == 'personalized' else 'personalized'
    cols[2].metric(f'{other.capitalize()} score', f"{r[other]:.1f}")
    cols[3].metric('Chase rate', f"{r['chase_rate']:.1%}", help='Swings at pitches outside the zone')
    cols[4].metric('Zone-swing rate', f"{r['zone_swing_rate']:.1%}", help='Swings at pitches in the zone')


def decision_map(b: int, title: str = '') -> go.Figure:
    """Mean decision value per location, swings and takes side by side."""
    loc = T['location']
    d = loc[(loc['batter'] == b) & (loc['n'] >= MIN_CELL)]
    xs = np.sort(loc['x'].unique()); zs = np.sort(loc['z'].unique())
    fig = make_subplots(1, 2, subplot_titles=('When he swung', 'When he took'), horizontal_spacing=0.08)
    for k, action in enumerate(['swing', 'take'], start=1):
        a = d[d['action'] == action]
        grid = a.pivot_table(index='z', columns='x', values=f'value_{out}').reindex(index=zs, columns=xs) * 100
        n = a.pivot_table(index='z', columns='x', values='n').reindex(index=zs, columns=xs)
        fig.add_trace(go.Heatmap(
            x=xs, y=zs, z=grid.values, customdata=n.values, zmid=0, zmin=-12, zmax=12,
            colorscale='RdBu', colorbar=dict(title='runs / 100', len=0.8) if k == 2 else None,
            showscale=k == 2,
            hovertemplate='x %{x:.1f} ft · height %{y:.2f} of zone<br>value %{z:+.1f} runs / 100 pitches'
                          '<br>%{customdata} pitches<extra></extra>'), 1, k)
        fig.add_shape(type='rect', x0=-PLATE_HALF_WIDTH, x1=PLATE_HALF_WIDTH, y0=0, y1=1,
                      line=dict(color='black', width=2), row=1, col=k)
        fig.update_xaxes(title_text='← outside   ·   inside → (ft)', range=[-2.2, 2.2], row=1, col=k)
        fig.update_yaxes(title_text='height (0 = bottom, 1 = top of zone)', range=[-1.25, 2.0], row=1, col=k)
    fig.update_layout(height=430, margin=dict(t=60, b=20), title=title)
    return fig


def count_chart(b: int) -> go.Figure:
    c = T['counts']
    d = c[c['batter'] == b].set_index('count')
    order = ['0-0', '1-0', '2-0', '3-0', '0-1', '1-1', '2-1', '3-1', '0-2', '1-2', '2-2', '3-2']
    d = d.reindex(order)
    fig = go.Figure([
        go.Bar(x=order, y=d[f'value_{out}'] * 100, name=NAMES[b],
               customdata=d['n'], hovertemplate='%{x}: %{y:+.2f} runs / 100 (%{customdata} pitches)<extra></extra>'),
        go.Scatter(x=order, y=d[f'league_{out}'] * 100, name='league', mode='markers',
                   marker=dict(symbol='line-ew-open', size=24, color='black', line=dict(width=2))),
    ])
    fig.update_layout(height=320, yaxis_title='decision value, runs / 100 pitches', margin=dict(t=30, b=20),
                      legend=dict(orientation='h', y=1.12))
    return fig


def trend_chart(b: int) -> go.Figure:
    d = T['dates'][T['dates']['batter'] == b].sort_values('game_date')
    roll = d[f'sum_{out}'].rolling(ROLL_GAMES, min_periods=5).sum() / d['n'].rolling(ROLL_GAMES, min_periods=5).sum()
    league = T['league'][f'value_{out}'].iloc[0]
    fig = go.Figure([go.Scatter(x=d['game_date'], y=roll * 100, mode='lines', name=f'{ROLL_GAMES}-game rolling'),
                     go.Scatter(x=d['game_date'], y=[league * 100] * len(d), mode='lines', name='league',
                                line=dict(dash='dot', color='grey'))])
    fig.update_layout(height=300, yaxis_title='runs / 100 pitches', margin=dict(t=30, b=20),
                      legend=dict(orientation='h', y=1.15))
    return fig


def decisions_table(b: int, kind: str) -> pd.DataFrame:
    d = T['top_decisions']
    d = d[(d['batter'] == b) & (d['output'] == out) & (d['kind'] == kind)]
    d = d.sort_values('signed_edge', ascending=(kind == 'worst'))
    return pd.DataFrame({
        'date': pd.to_datetime(d['game_date']).dt.date, 'count': d['count'], 'pitch': d['pitch_type'],
        'zone': np.where(d['in_zone'], 'in', 'out'), 'decision': np.where(d['swing'], 'swing', 'take'),
        'outcome': d['outcome'], 'value of swinging': d['q_swing'].round(3),
        'value of taking': d['q_take'].round(3), 'decision value': d['signed_edge'].round(3)})


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------

if page == 'Leaderboard':
    st.title(f'{SEASON} leaderboard')
    st.caption(OUTPUT_LABELS[out])
    c1, c2 = st.columns([2, 1])
    query = c1.text_input('Search hitter', '')
    min_p = c2.slider('Minimum pitches', 500, int(H['pitches'].max()), 500, step=100)
    t = H[(H['pitches'] >= min_p) & H['name'].fillna('').str.contains(query, case=False)]
    t = t.sort_values(out, ascending=False)
    other = 'generic' if out == 'personalized' else 'personalized'
    st.dataframe(pd.DataFrame({
        'rank': t[f'rank_{out}'], 'hitter': t['name'], 'score': t[out].round(1),
        'percentile': t[f'pct_{out}'], f'{other} score': t[other].round(1),
        'personalized − generic': t['gap'].round(1), 'pitches': t['pitches'],
        'chase %': (t['chase_rate'] * 100).round(1), 'zone swing %': (t['zone_swing_rate'] * 100).round(1),
    }), hide_index=True, width='stretch', height=640)

elif page == 'Hitter':
    b = hitter_picker('Hitter', 'hitter', default=H.sort_values(out, ascending=False)['name'].iloc[0])
    st.title(NAMES[b])
    score_cards(b)
    st.subheader('Where his decisions gained or cost runs')
    st.caption('Average value of the chosen action over the alternative, per 100 pitches, by location '
               '(batter\'s view; the box is the strike zone). Blue = better than the alternative, '
               f'red = worse. Cells with fewer than {MIN_CELL} pitches are hidden.')
    st.plotly_chart(decision_map(b), width='stretch')
    c1, c2 = st.columns(2)
    with c1:
        st.subheader('By count')
        st.plotly_chart(count_chart(b), width='stretch')
    with c2:
        st.subheader('Through the season')
        st.plotly_chart(trend_chart(b), width='stretch')
    c1, c2 = st.columns(2)
    c1.subheader('Best decisions'); c1.dataframe(decisions_table(b, 'best'), hide_index=True, width='stretch')
    c2.subheader('Costliest decisions'); c2.dataframe(decisions_table(b, 'worst'), hide_index=True, width='stretch')

elif page == 'Compare':
    st.title('Compare two hitters')
    ranked = H.sort_values(out, ascending=False)['name']
    c1, c2 = st.columns(2)
    with c1:
        a = hitter_picker('First hitter', 'cmp_a', default=ranked.iloc[0])
    with c2:
        b = hitter_picker('Second hitter', 'cmp_b', default=ranked.iloc[-1])
    for who in (a, b):
        st.subheader(NAMES[who])
        score_cards(who)
        st.plotly_chart(decision_map(who), width='stretch')

else:
    st.title('About')
    st.markdown(f'''
**What the score measures.** For every pitch, two models estimate what swinging
and what taking would be worth in runs, given the pitch (location, count,
handedness, velocity, movement, type). A decision's value is the action the
hitter chose minus the alternative. A hitter's score is his average decision
value, scaled so that 100 is the average qualified hitter and 10 points is one
standard deviation.

**Two versions.**
- **Generic** asks whether each decision was good *for a typical hitter*.
- **Personalized** asks whether it was good *for this hitter*: it adds his hot
  zone and his whiff and foul tendencies, kept at about two thirds strength.
  A slugger gains when he attacks pitches he damages; a contact hitter is not
  punished for swings that suit him.

**Caveats.**
- {SEASON} is the first season under the automated ball-strike system. The
  models were trained on 2021–2025 and checked on held-out seasons; {SEASON}
  is reported as an out-of-regime season, not a clean test.
- The score is partly sensitive to the pitches a hitter is thrown: being thrown
  more obvious balls makes good decisions easier. This is a known limitation
  shared by published decision metrics.
- Individual outcomes are mostly luck; the models value the decision, not
  whether the ball found a hole.

Methods and results: `FINDINGS.md` in the repository.
''')
