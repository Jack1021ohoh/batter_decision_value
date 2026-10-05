"""Batter decision value -- 2026 dashboard.

    uv sync --group dashboard
    uv run python dashboard/build_data.py      # after notebooks/final_models.ipynb
    uv run streamlit run dashboard/app.py

Reads the tables in dashboard/data/ (built from models/). Values are
`signed_edge`, the run value of the action a hitter chose minus the
alternative, shown per 100 pitches.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

DATA = Path(__file__).resolve().parent / 'data'
SEASON = 2026
# The personalized output only: the most accurate model, and the one this app is about.
# The generic output (a typical hitter) answers a different question; see the About page.
OUT = 'personalized'
PLATE_HALF_WIDTH = 17 / 24
MIN_CELL = 3          # hide map cells with fewer pitches than this
ROLL_GAMES = 15
FONT = 16             # chart text, px; the page's base size is in .streamlit/config.toml
# Savant's percentile palette: blue (poor) through grey to red (great).
PCT_COLORS = [[0, '#3661ad'], [0.5, '#c8c8c8'], [1, '#d82129']]

st.set_page_config(page_title='Batter Decision Value', layout='wide')


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
page = st.sidebar.radio('Page', ['Leaderboard', 'Hitter', 'Compare', 'About'], key='page')
st.sidebar.caption('Scores: 100 = average qualified hitter, 10 points = one standard deviation.')

# Streamlit keeps the scroll position across reruns; start a newly opened page at the top.
if st.session_state.get('shown_page') != page:
    st.session_state['shown_page'] = page
    st.session_state['page_views'] = st.session_state.get('page_views', 0) + 1
    st.html(f'''<script>(() => {{  // view {st.session_state['page_views']}
      window.__scrollRuns = (window.__scrollRuns || 0) + 1;
      const main = document.querySelector('[data-testid="stMain"]');
      [0, 100, 300].forEach(t => setTimeout(() => {{ if (main) main.scrollTo(0, 0); }}, t));
    }})();</script>''', unsafe_allow_javascript=True)


def hitter_picker(label: str, key: str, default: str | None = None) -> int:
    ids = list(NAMES)
    if st.session_state.get(key) not in NAMES:     # first visit, or set from the leaderboard
        st.session_state[key] = next((b for b, n in NAMES.items() if n == default), ids[0])
    return st.selectbox(label, ids, format_func=NAMES.get, key=key)


def open_hitter(batters: list[int]) -> None:
    """Leaderboard row click: switch to that hitter's page."""
    rows = st.session_state['leaderboard'].selection.rows
    if rows:
        st.session_state['hitter'] = batters[rows[0]]
        st.session_state['page'] = 'Hitter'


def styled(fig: go.Figure, **layout) -> go.Figure:
    # Streamlit's chart theme resets the layout font; size each text element explicitly.
    fig.update_layout(font=dict(size=FONT), legend_font_size=FONT, **layout)
    fig.update_xaxes(tickfont_size=FONT, title_font_size=FONT)
    fig.update_yaxes(tickfont_size=FONT, title_font_size=FONT)
    fig.update_annotations(font_size=FONT + 1)      # subplot titles and side values
    return fig


def score_cards(b: int, cols=None) -> None:
    r = H.set_index('batter').loc[b]
    cols = cols or st.columns(5)
    cols[0].metric('Decision score', f"{r[OUT]:.1f}",
                   help='100 = average qualified hitter; 10 points = one standard deviation')
    cols[1].metric('Rank', f"{int(r[f'rank_{OUT}'])} / {len(H)}", f"{r[f'pct_{OUT}']:.0f}th percentile",
                   delta_color='off')
    cols[2].metric('Chase rate', f"{r['chase_rate']:.1%}", help='Swings at pitches outside the zone')
    cols[3].metric('Zone-swing rate', f"{r['zone_swing_rate']:.1%}", help='Swings at pitches in the zone')
    cols[4].metric('Whiff rate', f"{r['whiff_rate']:.1%}", help='Swings and misses per swing')


def percentile_chart(b: int) -> go.Figure:
    """Savant-style percentile rankings among qualified hitters (100 = best)."""
    r = H.set_index('batter').loc[b]
    runs = lambda v: f'{v * 100:+.2f}'
    rows = [  # label, percentile column, value shown
        ('Decision score', f'pct_{OUT}', f'{r[OUT]:.1f}'),
        ('Decisions in the zone', f'pct_zone_value_{OUT}', runs(r[f'zone_value_{OUT}'])),
        ('Decisions out of the zone', f'pct_chase_value_{OUT}', runs(r[f'chase_value_{OUT}'])),
        ('Two-strike decisions', f'pct_two_strike_value_{OUT}', runs(r[f'two_strike_value_{OUT}'])),
        ('Chase %', 'pct_chase_rate', f"{r['chase_rate']:.1%}"),
        ('Whiff %', 'pct_whiff_rate', f"{r['whiff_rate']:.1%}"),
    ]
    labels = [lab for lab, _, _ in rows][::-1]
    pct = [r[c] for _, c, _ in rows][::-1]
    vals = [v for _, _, v in rows][::-1]
    fig = go.Figure()
    for y in labels:   # the grey track
        fig.add_shape(type='line', x0=0, x1=100, y0=y, y1=y, line=dict(color='#e3e3e3', width=10), layer='below')
    fig.add_trace(go.Bar(x=pct, y=labels, orientation='h', width=0.18, showlegend=False, hoverinfo='skip',
                         marker=dict(color=pct, cmin=0, cmax=100, colorscale=PCT_COLORS)))
    fig.add_trace(go.Scatter(
        x=pct, y=labels, mode='markers+text', text=[f'{p:.0f}' for p in pct], showlegend=False,
        textfont=dict(color='white', size=FONT - 2),
        marker=dict(size=34, color=pct, cmin=0, cmax=100, colorscale=PCT_COLORS, line=dict(color='white', width=2)),
        customdata=vals, hovertemplate='%{y}: %{customdata} · %{x:.0f}th percentile<extra></extra>'))
    for y, v in zip(labels, vals):
        fig.add_annotation(x=106, y=y, text=v, showarrow=False, xanchor='left')
    fig.update_xaxes(range=[-4, 122], showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, automargin=True, ticksuffix='  ')
    return styled(fig, height=60 * len(rows) + 40, margin=dict(t=10, b=10, l=230, r=10), bargap=0)


def decision_map(b: int, title: str = '') -> go.Figure:
    """Mean decision value per location, swings and takes side by side."""
    loc = T['location']
    d = loc[(loc['batter'] == b) & (loc['n'] >= MIN_CELL)]
    xs = np.sort(loc['x'].unique()); zs = np.sort(loc['z'].unique())
    fig = make_subplots(1, 2, subplot_titles=('When he swung', 'When he took'), horizontal_spacing=0.08)
    for k, action in enumerate(['swing', 'take'], start=1):
        a = d[d['action'] == action]
        grid = a.pivot_table(index='z', columns='x', values=f'value_{OUT}').reindex(index=zs, columns=xs) * 100
        n = a.pivot_table(index='z', columns='x', values='n').reindex(index=zs, columns=xs)
        fig.add_trace(go.Heatmap(
            x=xs, y=zs, z=grid.values, customdata=n.values, zmid=0, zmin=-12, zmax=12,
            colorscale='RdBu_r', colorbar=dict(title='runs / 100', len=0.8) if k == 2 else None,
            showscale=k == 2,
            hovertemplate='x %{x:.1f} ft · height %{y:.2f} of zone<br>value %{z:+.1f} runs / 100 pitches'
                          '<br>%{customdata} pitches<extra></extra>'), 1, k)
        fig.add_shape(type='rect', x0=-PLATE_HALF_WIDTH, x1=PLATE_HALF_WIDTH, y0=0, y1=1,
                      line=dict(color='black', width=2), row=1, col=k)
        fig.update_xaxes(title_text='← outside   ·   inside → (ft)', range=[-2.2, 2.2], row=1, col=k)
        fig.update_yaxes(title_text='height (0 = bottom, 1 = top of zone)', range=[-1.25, 2.0], row=1, col=k)
    return styled(fig, height=500, margin=dict(t=60, b=70), title=title)


def count_chart(b: int) -> go.Figure:
    c = T['counts']
    d = c[c['batter'] == b].set_index('count')
    order = ['0-0', '1-0', '2-0', '3-0', '0-1', '1-1', '2-1', '3-1', '0-2', '1-2', '2-2', '3-2']
    d = d.reindex(order)
    fig = go.Figure([
        go.Bar(x=order, y=d[f'value_{OUT}'] * 100, name=NAMES[b],
               customdata=d['n'], hovertemplate='%{x}: %{y:+.2f} runs / 100 (%{customdata} pitches)<extra></extra>'),
        go.Scatter(x=order, y=d[f'league_{OUT}'] * 100, name='league', mode='markers',
                   marker=dict(symbol='line-ew-open', size=24, color='black', line=dict(width=2))),
    ])
    return styled(fig, height=360, yaxis_title='runs / 100 pitches', margin=dict(t=30, b=20),
                  legend=dict(orientation='h', y=1.12))


def trend_chart(b: int) -> go.Figure:
    d = T['dates'][T['dates']['batter'] == b].sort_values('game_date')
    roll = d[f'sum_{OUT}'].rolling(ROLL_GAMES, min_periods=5).sum() / d['n'].rolling(ROLL_GAMES, min_periods=5).sum()
    league = T['league'][f'value_{OUT}'].iloc[0]
    fig = go.Figure([go.Scatter(x=d['game_date'], y=roll * 100, mode='lines', name=f'{ROLL_GAMES}-game rolling'),
                     go.Scatter(x=d['game_date'], y=[league * 100] * len(d), mode='lines', name='league',
                                line=dict(dash='dot', color='grey'))])
    return styled(fig, height=360, yaxis_title='runs / 100 pitches', margin=dict(t=30, b=20),
                  legend=dict(orientation='h', y=1.15))


def decisions_table(b: int, kind: str) -> pd.DataFrame:
    d = T['top_decisions']
    d = d[(d['batter'] == b) & (d['output'] == OUT) & (d['kind'] == kind)]
    d = d.sort_values('over_typical', ascending=(kind == 'worst'))
    return pd.DataFrame({
        'date': pd.to_datetime(d['game_date']).dt.date, 'count': d['count'], 'pitch': d['pitch_type'],
        'zone': np.where(d['in_zone'], 'in', 'out'), 'decision': np.where(d['swing'], 'swing', 'take'),
        'outcome': d['outcome'], 'league swing %': (d['league_p_swing'] * 100).round(0),
        'over a typical hitter': d['over_typical'].round(3), 'decision value': d['signed_edge'].round(3),
        'value of swinging': d['q_swing'].round(3), 'value of taking': d['q_take'].round(3)})


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------

if page == 'Leaderboard':
    st.title(f'{SEASON} leaderboard')
    st.caption('Click a row to open the hitter\'s page.')
    c1, c2 = st.columns([2, 1])
    query = c1.text_input('Search hitter', '')
    min_p = c2.slider('Minimum pitches', 500, int(H['pitches'].max()), 500, step=100)
    t = H[(H['pitches'] >= min_p) & H['name'].fillna('').str.contains(query, case=False)]
    t = t.sort_values(OUT, ascending=False)
    st.dataframe(pd.DataFrame({
        'rank': t[f'rank_{OUT}'], 'hitter': t['name'], 'score': t[OUT].round(1),
        'percentile': t[f'pct_{OUT}'], 'pitches': t['pitches'],
        'chase %': (t['chase_rate'] * 100).round(1), 'zone swing %': (t['zone_swing_rate'] * 100).round(1),
        'whiff %': (t['whiff_rate'] * 100).round(1),
    }), hide_index=True, width='stretch', height=640, key='leaderboard',
        on_select=partial(open_hitter, t['batter'].tolist()), selection_mode='single-row')

elif page == 'Hitter':
    b = hitter_picker('Hitter', 'hitter', default=H.sort_values(OUT, ascending=False)['name'].iloc[0])
    st.title(NAMES[b])
    score_cards(b)
    st.subheader('Percentile rankings')
    st.caption(f'Among the {len(H)} qualified hitters; 100 = best. Decision values are runs per 100 pitches '
               'gained over the alternative action.')
    st.plotly_chart(percentile_chart(b), width='stretch')
    st.subheader('Where his decisions gained or cost runs')
    st.caption('Average value of the chosen action over the alternative, per 100 pitches, by location '
               '(batter\'s view; the box is the strike zone). Red = better than the alternative, '
               f'blue = worse. Cells with fewer than {MIN_CELL} pitches are hidden.')
    st.plotly_chart(decision_map(b), width='stretch')
    c1, c2 = st.columns(2)
    with c1:
        st.subheader('By count')
        st.plotly_chart(count_chart(b), width='stretch')
    with c2:
        st.subheader('Through the season')
        st.plotly_chart(trend_chart(b), width='stretch')
    st.subheader('Best and costliest decisions')
    st.caption('Ranked by runs over what a typical hitter would have gained on the same pitch, so a choice '
               'nearly everyone makes (taking ball four a foot outside) earns almost nothing. '
               '"League swing %" is how often hitters swing at a pitch like it in this count.')
    st.markdown('**Best**')
    st.dataframe(decisions_table(b, 'best'), hide_index=True, width='stretch')
    st.markdown('**Costliest**')
    st.dataframe(decisions_table(b, 'worst'), hide_index=True, width='stretch')

elif page == 'Compare':
    st.title('Compare two hitters')
    ranked = H.sort_values(OUT, ascending=False)['name']
    c1, c2 = st.columns(2)
    with c1:
        a = hitter_picker('First hitter', 'cmp_a', default=ranked.iloc[0])
    with c2:
        b = hitter_picker('Second hitter', 'cmp_b', default=ranked.iloc[-1])
    cols = st.columns(2)
    for i, (col, who) in enumerate(zip(cols, (a, b))):
        with col:
            st.subheader(NAMES[who])
            st.plotly_chart(percentile_chart(who), width='stretch', key=f'pct_{i}')
    for i, who in enumerate((a, b)):
        st.subheader(NAMES[who])
        score_cards(who)
        st.plotly_chart(decision_map(who), width='stretch', key=f'map_{i}')

else:
    st.title('About')
    st.markdown(f'''
**What the score measures.** For every pitch, two models estimate what swinging
and what taking would be worth in runs, given the pitch (location, count,
handedness, velocity, movement, type). A decision's value is the action the
hitter chose minus the alternative. A hitter's score is his average decision
value, scaled so that 100 is the average qualified hitter and 10 points is one
standard deviation.

**Good for this hitter.** The swing model knows the hitter: his hot zone and
his whiff and foul tendencies from the two previous seasons, kept at about two
thirds strength. A slugger gains when he attacks pitches he damages; a contact
hitter is not punished for swings that suit him. This was the most accurate
model on held-out seasons. The project also has a generic version, which asks
whether a decision was good for a typical hitter; it answers a different
question and is not shown here.

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
