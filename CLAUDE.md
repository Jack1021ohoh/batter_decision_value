# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A notebook-only research project that scores MLB batters' swing/take decisions
from Statcast pitch data (2021–2024 as built; the v4 plan extends to 2026).
There is no package,
build, lint, or test suite — everything lives in Jupyter notebooks.

Two planning documents govern the next iteration (v4) and should be read before
proposing methodology changes:
- `IMPROVEMENT_PLAN.md` — the roadmap. Two tracks (A: patch the existing
  two-model design; B: event-decomposition redesign) on one shared evaluation
  harness, with a "Settled decisions" section at the end listing points that
  were debated and closed.
- `mlb_swing_decision_related_work.md` — literature review (Yee–Deshpande,
  EAGLE, SEAGER, SwRV, SOTO, Nestico, Creally, Vock & Vock). Track B follows
  its §5 model direction and §7 risk list.

## Running

- Environment is a uv project: `pyproject.toml` + `uv.lock`, Python 3.12.
  `uv sync` to install, `uv run python ...` / `uv run jupyter lab` to execute.
  Dev group holds jupyterlab, ipykernel, nbstripout.
- **macOS prerequisite:** LightGBM needs the OpenMP runtime, which is not a pip
  package — `brew install libomp`. Without it `import lightgbm` fails with
  `Library not loaded: @rpath/libomp.dylib`.
- Verified working stack: pandas 3.0, numpy 2.5, scikit-learn 1.9, lightgbm
  4.7, xgboost 3.4, pingouin 0.6, pybaseball 2.2.7. LightGBM/XGBoost
  categorical features, `groupby.apply` returning `ConvexHull` objects, and
  `pybaseball.statcast` all tested under pandas 3.
- The `baseball_env` kernel named in the old notebooks' metadata (Python 3.11)
  predates this; point notebooks at the uv venv instead.
- Data is **not** in the repo. `data_fetch.ipynb` writes
  `./data/<year>_data.csv`. The 2021–2024 cells are the original raw pulls;
  2025 onward go through `fetch_season(year)`, which takes the season date
  range from the MLB Stats API, keeps `game_type == 'R'`, and drops overseas
  games. Each pull is slow (a full season) — run only the years you need.
- **Overseas games are excluded** (temporary tracking installations): Mexico
  City, London, Seoul, Tokyo. Toronto is kept — Rogers Centre is a permanent
  park. The filter is `country not in {USA, Canada}`, resolved by a
  `game_pk` → venue lookup against the Stats API, because the pitch-level
  export has no venue column and international series use neutral sites where
  `home_team` is still an MLB club. `drop_overseas()` in `data_fetch.ipynb`;
  re-run the last cell to apply it to the pre-existing 2023/2024 CSVs.
- The analysis notebooks start with `%cd C:/Users/citioplab/works/codes/baseball_projects/`
  (a hardcoded Windows path). Change or delete that cell so `./data/` resolves
  from the repo root.
- Run `batter_decision_value_v3.ipynb` top-to-bottom. `v2` and `v1`
  (`batter_decision_value.ipynb`) are kept for comparison, not maintained.
- Notebooks are committed with outputs (2–5 MB each); clear outputs or diff on
  source only.

## Pipeline (v3), in execution order

All of this is in `batter_decision_value_v3.ipynb`; cell numbers refer to it.

1. **Clean** (`df_clean`, cells 7–9). `description` → `des_new` via `des_dict`;
   balls in play are re-labelled by their `events` via `ev_dict` (single,
   field_out, home_run…). Rows with 3 strikes / 4 balls are dropped.
   `swing` = 1 if `description ∈ swing_in`.
2. **Target.** `delta_run_exp_mean` = mean Statcast `delta_run_exp` grouped by
   `(des_new, count)`, computed *per year inside `df_clean`*. The models predict
   this (outcome, count) mean, not raw `delta_run_exp`.
3. **Nitro zone** (cells 10–14). Per batter, per season: convex hull of
   `(plate_x, plate_z)` for the top-5% `launch_speed` balls in play (≥ 60 BIP,
   else batter is dropped by the inner merge). `in_nitro` = point-in-hull test
   on every pitch.
4. **Models** (cells 32–69). Features `['plate_x','plate_z','in_nitro','count']`.
   - Take model (LightGBM, `swing == 0`) and swing model (LightGBM,
     `swing == 1`), both regressing `delta_run_exp_mean`. XGBoost versions are
     trained for comparison; LightGBM is retrained on all 2021–23 for use.
   - Called-strike model (XGBoost binary, `plate_x`/`plate_z` only) on takes.
5. **Per-pitch score** (cell 79):
   `y_pred = swing_pred·P(strike) − take_pred·(1 − P(strike))`.
6. **Player metric** (cells 74–75, 89). Over each batter's *takes only*:
   `hittable_take% = share with y_pred ≥ 0`, `correct_take% = share with
   y_pred ≤ 0`, `decision_value = z(correct − hittable)·10 + 100`, min 500
   pitches. Stability check = R² of 2022 vs 2023 scores (cells 82–86).

2021–2023 are training years; 2024 is held out and used only for the final
leaderboard, not for model evaluation.

## Conventions and gotchas

- `seed = 1126` everywhere.
- Statcast `plate_x` is from the catcher's view: positive = first-base side, so
  the same value is inside to a LHB and outside to a RHB. Neither `stand` nor
  `sz_top`/`sz_bot` is used yet (planned: IMPROVEMENT_PLAN.md §0.3).
- **2026 location data is on a different reference plane.** Per Statcast's CSV
  docs, `plate_x`/`plate_z` moved from front-of-plate to middle-of-plate in
  2026, and `sz_top`/`sz_bot` switched from operator-set to the ABS-defined
  zone. Measured effect: the ball sits ~1 inch lower at middle-of-plate, with a
  ~0.8 in spread by pitch type (FF least, CU most), so a constant offset does
  not fix it. Convert 2021–25 forward with the trajectory delta in
  IMPROVEMENT_PLAN.md §0.1.1 before pooling seasons.
- Strike-zone overlay for plots is fixed at x ∈ [−0.708, 0.708], z ∈ [1.5, 3.5]
  (`strike_zone` df + `draw_line`).
- `y_pred` on 2024 uses a nitro hull built from 2024 itself; on training years
  the hull is built from the same season it predicts. Both are known leaks
  (IMPROVEMENT_PLAN.md §0.1, A4).
- Version history and the reason each change was made:

  | Version | Change | YoY R² |
  |---|---|---|
  | v1 | location + count, mean chosen-action value over all pitches | 0.57 |
  | v2 | + `in_nitro` | 0.34 |
  | v3 | takes-only rate metric | 0.16 |

  The README's "year-over-year stable" claim predates v3's result.
