# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Scores MLB batters' swing/take decisions from Statcast pitch data, 2021–2026.
No build, lint, or test suite; `src/` holds the shared library and the analysis
lives in notebooks.

**The v1–v3 notebooks are gone** — retired for the bugs listed in README.md
("Why v3 was retired"), recoverable from git history at `fa48b14`
(`git show fa48b14:batter_decision_value_v3.ipynb`). Do not reintroduce their
logic; `IMPROVEMENT_PLAN.md` supersedes it. The v1 metric is still worth
reproducing as a benchmark (plan "Baselines to reproduce"), but from the spec
in the plan, not by copying the old notebook.

Two planning documents govern the rebuild and should be read before proposing
methodology changes:
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
  `./data/<year>_data.csv` for every season through one `fetch_season(year)`
  helper: season date range from the MLB Stats API, `game_type == 'R'`,
  overseas games dropped. Edit `YEARS` in the pull cell — each season is a slow
  download. It returns the output path, not the frame, since a season is ~700k
  rows × ~119 columns.
- **Overseas games are excluded** (temporary tracking installations): Mexico
  City, London, Seoul, Tokyo. Toronto is kept — Rogers Centre is a permanent
  park. The filter is `country not in {USA, Canada}`, resolved by a
  `game_pk` → venue lookup against the Stats API, because the pitch-level
  export has no venue column and international series use neutral sites where
  `home_team` is still an MLB club. `drop_overseas()` in `data_fetch.ipynb`.
  Any CSV pulled before this filter existed (2023/2024 contain overseas games)
  is stale — re-run `fetch_season()` for those years.
- `notebooks/eda.ipynb` is committed with outputs (~1 MB); it is re-executed
  with `uv run jupyter nbconvert --to notebook --execute --inplace` after edits
  so stored results always match the source. Never leave narrative claiming
  numbers the stored outputs do not show.

## v4 code (current work)

- `src/data.py` is the single source of loading/cleaning truth. `build_cache()`
  trims the raw CSVs to ~45 columns as `data/cache/<year>.parquet` (2.4 GB →
  374 MB, ~40s → ~0.1s per season); `load_seasons()` is the entry point and
  applies `to_middle_of_plate()` + `clean()`. Also holds `add_zone_frame()`,
  `in_rulebook_zone()`, `run_value_table()`, `drop_pitchers_batting()`, and the
  fetch helpers shared with `data_fetch.ipynb`.
- `src/baselines.py` — baseline models. `fit_v1()` takes a feature list, so v2
  is the same call with `in_nitro` appended rather than a copy.
  `predict_chosen()` is v1's defining choice (score the action actually taken);
  `predict_both()` gives the counterfactual pair every later variant needs.
- `src/evaluate.py` — the shared harness: `rmse_vs_count_baseline` (never report
  bare RMSE — the target is a function of (outcome, count), so count-only is the
  floor), `player_metric`, `yoy_reliability`, `split_half`,
  `zone_pct_correlation` (the SOTO test), `predictive_validity`, and `harness()`
  which returns one scorecard row.
- `src/decision.py` — the per-pitch scores, swappable: `chosen_value`,
  `signed_edge`, `regret`, `close_weighted`, `correct_decision`.
  `DEFAULT_SCORE = 'correct_decision'` — chosen on evidence, see v3.ipynb.
  **Do not reinstate a run-value-weighted score without re-running the
  comparison**: the magnitude is what carries the pitch-mix bias.
- `src/features.py` — the nitro zone: `batter_hulls`, `add_in_nitro` (left-join
  semantics; v2's inner merge silently deleted hitters without a hull),
  `season_in_nitro` (builds each season's hulls from prior seasons only, and
  asserts it). `is_inside_hull_rowwise` is kept to verify the vectorized test.
  Also the continuous version — `hot_zone_surface`, `add_hot_zone`,
  `season_hot_zone` (fixed two-season prior window). Measured inert: see v3.
- `notebooks/` — `eda.ipynb` (§1–§7, ends in a decisions table),
  `v1_baseline.ipynb`, `v2_baseline.ipynb`, and `v3.ipynb` (the patch). All do
  `sys.path.insert(0, '..')`; run from the repo root.
- Key EDA results now baked into `IMPROVEMENT_PLAN.md`: ABS band is exactly
  27%–53.5% of height; run values drift ≤0.014 runs across seasons (one table
  suffices); counterfactual support fails only on 3-0 off the plate; 41% of
  batter-location cells hold <10 balls in play, so per-hitter surfaces must be
  shrunk over a multi-season window.
- **ABS judges "any part of the ball", not the ball's centre** — same
  convention as the rulebook, so `in_rulebook_zone(ball_edge=True)` is correct
  for every season. Verified against the empirical called-strike boundary
  (EDA §4). The real 2026 change is the nominal zone: 2.64 in lower at the top
  (3.215 ft vs 3.435 ft in 2025) and ~2.8 in shorter.

## Conventions and gotchas

- `seed = 1126` everywhere.
- Statcast `plate_x` is from the catcher's view: positive = first-base side, so
  the same value is inside to a LHB and outside to a RHB. `add_zone_frame()`
  mirrors it into the batter frame (`plate_x_bat`, positive = inside).
- **2026 location data is on a different reference plane.** Per Statcast's CSV
  docs, `plate_x`/`plate_z` moved from front-of-plate to middle-of-plate in
  2026, and `sz_top`/`sz_bot` switched from operator-set to the ABS-defined
  zone. Measured effect: the ball sits ~1 inch lower at middle-of-plate, with a
  ~0.8 in spread by pitch type (FF least, CU most), so a constant offset does
  not fix it. Convert 2021–25 forward with the trajectory delta in
  IMPROVEMENT_PLAN.md §0.1.1 before pooling seasons.
- Hitter-level features for season *t* must be built from seasons < *t*. The
  v2/v3 nitro hull was drawn from the same season it scored, which leaked the
  outcome into its own feature.
- **v1/v2/v3 now refer to the rebuilt versions** in `notebooks/`, not the
  retired notebooks of the same name. The retired ones are at `fa48b14`.
- Four findings from v3 that should not be re-litigated without new evidence:
  the metric is `correct_decision` and magnitude-weighted scores are more
  pitch-mix contaminated; the swing model gains 0.1pp from every pitch
  characteristic available (0.8% → 0.9% over count-only); the take model is a
  called-strike probability (median R² 0.991); **personalization is a large
  effect that interacts with the metric** — hot zones differ between hitters at
  the same location by 0.6× the league location effect, and the feature takes
  next-season predictive r from 0.099 to 0.228 under `chosen_value`, while
  `correct_decision` cannot see it at all (it flips the recommended action on
  1.7% of pitches and is sign-based). Never test a feature against one metric.
- Retired version history, and why each step hurt:

  | Version | Change | YoY R² |
  |---|---|---|
  | v1 | location + count, mean chosen-action value over all pitches | 0.57 |
  | v2 | + same-season `in_nitro` hull | 0.34 |
  | v3 | metric changed to a takes-only rate | 0.16 |

  The two drops are independent — v2 was the feature, v3 was the metric.
