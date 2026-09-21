# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Scores MLB batters' swing/take decisions from Statcast pitch data, 2021–2026.
No build, lint, or test suite; `src/` holds the shared library and the analysis
lives in notebooks.

Two documents should be read before proposing methodology changes:
- `FINDINGS.md` — what the data established: results, the metric analysis, the
  2026 ABS regime, and method notes. Every number is reproducible from a
  notebook. **Check here before re-deriving anything**; several conclusions in
  it reversed earlier assumptions and the reasoning is recorded.
- `mlb_swing_decision_related_work.md` — literature review (Yee–Deshpande,
  EAGLE, SEAGER, SwRV, SOTO, Nestico, Creally, Vock & Vock).

`IMPROVEMENT_PLAN.md` is the working roadmap and is **gitignored on purpose** —
it is scaffolding, it still describes paths that were abandoned, and a stale
plan misleads. It may be absent on a fresh clone; that is fine. Durable results
belong in `FINDINGS.md`, not there.

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
- Data is **not** in the repo. `notebooks/data_fetch.ipynb` writes
  `./data/<year>_data.csv` for every season through one `fetch_season(year)`
  helper: season date range from the MLB Stats API, `game_type == 'R'`,
  overseas games dropped. The helper itself lives in `src/data.py` so the fetch
  and the cleaning share one definition of the season bounds and the exclusion
  rule; the notebook is a thin driver. Edit `YEARS` in the pull cell — each
  season is a slow download. `pybaseball` is imported lazily inside
  `fetch_season()`, so `import src.data` stays fast for analysis.
- **Overseas games are excluded** (temporary tracking installations): Mexico
  City, London, Seoul, Tokyo. Toronto is kept — Rogers Centre is a permanent
  park. The filter is `country not in {USA, Canada}`, resolved by a
  `game_pk` → venue lookup against the Stats API, because the pitch-level
  export has no venue column and international series use neutral sites where
  `home_team` is still an MLB club. `drop_overseas()` in `src/data.py`.
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
  fetch helpers (`season_bounds`, `game_venues`, `drop_overseas`,
  `fetch_season`).
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
  `season_hot_zone` (fixed two-season prior window). **Not inert** — it is the
  most valuable feature found, but only under a magnitude-sensitive metric; a
  sign-based one cannot register it. See FINDINGS.md.
- `notebooks/` — `data_fetch.ipynb` (the pulls), `eda.ipynb` (§1–§7, ends in a
  decisions table), `v1_baseline.ipynb`, `v2_baseline.ipynb`, and `v3.ipynb`
  (the patch). All do `sys.path.insert(0, '..')`.
- Key EDA results now in `FINDINGS.md`: ABS band is exactly
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
  `data.to_middle_of_plate()` before pooling seasons.
- Hitter-level features for season *t* must be built from seasons < *t*, or the
  feature encodes the outcome it is used to predict. `season_in_nitro` and
  `season_hot_zone` enforce and assert it.
- Findings that should not be re-litigated without new evidence
  (all in `FINDINGS.md`, all reproducible from `notebooks/v3.ipynb`):
  - The selected score is **`signed_edge`**. It carries a pitch-mix
    contamination that the published metrics share and four attempts failed to
    remove; dropping the magnitude fixes it but makes the score blind to the
    most valuable feature available.
  - **Never judge the swing model by pitch-level RMSE.** Only 1.9% of that
    variance is learnable; scored on the conditional means it estimates, the
    model recovers 90.6–94.5%.
  - The take model **is** a called-strike probability (median R² 0.991), so a
    hitter-specific feature cannot help it. Contact quality does not change an
    umpire's call.
  - **Personalization is a large effect that interacts with the metric.** Hot
    zones differ between hitters at the same location by 0.6× the league
    location effect. A sign-based score cannot register it, since the feature
    flips the recommended action on only 1.7% of pitches. **Never test a
    feature against one metric.**
  - **Never vary two things at once.** Compare feature sets with the metric,
    learner, hyperparameters and seasons held fixed; compare metrics with the
    models held fixed.
