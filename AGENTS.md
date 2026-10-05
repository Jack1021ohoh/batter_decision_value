# AGENTS.md

This file provides guidance to coding agents (Claude Code, Codex and others)
when working with code in this repository. `CLAUDE.md` only imports it, so
this is the one place to edit.

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
  applies `to_middle_of_plate()` + `clean()`. Also holds the common zone
  (`listed_heights`, `common_zone_bounds`, `add_common_zone`),
  `add_zone_frame()`, `in_rulebook_zone()`, `run_value_table()`,
  `drop_pitchers_batting()`, and the fetch helpers (`season_bounds`,
  `game_venues`, `drop_overseas`, `fetch_season`).
- `src/baselines.py` — baseline models. `fit_v1()` takes a feature list, so v2
  is the same call with `in_nitro` appended rather than a copy. `fit_direct()`
  is the same two-model design under `LEARNER` (learning rate 0.05, rounds by
  early stopping on 10% of training games) and is what every version uses —
  hyperparameters are pipeline, not design. `fit_v1` (v1's original fixed
  settings) survives only for the one cell in v1 that shows why they were
  replaced. `fit_direct(recalibrate=True)` fits a linear map on the
  early-stopping games; it was tried and did not help (see FINDINGS).
  Hitter features (`SWING_ONLY_FEATURES`) go to the swing model only unless
  `swing_only=False`; v2 uses that to reproduce its as-designed routing.
  `predict_chosen()` is v1's defining choice (score the action actually taken);
  `predict_both()` gives the counterfactual pair every later variant needs.
- `src/evaluate.py` — the shared harness. `run_folds()` fits a variant on the
  rolling-origin folds (`GENERIC_FOLDS`, `PERSONALIZED_FOLDS`) — run-value
  table, both models and the count-only reference all learned per fold — and
  returns only held-out seasons (2023–25); it refuses 2026.
  `run_folds(fit=...)` takes any fit function returning an object with
  `.predict(df, action)`.
  - **Choosing models:** `paired_accuracy(run_a, run_b)` — squared error on
    the same pitches with a game-resampled interval; this decides.
    `accuracy(run)` against the count-only predictor.
  - **Calibration:** `calibration_table` / `prediction_calibration` (grouped
    by predicted value — the whole model), `bin_calibration` (by location ×
    count — cannot credit within-cell features), `propensity_calibration`
    with `swing_propensity` (is `Q_swing` sound where it is extrapolated),
    `outcome_diagnostics` for the decomposed classifier,
    `in_play_diagnostics` for the in-play event classifier;
    `paired_accuracy(..., rows='in_play')` scores the in-play branch alone.
    `recalibrate_last_season` was tried and rejected (over-corrects).
  - **Guardrails:** `harness()` returns one scorecard row (model columns, then
    construct validity, `split_half`, `yoy_reliability`,
    `zone_pct_correlation` — the SOTO test — and `predictive_validity`);
    `metric_differences(run_a, run_b)` gives hitter-resampled intervals on
    their paired differences; `zone_contamination(held, score)` gives Zone% r
    raw and under four controls. `in_sample_checks()` reports the last fold on its
    own training seasons, separately. Also `whiff_auc`,
    `rmse_vs_count_baseline`.
- `src/decomposition.py` — the whiff / foul / in-play swing model behind both
  outputs. `fit_decomposed` shares `fit_take` with `fit_direct`, so the twins'
  `Q_take` is identical (asserted in the notebook). Contact priors route to the
  outcome classifier, damage priors to the in-play model.
  `in_play='classifier'` (used by both outputs) values a ball in play as
  `Σ P(event)·RE(event, count)` over six events; `'regression'` (the default,
  kept so v4 reproduces) regresses its run value. `save_models` /
  `load_models` write and rebuild a fitted model (XGBoost files + metadata).
- `src/decision.py` — the per-pitch scores, swappable: `chosen_value`,
  `signed_edge`, `regret`, `close_weighted`, `correct_decision`.
  `DEFAULT_SCORE = 'signed_edge'` — keeps the run-value magnitude, as every
  published metric does, and unlike a sign-based score can register a feature
  that shifts `Q_swing` without flipping the decision. **Changing it re-opens a
  settled comparison** — the magnitude is what carries the pitch-mix bias, and
  every attempt to have both properties failed (`FINDINGS.md`).
- `src/features.py` — the nitro zone: `batter_hulls`, `add_in_nitro` (left-join
  semantics, so a hitter with no hull scores False rather than being dropped),
  `season_in_nitro` (builds each season's hulls from prior seasons only, and
  asserts it). `is_inside_hull_rowwise` is kept to verify the vectorized test.
  Also location surfaces for any per-row quantity (`value_surface`,
  `add_surface`, `season_surface`, fixed two-season prior window):
  `season_hot_zone` (exit velocity) and `season_contact_priors` (whiff and foul
  rates). Both improve the swing model's held-out accuracy; how much they move
  the *score* depends on the metric (a sign-based one barely registers them).
  See FINDINGS.md.
- `notebooks/` — `data_fetch.ipynb` (the pulls), `eda.ipynb` (§1–§7, ends in a
  decisions table), `v1_baseline.ipynb`, `v2_baseline.ipynb`, and `v3.ipynb`
  (the patch), `v4_decomposition.ipynb` (Track B), `v5_in_play.ipynb` (the
  in-play event classifier and the two outputs), and `final_models.ipynb` (the
  outputs refitted on every development season, saved to `models/`, and scored
  on 2026). All do `sys.path.insert(0, '..')`. The model-selection notebooks
  load 2021–2025 only; `final_models.ipynb` also loads 2026, to score it.
- **Keep each experiment in its own notebook.** A notebook re-runs end to end,
  so adding a section to an existing one ties every small decision to
  re-fitting everything before it. v5 exists for that reason.
- `models/` (gitignored) holds the saved final models and the 2026 scores
  (`pitch_values_2026.parquet`, `hitter_scores_2026.parquet`), rebuilt by
  `final_models.ipynb`.
- Key EDA results now in `FINDINGS.md`: ABS band is exactly
  27%–53.5% of height; run values drift 0.007 runs on a pitch-weighted
  average and ≤0.021 for the cells covering 95% of pitches (one table per
  fold suffices); counterfactual support is thinnest on 3-0 off the plate,
  where the decision is least ambiguous; 43% of batter-location cells hold <10 balls in
  play, so per-hitter surfaces must be smoothed and shrunk, on a two-season
  prior window.
- **ABS judges "any part of the ball", not the ball's centre** — same
  convention as the rulebook, so `in_rulebook_zone(ball_edge=True)` widens the
  zone by a ball radius on all four edges, in every season. Verified against
  the empirical called-strike boundary (EDA §4). The real 2026 change is the
  nominal zone: 2.64 in lower at the top (3.215 ft vs 3.435 ft in 2025) and
  ~2.9 in shorter.
- **One zone for every season: the ABS band on listed height.**
  `in_rulebook_zone()` and `plate_z_norm` both use `zone_top`/`zone_bot` from
  `add_common_zone()`, never `sz_top`/`sz_bot`, which are operator-set through
  2025 and the ABS zone from 2026. `zone='nominal'` exists for analyses that
  deliberately measure against the zone as recorded. Listed height is rounded
  to the inch, so the common zone is off by up to ~0.27 in at the top.

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
- **Every design decision is made on held-out folds; 2026 stays out of
  selection.** Use `E.run_folds` + `E.harness`, never a model scored on its own
  training seasons. Whenever a personalized variant is in a comparison, every
  row uses `PERSONALIZED_FOLDS`. 2026 is **not** a clean test — earlier
  versions scored it and printed a 2026 leaderboard — so describe it as a
  previously inspected, out-of-regime evaluation. That evaluation is done:
  `final_models.ipynb` fits each output through 2025 and scores 2026 once.
  The confirmatory test is 2027. That does not keep 2026 out of training: once
  2027 is complete, 2026 is ordinary data — training, and a held-out fold for
  selection (the only ABS-regime one) — the final model is fitted through
  2026, and 2027 is scored once after every decision is locked.
- Findings that should not be re-litigated without new evidence
  (all in `FINDINGS.md`, reproducible from the notebooks):
  - **Choose models by paired held-out accuracy** (`E.paired_accuracy`), then
    check calibration; the player-metric checks are guardrails that flag, never
    select. They show a score is stable and plausible, not that a model is
    right. Never read swing accuracy as a % improvement over count-only in
    isolation — outcome luck dominates the level; the paired difference is
    what counts.
  - **Two outputs**, both the decomposed swing model with the in-play event
    classifier, on the full pitch frame, with `signed_edge`: **generic** (no
    hitter features — a good decision for a typical hitter) and
    **personalized** (+ hot zone to the in-play model, + whiff/foul priors to
    the outcome classifier — a good decision for this hitter; the most accurate
    model found). The personalized score tracks plate discipline less, by
    design. The event classifier beat the in-play regression for the
    personalized output and tied for the generic one, which uses it by decision
    so both outputs differ only in hitter features (v5).
  - **The decomposition beats the direct regression on accuracy** (swing MSE
    −0.085% generic, −0.109% with the hot zone, intervals excluding 0) while
    leaving the score almost unchanged (r = 0.998). An earlier "tie" came from
    judging on player checks alone.
  - The selected score is **`signed_edge`**. By the criterion fixed in advance
    `correct_decision` measures better on construct validity; `signed_edge` is
    kept because it retains the run-value magnitude. Its pitch-mix
    contamination is real and unsolved.
  - **Calibration is close, not exact** (swing slopes 0.94–1.09, varying by
    fold). Two recalibration maps — on early-stopping games, and on the
    previous held-out season — both failed; models are used as fitted.
  - The take model **is** a called-strike probability (median R² 0.991, held
    out). Hitter features in it made it *less* accurate every time; they go to
    the swing side only.
  - **Personalization interacts with the metric.** Hot zones differ between
    hitters at the same location by 0.6× the league location effect, but the
    feature flips the recommended action on only 2.0% of pitches, so a
    sign-based score barely registers it. **Never test a feature against one
    metric.**
  - **Never vary two things at once.** Compare feature sets with the metric,
    learner, hyperparameters and seasons held fixed; compare metrics with the
    models held fixed.
