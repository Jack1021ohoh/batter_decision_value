# Improvement Plan — Batter Decision Value (v4)

Two tracks, one evaluation harness.

- **Track A — patch the existing two-model design.** Fix the known bugs in v3
  and add the missing pitch-frame features. The result is a corrected
  Nestico/SOTO-v2-style model. It is the *reproduced baseline* that Track B
  has to beat, and it is cheap.
- **Track B — event-decomposition redesign.** Replace direct ΔRE regression
  with called-strike / swing-outcome / contact-quality sub-models, add
  opportunity standardization, and add a personalized contact-quality term.
  This is the contribution (see `mlb_swing_decision_related_work.md` §5, §7,
  §10).

Both tracks share the data split, the run-value lookup, and the evaluation
harness, so every number is comparable.

Where the project stands:

| Version | Features | Player metric | YoY R² (2022→2023) |
|---|---|---|---|
| v1 | `plate_x, plate_z, count` | mean predicted ΔRE of chosen action, all pitches | 0.57 |
| v2 | + `in_nitro` | same as v1 | 0.34 |
| v3 | + `in_nitro` | takes-only rate (`correct_take − hittable_take`) | 0.16 |

The two drops are independent: v1→v2 came from adding same-season `in_nitro`
with the metric unchanged; v2→v3 from changing the metric with features
unchanged.

---

## 0. Shared foundation

### 0.1 Data and split

Six regular seasons, 2021–2026. 2026 is the ABS-challenge era and is the test
season; see §0.1.1 for the measurement changes that must be harmonized first.

| Role | Seasons | Use |
|---|---|---|
| Train | 2021–2024 | fit all sub-models (CV stage) |
| Validate | 2025 | calibration, early stopping, feature and hyperparameter decisions — **and** the in-regime held-out reference |
| Test | 2026 | final evaluation, out-of-sample **and** out-of-regime |

- `data_fetch.ipynb` pulls every season through one `fetch_season(year)`
  helper (API-derived date range, `game_type == 'R'`, overseas games dropped).
  2026 is ~95% complete as of 2026-09-17 — re-pull after the finale.
- **Exclude overseas games** (tracking is a temporary installation there, so
  calibration may differ): Mexico City 2023/2024/2026, London 2023/2024,
  Seoul 2024, Tokyo 2025 — 14 games over the six seasons. Toronto is **kept**:
  Rogers Centre is a permanent park with a permanent Hawk-Eye rig. Implemented
  as `country not in {USA, Canada}` via a `game_pk` → venue lookup from the
  MLB Stats API (`/schedule?hydrate=venue(location)`), since the pitch-level
  Statcast export carries no venue field and international series are played
  at neutral sites where `home_team` is still an MLB club.
- Drop pitchers batting (2021 NL). Drop 3-strike / 4-ball rows as now.
- Hitter-level features (nitro zone, contact-quality surface, prior bat speed)
  for season *t* are built from seasons < *t* only. More seasons of history is
  the real benefit of extending the data: the league-level sub-models are
  saturated by ~2M pitches, but the per-batter contact-quality surface is
  estimated from a few hundred balls in play per batter-season, so a 2021–25
  prior window for scoring 2026 is materially better than a 2021–23 one.
- Stability checks use four season pairs (22→23, 23→24, 24→25 in-regime, and
  25→26 across the regime change).

#### 0.1.1 2026 measurement harmonization (do before anything else)

Three things changed in 2026, and the feature changes matter more than the
label change. From the Statcast CSV docs, verbatim:

- `plate_x` / `plate_z`: "Through 2025, this was front-of-plate. From 2026 on,
  this is middle-of-plate to align with the ABS system."
- `sz_top` / `sz_bot`: through 2025 operator-set when the ball is halfway to
  the plate; "From 2026 on, this is the top/bottom of the batter's ABS-defined
  strike zone."
- Called strikes can now be overturned by challenge (a low single-digit share
  of pitches — real, but small next to the feature redefinition).

**Measured size of the shift** (2026-09-17, one June day per era, ~4.5k
pitches each, method below):

| | mean | p5 / p95 | by pitch type |
|---|---|---|---|
| vertical (`dz`, front→middle) | **−1.00 in** (2025), −0.94 in (2026) | −1.54 / −0.56 in | FF −0.70 in … CU −1.50 in |
| horizontal (`dx`) | +0.15 in | \|dx\| p95 0.69 in | negligible |

So the ball sits about **one inch lower** at middle-of-plate than at
front-of-plate, with a **~0.8 in pitch-type-dependent spread** (fastballs move
least, curveballs most). This is a modest correction, not a catastrophic
incompatibility — but it is worth making, because called-strike probability is
steep at the zone edge and the bias correlates with pitch type, so it cannot be
absorbed by a constant offset.

**Fix — convert 2021–25 forward to middle-of-plate.** The plane-to-plane shift
depends only on velocity and acceleration, not on absolute position, so it is
computed exactly and added to the existing `plate_x` / `plate_z`:

```python
Y0, FRONT, MIDDLE = 50.0, 17/12, 8.5/12   # vx0..az are specified at y = 50 ft

def _t_to(df, y_ref):
    return (-df.vy0 - np.sqrt(df.vy0**2 - 2*df.ay*(Y0 - y_ref))) / df.ay

def to_middle(df):                         # seasons <= 2025 only
    ta, tb = _t_to(df, FRONT), _t_to(df, MIDDLE)
    dt, dt2 = tb - ta, tb**2 - ta**2
    return (df.plate_x + df.vx0*dt + 0.5*df.ax*dt2,
            df.plate_z + df.vz0*dt + 0.5*df.az*dt2)
```

2026 rows are already middle-of-plate and pass through unchanged. Use the
resulting `x_ref` / `z_ref` everywhere in place of `plate_x` / `plate_z`.

Do **not** anchor the propagation at `release_pos_x/y/z`: Statcast specifies
`vx0…az` at y = 50 ft, not at the release point (~54 ft), so mixing the two
gives inconsistent positions. The delta form above avoids the anchor entirely.
All required fields (`vx0,vy0,vz0,ax,ay,az`) are present in 2026 pulls —
verified.

For the zone, derive `sz_top` / `sz_bot` from batter height for every season
rather than mixing operator-set with ABS-defined values. The ABS zone is a
fixed percentage band of height. **Confirmed by EDA §4**: from 2026 a batter's
`sz_top`/`sz_bot` never vary (within-batter std 0.0000, vs 0.073–0.098 before),
and `sz_bot/sz_top` = **0.5047 with std 0.0000** across all hitters, matching
27.0/53.5 = 0.5047 exactly. The band is 27%–53.5% of height.

Zone **convention is unchanged**: the 2026 rules define a strike as a pitch any
part of which passes through the zone, and ABS applies that same standard on a
2D plane at the plate midpoint — it does not judge the ball's centre. Verified
against the empirical 50% called-strike boundary (EDA §4): in 2026 it sits
+0.163 ft past `sz_top`, +0.129 ft past `sz_bot`, and at 0.847 ft horizontally,
against a ball radius of 0.121 ft and a predicted 0.829 ft; a centre-based zone
would predict 0.000 and 0.708. **Use the ball-edge convention (±0.829 ft) for
every season.**

What did change is the *nominal* zone: the ABS top is 2.64 in lower than the
operator-set top it replaced (3.215 ft vs 3.435 ft in 2025), and the zone is
~2.8 in shorter overall. That is what moves the 2026 called-strike rates.

**Consequence for any cross-era zone comparison.** Measuring "distance past the
zone edge" in each season's own nominal units makes 2026 look *more* generous,
which is backwards. Use `add_common_zone()`: batter height is recoverable
exactly from the 2026 band (`sz_top/0.535` and `sz_bot/0.270` agree to 0.0000
in across all 659 batters), giving one zone that means the same thing in every
season. On that footing the called zone clearly tightened — the 50% boundary
goes 0.172 → 0.122 ft and the effective called area 3.18 → 2.91 sq ft (−8.6%),
with 2026 landing on the ball radius (0.121 ft), i.e. the true rulebook
boundary.

The height-derived zone is also the better *feature*: operator-set bounds carry
a within-batter std of 0.073–0.098 ft of pure measurement noise, which
`plate_z_norm` currently inherits for 2021–25.

#### 0.1.2 Model selection: rolling-origin CV

Do not pick a single validation season. Roll the origin so every design
decision gets three chronological estimates and no season is wasted:

```
train 2021–22 → validate 2023
train 2021–23 → validate 2024
train 2021–24 → validate 2025
```

If the folds show drift, weight recent seasons more or add a season index.

#### 0.1.3 Final refit and the A/B protocol

Keep **two** models. Refitting on everything through 2025 destroys the
in-regime reference, because 2025 becomes in-sample and its calibration is no
longer comparable to 2026.

| | Trained on | Used for |
|---|---|---|
| Model A | 2021–2024 | held-out 2025 metrics = in-regime reference |
| Model B | 2021–2025 | final 2026 leaderboard |

Score 2026 with **both**. That isolates each effect:

- A on 2025 vs. A on 2026 → same model, different regime = the ABS effect.
- A on 2026 vs. B on 2026 → same regime, different training data = what the
  extra season bought.
- B on 2026 → the published leaderboard.

Comparing A/2025 against B/2026 changes two things at once and attributes
nothing.

For Model B there is no held-out season left for early stopping: fix the round
count from the CV folds (mean best iteration, scaled up ~10–15% for the larger
training set) and keep hyperparameters locked from the CV stage. Re-tuning here
turns the refit into another selection pass.

**One-shot rule.** 2026 is scored once per model, after every feature and
hyperparameter decision is locked. Adjusting anything because the 2026 numbers
looked wrong makes 2026 a validation set and the final evaluation dishonest.

### 0.2 Run-value lookup

`RE(outcome, count)` = mean Statcast `delta_run_exp` by `(des_new, count)`.
**EDA §3 settles two things**: `delta_run_exp` is a deterministic
base–out–count lookup (within-group std 0.00000000), and the table drifts by
≤0.014 runs across all six seasons — so one table, no recency weighting.
Compute it on the training seasons of whichever model is being fit (2021–24 for
Model A, 2021–25 for Model B), apply the same table to every season that model
scores, and never let 2026 into it. v3 recomputes it per year inside
`df_clean`, so 2024 targets use 2024 means. Map `field_error` separately rather
than to `field_out`. Keep HBP as its own outcome.

### 0.3 Pitch frame (used by both tracks)

| Feature | Why |
|---|---|
| `plate_x_b = −x_ref if stand == 'R' else x_ref` | Statcast `plate_x` is catcher-relative; `+0.7` is outside to RHB, inside to LHB. Positive = inside for everyone. Built on the harmonized `x_ref` (§0.1.1), not raw `plate_x`. |
| `plate_z_n = (z_ref − sz_bot) / (sz_top − sz_bot)` | 3.4 ft is a strike to a 6'4" hitter and a ball to a 5'7" one. Height-derived zone bounds (§0.1.1). |
| `stand`, `p_throws` | Lefty-strike asymmetry; platoon. |
| `release_speed`, `pfx_x`, `pfx_z`, `pitch_type` | Observable at decision time. Main role is reducing swing-side confounding (hitters swing when they see the pitch, take when fooled). |
| `count` | as now |

Not used as features: anything realized during or after the swing (bat speed on
that swing, EV, LA) — execution, not decision.

### 0.4 Decision quantities (identical definition in both tracks)

```
Δ(s)      = Q_swing(s) − Q_take(s)                 # positive → swing is better
regret_i  = max(Q_swing, Q_take) − Q_chosen        # ≥ 0
```

Player metric = mean regret per 100 pitches (lower is better), reported as
SOTO+-style 100 ± 10 (sign flipped so higher = better). Secondary: correct
decision %, chase regret (swings with Δ<0), missed-opportunity regret (takes
with Δ>0). Qualification: ≥ 500 pitches.

This replaces v3 cells 74–75 (takes-only rate) and v3 cell 79
(`swing·P(strike) − take·(1−P(strike))`, which applies `P(ball)` twice because
`Q_take` already contains it).

### 0.5 Evaluation harness (run on every model, every track)

Sub-model level (rolling-origin CV per §0.1.2, final on 2026 per §0.1.3):
- Called strike: log loss, Brier, reliability diagram; by count, `stand`, zone
  region.
- Swing outcome (Track B): multiclass log loss, per-class calibration.
- Run-value regressions: RMSE **against a count-only baseline** (the target is
  a function of `(outcome, count)`, so count-only is the floor; location's
  contribution is the gap).
- Counterfactual support: bin taken pitches by called-strike-prob decile ×
  count, count training swings per bin; bootstrap (5–10 refits) the swing side
  and report the share of pitches whose Δ flips sign.

Player-metric level:
- Split-half reliability within season (odd/even PAs, Spearman–Brown).
- YoY R² for each of the four season pairs (22→23, 23→24, 24→25, 25→26).
- Regime comparison: Model A's 2025 vs. 2026 readout (§0.1.3).
- Note when a season's player metric is in-sample for the model scoring it
  (mild optimism; small here since the league models carry no batter identity).
- Predictive validity: season-*t* metric → season-*t+1* wOBA, BB%, K%, chase
  rate, controlling for season-*t* wOBA.
- **Zone% test**: `|corr(metric, Zone%)|` — the SOTO failure mode. Report for
  every variant; a standardized metric should drive this toward zero.
- Benchmarks on the same hitters: O-Swing%, Z-O-Swing%, Statcast Swing/Take
  runs, Creally five-zone linear weights, and v1.

---

## Track A — patched two-model baseline

Goal: the corrected version of what exists. Small, sequential, each step scored
on the harness. Expected to recover ≥ v1's stability with the personalization
idea intact.

### A1. Metric fix (§0.4) with v1 features
Establishes the baseline number. Target: YoY R² ≥ 0.57 on 22→23.

### A2. Pitch frame (§0.3)
Add in three steps and score each: zone frame + handedness → pitch
characteristics. Keep whichever help across the CV folds (§0.1.2).

### A3. Take model: verify or replace
The take target is `RE(ball|CS|HBP, count)`, so given location and count the
only thing the take model learns from location is `P(CS | s)`; count-specific
run value comes from the lookup via the `count` feature. Verify on the CV
validation folds:
regress `take_pred` on `called_strike_prob` within each count — expect a
near-perfect line (slope ≈ `RE(CS,c) − RE(ball,c)`). If R² < ~0.95, replace the
LightGBM take regressor with the structural form
`Q_take = P(CS|s)·RE(CS,c) + (1−P(CS|s))·RE(ball,c)` using the called-strike
classifier (this is Track B's take model; the tracks converge here).

Either way one model owns `P(CS | s)`; the separate called-strike model is not
multiplied against the take model.

### A4. Nitro zone — **done, as the v2 rebuild** (`notebooks/v2_baseline.ipynb`)

The de-leaked hull is built and scored. Three results, one of which overturns
an assumption this plan carried.

**The hull is not the problem.** v2 rebuilt edges v1 on every measure over the
same 2022–2026 window: split-half 0.786 vs 0.753, YoY R² 0.530 vs 0.517, Zone%
|r| 0.033 vs 0.077, next-season partial r 0.105 vs 0.095. The historical
collapse from 0.57 to 0.34 was **the leak**, compounded by a simultaneous
switch of learner and hyperparameters — not evidence against the feature. v2's
idea was never actually tested by v2.

This corrects the framing above and in B4: EDA §6 measured hull-class
estimators as poor at reproducing the *hot-zone signal* (r ≈ 0.30 vs 0.68), and
that remains true, but it does not follow that the hull damages the *metric*.
It does not.

**`in_nitro` is largely a location proxy.** 91% of flagged pitches are in the
strike zone and it correlates 0.51 with zone membership, which is why it takes
81–89% of model gain while moving RMSE only in the fourth decimal — the tree
uses it as a cheap "middle of the zone" split. Genuine personalization is
present but is the smaller part: between-hitter std within a location cell is
0.247, cell means span 0.00–0.70.

**It contributes nothing to the take model**, as predicted: take RMSE 0.04283
without it, 0.04298 with it — marginally worse, i.e. noise. Swing-model-only
routing is a design change rather than a correction, so it belongs to the patch
work, but the evidence for it is now measured rather than assumed.

**New defect, introduced by the fix itself.** Hull size scales with accumulated
history: median prior balls in play go 155 (2022) → 285 (2026), so the top-5%
set grows from ~7 points to ~14 and `in_nitro` fires on 14.5% of pitches in 2022
against 26.1% in 2026. The same hitter with an unchanged hot zone gets a larger
flagged region in later seasons. **Any prior-window feature needs a fixed
window or a fixed-area rule**, and this applies to B4's surface too.

Coverage: 79–85% of qualified hitters have a prior-season hull (worst in 2022,
whose only prior is 2021); the rest score `False`. 2021 cannot be scored at all,
so v2 covers 2022–2026 and four YoY pairs, and v1 is re-reported on that window
for the comparison.

### A5. Deliverable
"v4a": corrected SOTO-v2-class model with a full harness readout. This is the
row Track B is compared against.

---

## Track B — event-decomposition redesign

Goal: get past the ceiling of outcome regression (Salorio retired SOTO because
Zone% explained ~23% of its variance and location+count cannot learn swing
value). Structure follows EAGLE / Yee–Deshpande, in gradient boosting.

### B1. Sub-models

```
Q_take(s)  = P(CS | s) · RE(CS, c)  +  (1 − P(CS | s)) · RE(ball, c)      [+ HBP term]

Q_swing(s) = P(whiff | s) · RE(whiff, c)
           + P(foul  | s) · RE(foul, c)
           + P(BIP   | s) · E[RE | BIP, s]
```

| Sub-model | Type | Rows | Features |
|---|---|---|---|
| Called strike | binary GBM | takes | §0.3 + (optional) catcher, umpire |
| Swing outcome | 3-class GBM (whiff / foul / BIP) | swings | §0.3 |
| Contact quality `E[RE \| BIP, s]` | regression GBM, target `RE(event, c)` | balls in play | §0.3 (+ hitter features in B4) |
| Run values | lookup | — | §0.2 |

Why this beats the direct regression: whiff/foul/BIP probabilities are
well-supported and well-identified; contact quality is the only hard,
hitter-dependent piece and it is now isolated and separately calibratable.
The v3 swing RMSE of 0.297 currently hides which part is failing.

Bootstrap the swing-side sub-models (5–10 refits) for a confidence band on Δ;
flag pitches whose Δ sign is unstable. This substitutes for BART's posterior at
a fraction of the cost.

### B2. Decision value
Compute `Q_swing`, `Q_take`, Δ, regret (§0.4). Generic version first (no hitter
features anywhere). Harness readout; compare with v4a.

### B3. Opportunity standardization
The piece the public metrics lack.

1. **Direct standardization (first).** Stratify pitches by count × attack zone
   (heart / shadow-in / shadow-out / chase / waste). Compute each hitter's
   regret per stratum, reweight by league stratum frequencies. Report
   observed-opportunity and standardized leaderboards side by side. Acceptance:
   Zone% test drops materially vs. the observed-opportunity version.
2. **Policy evaluation (if 1 leaves residual bias).** Fit
   `π_h(s) = P(swing | s, hitter)` (GBM with hitter effect, or per-hitter
   residual model). Standardized skill =
   `E_{s∼ref}[ π_h(s)·Q_swing(s) + (1−π_h(s))·Q_take(s) − max(Q_swing, Q_take) ]`
   over a common reference pitch distribution. Separates "what his decisions
   cost this year" from "how good he is at deciding."

### B4. Personalized contact quality

**The hot zone is a real, stable trait; the v2 hull was just the worst way to
estimate it.** EDA §6 measured year-over-year reliability of the per-hitter
exit-velocity surface three ways:

| estimator | YoY r |
|---|---|
| raw 4×4 cells, no smoothing or shrinkage | **0.28–0.33** |
| kernel-smoothed + empirical-Bayes shrunk | **0.64–0.68** |
| same, 2-season prior window | **0.71** |

A top-5% hull is worse than the first row: it discards 95% of the balls in play
and leaves a polygon defined by a handful of points. That is the mechanism
behind the v1→v2 stability drop.

Recipe, with the settings that produced those numbers:

1. **Every ball in play**, continuous response (exit velocity or xwOBAcon),
   never a top-k threshold.
2. **Kernel-smooth over location**, bandwidth ≈ 0.35 ft in the batter frame
   (`plate_x_bat`, `plate_z_norm`) — a hot zone is spatially smooth, so each
   ball in play informs its neighbourhood.
3. **Empirical-Bayes shrink toward the league surface** by effective sample
   size (k ≈ 60); low-sample hitters revert to league, not to noise.
4. **Pool a multi-season prior window** — 2 seasons beats 1 (0.71 vs 0.68), and
   it is free since seasons < t are required anyway.
5. Optionally **parameterise low-rank**: hitter surfaces are nearly
   3-dimensional (PC1 60.5%, PC2 21.0%, PC3 7.7% — 89% in three, 96.5% in
   five), so ~3 coefficients per hitter is more stable and cheaper than a free
   grid. This is also *why* shrinkage recovers so much.

**Acceptance criterion: YoY r ≈ 0.7 before the feature enters the swing
model.** A feature less reliable than the metric it is meant to improve cannot
help it.

If reliability needs more, two sources are still unexploited: all swings rather
than only balls in play (whiffs and fouls carry coverage information), and bat
tracking from 2024 (bat speed, attack angle) as a physical prior on where a
hitter generates power.

Add prior-season hitter features to the contact-quality sub-model only:
- The shrunken, smoothed surface above (continuous replacement for the hull;
  degrades to league average for rookies).
- Rolling prior-season contact%, whiff%, damage rate.
- Prior-season bat speed. 2024 is the first season it exists, so it is
  available as a prior for 2025 and 2026 only; keep it out of earlier seasons
  rather than imputing.

Report **generic** and **personalized** decision value as two metrics
(related-work §5.3). A power hitter can correctly swing at a pitch that is
negative-value for an average hitter; both numbers are informative.

### B5. Overlap / confounding diagnostics
- Swing propensity `P(swing | s)`; overlap plots by location × count.
- Trim or flag extreme-propensity pitches.
- Compare direct contact-quality regression vs. IPW-weighted; if they diverge,
  report both (or AIPW).

### B6. Deliverable
"v4b": generic + personalized + standardized decision value, with uncertainty,
full harness readout, and benchmark comparison against v4a, v1, Creally,
SEAGER/SwRV-style metrics, and Statcast Swing/Take.

---

## Baselines to reproduce (cheap, do early)

**Every baseline must be re-run on the corrected pipeline.** The historical
0.57 / 0.34 / 0.16 were measured on the old data path — overseas games
included, pitchers batting, run-value tables refit per season, `field_error`
mapped to `field_out`, a single 2022→2023 pair, and no 2026 harmonization.
A new v4 number compared against a historical 0.57 compares two different
measurements. The scorecard is only meaningful when every row comes from the
same harness.

- **Creally five-zone linear weights**: mean swing and take value per count ×
  attack zone; hitter score = sum over decisions. Transparent floor.
- **v1**, the pre-nitro-zone design that scored YoY R² = 0.57. Full spec below,
  so it is reproducible without reading the retired notebook.

  | | v1 |
  |---|---|
  | features | `plate_x`, `plate_z`, `count` (categorical) |
  | target | `delta_run_exp_mean` — mean `delta_run_exp` by (outcome, count) |
  | models | two XGBoost regressors, split on `swing` |
  | take params | `max_depth=8, learning_rate=0.03, objective=reg:squarederror, tree_method=hist`, 200 rounds |
  | swing params | `max_depth=7, learning_rate=0.01`, otherwise identical, 200 rounds |
  | per-pitch score | `y_pred` = prediction of the model matching the action actually taken |
  | player metric | mean `y_pred` over **all** pitches, z-scored, ×10 + 100 |
  | qualification | ≥ 500 pitches |
  | seed | 1126 |

  **Reimplement this on `src/data.py`, do not run the retired notebook.** v1
  carries the data defects the rebuild fixes — per-season run-value tables, no
  pitcher-batting filter, `field_error` mapped to `field_out`, and no 2026
  harmonization, which would silently pool front-of-plate with middle-of-plate
  locations. Running it verbatim would confound v1's design with v1's bugs; the
  point of the baseline is to isolate the design.

  The retired notebooks are at `fa48b14` if anything above needs checking:
  `git show fa48b14:batter_decision_value.ipynb`.
- **v2 = v1 + `in_nitro`**, run twice, to decompose the historical 0.57 → 0.34
  drop into its two confounded causes:

  | run | hull built from | isolates |
  |---|---|---|
  | v2-leaky | the same season it scores | the leak's contribution |
  | v2-clean | prior seasons only (this is A4) | the noisy-estimator contribution |

  EDA §6 established that a hull is a poor *estimator* of the hot-zone signal
  (raw cells reproduce at r ≈ 0.30 against r ≈ 0.68 smoothed and shrunk), but
  that is feature reliability, not the metric-level effect, and it says nothing
  about how much of the v2 damage was leakage. These two runs answer both, and
  v2-clean is needed for the scorecard regardless since it is Track A4.

- **O-Swing%, Z-O-Swing%** from the same data.

---

## v1 baseline: what it established

Run in `notebooks/v1_baseline.ipynb` (code in `src/baselines.py`,
`src/evaluate.py`).

- **The reimplementation is faithful.** 2022→2023 — the only pair the
  historical 0.57 covered — comes in at R² = 0.578 on a completely different
  pipeline. That agreement is what licenses calling this v1.
- **Reliable**: YoY R² 0.43–0.58, split-half 0.75.
- **Not confounded by pitch mix.** Zone% explains 0.1–1.2% of the metric's
  variance, against the ~23% that retired SOTO. The standing assumption that
  v1's mean-of-chosen-action is badly pitch-mix confounded **does not hold at
  that magnitude** — v1 is a stronger baseline than assumed, and Track A and B
  have a higher bar to clear.
- **Barely valid.** Partial r against next-season production is 0.076, and the
  2025→2026 pair is slightly negative. v1 is reliable without being useful.
  **This is the number to beat**, not the YoY figure.
- **The weakness is located.** The swing model beats a count-only lookup by
  0.8%; the take model by 43.6%. v1's decision value is driven almost entirely
  by its take model, which is a called-strike probability in disguise. Track B
  targets exactly this.
- Face validity holds: 2026 top is Soto, Torres, Tucker, Seager, Acuña; bottom
  is Báez, Sosa, Story. Contrast v3, which put Arráez and Hoerner at the
  bottom.

## v3 (the patch): what it established

`notebooks/v3.ipynb`. Four steps, each changing one thing.

**A1 — the metric was the whole patch, and the simplest score won.** Construct
validity goes from −0.849/+0.594 to −0.905/+0.831 on v1's features purely by
changing how per-pitch scores aggregate. The winner is `correct_decision`: +1
if the hitter picked the better action, −1 if not, **no run-value weighting**.
Every magnitude-weighted alternative is more pitch-mix contaminated, because
the magnitude is what varies with the pitches a hitter sees — `signed_edge`
pays 5× more for an obvious take than a close call (Zone% 0.276), and
**`regret`, which §0.4 specified, fails the Zone% veto outright at 0.468** with
next-season partial r of 0.006. §0.4 should be amended.

**A2 — the swing model cannot be helped.** 0.8% over a count-only lookup with
location; **0.9%** after adding batter-frame location, handedness, velocity,
movement and pitch type. The take model gains steadily (43.7% → 46.8%). Nothing
observable before the pitch arrives predicts what happens once a hitter swings.
Features still improve construct validity (−0.905/+0.831 → −0.927/+0.871), so
sub-model fit and metric quality are separate questions.

**A3 — the take model is a called-strike probability.** Median R² **0.991**
regressing `take_pred` on a fitted `P(CS)` within each count, slopes matching
`RE(CS,c) − RE(ball,c)` to three decimals (3-2: −0.611 vs −0.614). It is
learning an umpire, not a run-value surface. The structural form can be adopted
for interpretability at no cost in accuracy, and Tracks A and B converge here.

**A4 — personalization is one of the largest effects here, and A1's metric hid
it.** Under `correct_decision` all three variants are identical to ±0.002, which
looks like the feature being worthless. It is not:

- Hitters differ at the same location by **1.78 mph** of expected exit
  velocity, about **0.6× the entire league-wide location effect** (87.3–90.3 mph
  across the zone), and they differ in *where* their best region sits — the
  modal best cell holds only 20% of hitter-seasons.
- The feature moves `Q_swing` by SD 0.0096 runs against `Q_swing`'s own 0.0407,
  a quarter of its scale — but flips the *recommended action* on only **1.7%**
  of pitches. `correct_decision` is a sign, so 98% of the effect cannot reach it.

Measured against metrics that can see it:

| metric | split-half | YoY R² | next-season r |
|---|---|---|---|
| `chosen_value` | 0.751 → **0.888** | 0.507 → **0.690** | 0.099 → **0.228** |
| `signed_edge` | 0.818 → 0.836 | 0.587 → 0.618 | 0.069 → 0.105 |
| `correct_decision` | 0.815 → 0.816 | 0.566 → 0.564 | 0.091 → 0.091 |

Under `chosen_value` the gains are large but construct validity *degrades*
(−0.864 → −0.643) — the feature pulls the score toward measuring hitting
ability, which is also why it predicts production so much better. Under
`signed_edge` reliability, contamination and usefulness all improve together
with construct validity barely moving, which makes it the best overall pairing.

**A1 and A4 are not independent.** `correct_decision` still has much the best
construct validity (−0.943/+0.898) and dropping magnitude is still what removes
the pitch-mix bias — but the same insensitivity makes it unable to benefit from
an informative feature. Testing a feature against one metric was the error.

**Consequences for Track B.** The event decomposition's premise is in doubt —
A2 puts the swing-side ceiling very low regardless of model form. But **B4's
personalized contact surface is vindicated**, and is the most valuable feature
found here once measured with a metric that can register it. The take-side result is the usable one.

## Execution order

| Step | Track | Output |
|---|---|---|
| 1 | 0 | 2025 + 2026 pulls, `game_type` filter, trajectory harmonization (§0.1.1), RE lookup, pitch-frame features, `src/` skeleton, harness, rolling-origin CV scaffold |
| 2 | baselines | Creally + v1 + v2-leaky + O-Swing% on the harness, all re-run on the corrected pipeline |
| 3 | A1–A2 | metric fix; pitch-frame features scored |
| 4 | A3 | take-model verification → structural `Q_take` (shared with B) |
| 5 | B1–B2 | swing-outcome + contact-quality sub-models; generic v4b |
| 6 | A4 | prior-season hull, swing-only `in_nitro` → v4a complete (no stability test; EDA §6 answered it) |
| 7 | B3 | direct standardization + Zone% test |
| 8 | B4 | personalized contact quality; generic vs personalized readout |
| 9 | B5 | overlap diagnostics; IPW comparison |
| 10 | B3.2 | policy-evaluation version if needed |
| 11 | all | Model A/B refit (§0.1.3), one-shot 2026 evaluation, ABS regime comparison; README/CLAUDE.md updated to match |

Steps 3–4 and 5 can run in parallel; A3's structural take model is reused by B.

---

## Scorecard

Fill one row per variant; all numbers from the same harness. Validate columns
from the CV folds during development; 2026 only at the end, once per model.

Last column is the partial correlation between season-*t* decision value and
season-*t+1* ΔRE per plate appearance, controlling for season-*t* ΔRE/PA.

| Variant | CS log loss | Swing-side RMSE vs count-only | Split-half r | YoY R² (22→23 / 23→24 / 24→25 / 25→26) | Zone% \|r\| | Next-season partial r |
|---|---|---|---|---|---|---|
| v3 historical (old pipeline, not comparable) | — | 0.296 | — | 0.16 | — | — |
| v1 historical (old pipeline, not comparable) | — | — | — | 0.57 | — | — |
| **v1 re-run** | n/a | **0.2985 / 0.3010 (+0.8%)** | **0.75** | **0.58 / 0.56 / 0.51 / 0.43** | **0.077** | **0.092** |
| v1 re-run, 2022–26 window | n/a | 0.2974 / 0.2998 | 0.753 | 0.58 / 0.56 / 0.51 / 0.43 | 0.077 | 0.095 |
| **v2 rebuilt (prior-season hull)** | n/a | **0.2973 / 0.2998** | **0.786** | **0.57 / 0.56 / 0.52 / 0.47** | **0.033** | **0.105** |
| Creally 5-zone | | | | | | |
| **v3 rebuilt (the patch)** | n/a | **0.2982 / 0.3010** | **0.806** | **0.56 mean** | **0.207** | **0.085** |

v3's row uses the `correct_decision` metric and the full pitch frame. Its
construct validity — the check that actually discriminates — is **−0.924 /
+0.871** (chase and zone-swing, each partialled on the other) against v1's
−0.849 / +0.594. That is the patch's actual gain; the columns above barely
move.
| B2 v4b generic | | | | | | |
| B3 v4b standardized | | | | | | |
| B4 v4b personalized | | | | | | |

---

## Repo structure

```
src/
  data.py        load, game_type filter, trajectory harmonization, cleaning, RE lookup
  features.py    pitch frame, attack zones, hitter priors (nitro hull, EV surface)
  models_a.py    Track A: two regressors
  models_b.py    Track B: called-strike, swing-outcome, contact-quality
  decision.py    Q_swing, Q_take, Δ, regret, standardization, policy eval
  evaluate.py    harness (§0.5)
notebooks/       one narrative notebook per track + one for comparison
data/            gitignored parquet cache
```

`requirements.txt`, `.gitignore`, remove the hardcoded `C:/Users/...` path,
clear notebook outputs before commit.

---

## Settled decisions (do not re-open without new evidence)

- Grade decisions on the pitch; `stand` / `sz_top` / `sz_bot` define ball
  position in the batter's frame and are not personalization.
- `in_nitro` / hitter contact features enter the swing side only; the take
  outcome does not depend on contact quality. Verified by the with/without
  check in A4.
- `Q_take` already contains `P(ball)`; never multiply it by `(1 − P(CS))`.
- The swing model learns a league-average, pitch-averaged counterfactual;
  personalization is a separate, explicitly labeled metric.
- Random pitch-level split was not leaking in v3 (train ≈ test RMSE); the
  chronological split is adopted for the prior-season hitter features and for
  the out-of-regime 2026 test, not because of v3 leakage.
- A feature can be a poor *estimator of its signal* and still not harm the
  *metric*. The hull reproduces the hot-zone signal at r ≈ 0.30 against 0.68
  for a smoothed surface, yet the rebuilt v2 slightly beats v1. Measure the
  metric, not just the feature.
- Prior-window features drift when the window grows: the hull's flagged area
  nearly doubles from 2022 to 2026 purely from accumulated history. Fix the
  window or the area.
- A noisy *estimate* is not the same as an absent *signal*. The per-hitter hot
  zone goes from YoY r ≈ 0.30 (raw cells) to ≈ 0.68 (smoothed + shrunk) to
  ≈ 0.71 (2-season prior) with no new data. Judge a feature by the best
  available estimator, not the naive one.
- Thin counterfactual support is not by itself a problem: it coincides with
  unambiguous decisions (3-0 off the plate), where a large |edge| makes the
  call robust to estimation error. Judge support by ambiguity, not raw counts.
- 2026 is usable despite ABS. Harmonize location to middle-of-plate first
  (§0.1.1): the measured shift is ~1 inch vertically with a ~0.8 in
  pitch-type-dependent spread — modest, but it tracks pitch type, so a constant
  offset will not absorb it and it lands where called-strike probability is
  steepest. Extending the data helps the hitter-specific surfaces, not the
  league sub-models, which are already saturated.
- Two models are kept at the end (A: 2021–24, B: 2021–25) so the in-regime
  reference survives the final refit; 2026 is scored once per model.
- Realized bat speed / EV on the swing being graded is never a feature.
- `automatic_ball` / `automatic_strike` are excluded as non-decisions (pitch
  clock and intentional walks, ~13.9k rows over six seasons); `field_error` is
  never mapped to `field_out` (+0.461 vs −0.250 runs).
- 2021 needs the pitcher-batting filter (402 batters); 2022+ needs it only
  marginally (16–30) but it is applied uniformly.
- The lefty strike shrinks by only ~20% under the ABS *challenge* system
  (−0.110 → −0.086), so handedness stays in the feature set for 2026.
- ABS judges "any part of the ball", not the ball's centre — the same
  convention as the rulebook, so zone membership is computed identically in
  every season. The 2026 difference is the nominal zone (2.64 in lower at the
  top), not the convention.
