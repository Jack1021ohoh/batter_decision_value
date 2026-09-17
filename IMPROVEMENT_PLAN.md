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

- Add pulls for 2025 and 2026 to `data_fetch.ipynb`; filter
  `game_type == 'R'` for every season. 2026 is ~95% complete as of
  2026-09-17 — re-pull after the regular-season finale.
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
fixed percentage band of height (reported as roughly 27%–53.5% at the middle
of the plate) — **verify the exact figures before relying on them.** Sanity
check already observed: 2026 `sz_top` has std 0.102 ft and `sz_bot` 0.051 ft,
far tighter than operator-set values, consistent with a deterministic
height-based zone.

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

### A4. Nitro zone without leakage
- Build the hull from prior seasons only (≥ 150 BIP across those seasons).
- Feed `in_nitro` to the swing model only. Check first: train the take model
  with and without it; expect identical RMSE to three decimals.
- Stability of the zone itself: `area(hull_t ∩ hull_{t+1}) / area(hull_t ∪ hull_{t+1})`
  per hitter. Low overlap = noise, not a trait.
- Fix `add_nitro_zone`'s inner merge (left merge, `in_nitro = False` when no
  hull); fix `plot_nitro_zone`'s `iloc[0, −2]`.

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
Add prior-season hitter features to the contact-quality sub-model only:
- Shrunken EV / xwOBAcon surface over `(plate_x_b, plate_z_n)`, kernel-smoothed,
  empirical-Bayes toward the league surface by BIP count (continuous
  replacement for the hull; degrades to league average for rookies).
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

- **Creally five-zone linear weights**: mean swing and take value per count ×
  attack zone; hitter score = sum over decisions. Transparent floor.
- **v1** as-is (location + count, mean chosen-action value) on the new split.
- **O-Swing%, Z-O-Swing%** from the same data.

---

## Execution order

| Step | Track | Output |
|---|---|---|
| 1 | 0 | 2025 + 2026 pulls, `game_type` filter, trajectory harmonization (§0.1.1), RE lookup, pitch-frame features, `src/` skeleton, harness, rolling-origin CV scaffold |
| 2 | baselines | Creally + v1 + O-Swing% on the harness |
| 3 | A1–A2 | metric fix; pitch-frame features scored |
| 4 | A3 | take-model verification → structural `Q_take` (shared with B) |
| 5 | B1–B2 | swing-outcome + contact-quality sub-models; generic v4b |
| 6 | A4 | prior-season hull, stability test, swing-only `in_nitro` → v4a complete |
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

| Variant | CS log loss | Swing-side RMSE vs count-only | Split-half r | YoY R² (22→23 / 23→24 / 24→25 / 25→26) | Zone% corr | Next-yr wOBA partial r |
|---|---|---|---|---|---|---|
| v3 as-is | — | 0.296 / ? | ? | 0.16 / ? / ? / ? | ? | ? |
| v1 | — | | | 0.57 / ? / ? / ? | | |
| Creally 5-zone | — | | | | | |
| A1 metric fix | | | | | | |
| A2 + pitch frame | | | | | | |
| A4 v4a (prior nitro) | | | | | | |
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
- 2026 is usable despite ABS. Harmonize location to middle-of-plate first
  (§0.1.1): the measured shift is ~1 inch vertically with a ~0.8 in
  pitch-type-dependent spread — modest, but it tracks pitch type, so a constant
  offset will not absorb it and it lands where called-strike probability is
  steepest. Extending the data helps the hitter-specific surfaces, not the
  league sub-models, which are already saturated.
- Two models are kept at the end (A: 2021–24, B: 2021–25) so the in-regime
  reference survives the final refit; 2026 is scored once per model.
- Realized bat speed / EV on the swing being graded is never a feature.
