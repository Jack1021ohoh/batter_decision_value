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

Five regular seasons, 2021–2025. Stop at 2025: the ABS challenge system starts
in 2026 and changes the called-strike process.

| Role | Seasons | Use |
|---|---|---|
| Train | 2021–2023 | fit all models |
| Validate | 2024 | calibration, early stopping, model selection, feature decisions |
| Test | 2025 | final evaluation only; not touched during development |

- Add a 2025 pull to `data_fetch.ipynb` (`statcast('2025-03-18','2025-09-28')`)
  and filter `game_type == 'R'` for every season.
- Drop pitchers batting (2021 NL). Drop 3-strike / 4-ball rows as now.
- Hitter-level features (nitro zone, contact-quality surface, prior bat speed)
  for season *t* are built from seasons < *t* only.
- Stability checks use three season pairs (22→23, 23→24, 24→25), not one.

### 0.2 Run-value lookup

`RE(outcome, count)` = mean Statcast `delta_run_exp` by `(des_new, count)`,
computed **once on 2021–2023** and applied to every season (v3 recomputes it
per year inside `df_clean`, so 2024 targets use 2024 means). Map `field_error`
separately rather than to `field_out`. Keep HBP as its own outcome.

### 0.3 Pitch frame (used by both tracks)

| Feature | Why |
|---|---|
| `plate_x_b = −plate_x if stand == 'R' else plate_x` | Statcast `plate_x` is catcher-relative; `+0.7` is outside to RHB, inside to LHB. Positive = inside for everyone. |
| `plate_z_n = (plate_z − sz_bot) / (sz_top − sz_bot)` | 3.4 ft is a strike to a 6'4" hitter and a ball to a 5'7" one. |
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

Sub-model level (validate on 2024, final on 2025):
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
- YoY R² for each of the three season pairs.
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
characteristics. Keep whichever help on 2024.

### A3. Take model: verify or replace
The take target is `RE(ball|CS|HBP, count)`, so given location and count the
only thing the take model learns from location is `P(CS | s)`; count-specific
run value comes from the lookup via the `count` feature. Verify on 2024:
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
- Prior-season bat speed (2025 only; 2024 is the first season it exists).

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
| 1 | 0 | 2025 pull, `game_type` filter, single RE lookup, pitch-frame features, `src/` skeleton, harness |
| 2 | baselines | Creally + v1 + O-Swing% on the harness |
| 3 | A1–A2 | metric fix; pitch-frame features scored |
| 4 | A3 | take-model verification → structural `Q_take` (shared with B) |
| 5 | B1–B2 | swing-outcome + contact-quality sub-models; generic v4b |
| 6 | A4 | prior-season hull, stability test, swing-only `in_nitro` → v4a complete |
| 7 | B3 | direct standardization + Zone% test |
| 8 | B4 | personalized contact quality; generic vs personalized readout |
| 9 | B5 | overlap diagnostics; IPW comparison |
| 10 | B3.2 | policy-evaluation version if needed |
| 11 | all | final 2025 evaluation; README/CLAUDE.md updated to match |

Steps 3–4 and 5 can run in parallel; A3's structural take model is reused by B.

---

## Scorecard

Fill one row per variant; all numbers from the same harness. Validate columns
on 2024 during development; 2025 only at the end.

| Variant | CS log loss | Swing-side RMSE vs count-only | Split-half r | YoY R² (22→23 / 23→24 / 24→25) | Zone% corr | Next-yr wOBA partial r |
|---|---|---|---|---|---|---|
| v3 as-is | — | 0.296 / ? | ? | 0.16 / ? / ? | ? | ? |
| v1 | — | | | 0.57 / ? / ? | | |
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
  data.py        load, game_type filter, cleaning, RE lookup
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
  chronological split is adopted for the prior-season hitter features, not
  because of v3 leakage.
- Realized bat speed / EV on the swing being graded is never a feature.
