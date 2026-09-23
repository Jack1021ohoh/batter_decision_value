# Findings

What the data established, separate from what we planned to do about it. Every
number here was measured in one of the notebooks and can be reproduced by
running it.

**Every model result is held out.** Each variant is fitted on rolling-origin
folds and each of 2023, 2024 and 2025 is scored by a model that never saw it;
year-over-year and next-season checks use the 2023→24 and 2024→25 pairs.
Generic variants train from 2021; whenever a personalized variant is in a
comparison, every row trains from 2022 and uses 2021 only as prior data. 2026
has not been scored — it is the final test, run once after every decision is
locked.

---

## Results

Two readings, and they answer different questions.

**As each version was designed** — v1 and v2 score the value of the action
taken, because that is their design. v2 is compared with v1 refit on the same
(personalized) folds.

| | chase\|zs | zs\|chase | split-half r | YoY R² | Zone% \|r\| | next-season partial r |
|---|---|---|---|---|---|---|
| v1 | −0.884 | 0.673 | 0.758 | 0.526 | 0.082 | 0.135 |
| v1, v2's folds | −0.883 | 0.669 | 0.757 | 0.526 | 0.081 | 0.137 |
| v2 | −0.792 | 0.535 | 0.792 | 0.535 | 0.031 | 0.135 |

**With the metric held fixed at `signed_edge`**, so only the feature set
varies and each difference is attributable — this is the comparison that
supports any claim about the patch. Full table in "The feature ladder" below.

| | chase\|zs | zs\|chase | split-half r | YoY R² | Zone% \|r\| | next-season partial r |
|---|---|---|---|---|---|---|
| v1 features | −0.920 | 0.699 | 0.818 | 0.594 | 0.300 | 0.082 |
| v3 features + hot zone | −0.895 | 0.677 | **0.838** | **0.624** | **0.184** | **0.154** |

The patch **nearly doubles** next-season predictive validity, cuts pitch-mix
contamination by 39% and raises reliability. It costs some construct validity
— whether the metric punishes chasing *and* rewards attacking hittable
pitches, each holding the other fixed — which slips by 0.02–0.03 on each half.

In-sample scoring would have flattered v3 only modestly: YoY R² 0.643 against
0.624 held out, Zone% 0.223 against 0.184.

---

## Choosing the per-pitch score

Every candidate is built from the same counterfactual pair, `q_swing` and
`q_take`, so the models are identical and only the aggregation differs.

| score | definition |
|---|---|
| `chosen_value` | value of the action taken |
| **`signed_edge`** | **`Q_chosen − Q_alternative` — how much better the choice was** |
| `regret` | `max(0, −signed_edge)`; zero whenever the hitter was right |
| `close_weighted` | signed correctness weighted by how close the call was |
| `correct_decision` | ±1, no magnitude |

On v1's features, generic folds:

| score | chase\|zs | zs\|chase | split-half | YoY R² | Zone% \|r\| | next-season |
|---|---|---|---|---|---|---|
| `chosen_value` | −0.884 | 0.673 | 0.758 | 0.526 | 0.082 | 0.135 |
| `signed_edge` | −0.921 | 0.701 | 0.819 | 0.594 | 0.301 | 0.085 |
| `regret` | −0.825 | 0.527 | 0.791 | 0.572 | 0.504 | 0.027 |
| `close_weighted` | −0.717 | 0.830 | 0.641 | 0.345 | 0.050 | 0.098 |
| `correct_decision` | **−0.954** | **0.914** | **0.825** | 0.586 | 0.241 | 0.113 |

**By the criterion fixed before the numbers — construct validity first —
`correct_decision` wins**, and still does with the hot zone added (−0.964 /
+0.931). As `close_weighted`'s kernel widens it converges on `correct_decision`
(correlation 0.997 at scale 0.8), improving the whole way. The run-value
magnitude is what carries the pitch-mix bias: `signed_edge` pays about eight
times more per pitch for an obvious take (+0.082 runs) than for getting a
genuinely close call right (+0.010).

**`signed_edge` is selected anyway, as a trade.** It keeps the run-value
magnitude, which is what every published metric uses — SwRV, SOTO, Nestico's
Decision Value, Creally's wDV, EAGLE all report run value per 100 pitches. A
sign-based score treats a razor-thin call and an obvious blunder alike, and it
barely registers a feature that changes how much a swing is worth without
flipping the decision. With the hot zone, `signed_edge` is the more reliable of
the two (split-half 0.841 against 0.828, YoY R² 0.629 against 0.583), less
contaminated on raw Zone% (0.185 against 0.215) and level on next-season
validity (0.157 against 0.150).

### The contamination problem

Magnitude-weighted scores can be contaminated by pitch mix, because the
magnitude varies with the pitches a hitter is thrown. Measured on the final
model (v3), with Zone% correlated against the score and something held fixed:

| score | raw | \| correct-decision rate | \| chase, zone-swing | \| production |
|---|---|---|---|---|
| `chosen_value` | −0.180 | −0.417 | −0.496 | −0.052 |
| `signed_edge` | 0.184 | −0.006 | −0.236 | 0.311 |
| `regret` | 0.547 | 0.603 | 0.527 | 0.585 |
| `correct_decision` | 0.213 | *(circular)* | 0.066 | 0.307 |

**The answer depends on the control, so no control settles it.** Each is a
screen, not proof: the correct-decision rate comes from the same model's Δ (and
*is* `correct_decision`), and chase and zone-swing rates vary with how hard the
pitches a hitter sees are. The only test with ground truth is a known-policy
simulation.

**The raw Zone% test is lenient.** Better hitters are thrown fewer strikes
(corr(Zone%, production) = −0.225), so a score that tracks hitter quality picks
up a negative Zone% pull. Holding production fixed raises every score's
correlation — `signed_edge`'s from 0.184 to 0.311.

**`correct_decision` is cleaner on v1's features but not on v3's.** On v1's
features `signed_edge` is the more contaminated (raw 0.301 against 0.241;
0.398 against 0.326 holding production fixed). The hot zone closes the gap.

**Rescaling cannot fix it.** Z-score, OPS+-style ratio and percentile rank
correlate 0.9996 or more with one another and give the same contamination —
they are monotone transforms of the same per-hitter mean.

**Three attempts to remove it at the source failed:**

1. **`regret`** — one-sided, so obvious correct decisions cannot inflate it.
   Its largest losses are hittable pitches taken, so more strikes mean more
   regret: the worst contamination of any candidate (0.504 on v1's features).
2. **Departure weighting** — `(swung − P(league swings)) × edge`, so a pitch
   everyone handles the same way contributes ~0 to everyone. The mean payout is
   indeed ~0 in every region, and it is the most reliable score tested
   (split-half 0.860, YoY R² 0.658). But raw Zone% rises to 0.323, past the
   veto, and next-season validity falls to 0.098.
3. **Opportunity standardization** — a hitter's mean within region × count
   strata (with or without pitch family), reweighted to the league mix. It
   raises raw Zone% (0.184 → 0.236 / 0.217) and the production-held figure
   (0.311 → 0.359 / 0.343), and costs construct validity; only under the
   behaviour control does it move toward zero. So the contamination survives
   *within* these strata: the mix across them is not what drives it. That does
   not show pitch mix plays no role — mix at a finer grain, such as location
   within a region or pitch quality, is untested.

The contamination is **a real, unsolved limitation that the published metrics
share** — it is what retired SOTO. `signed_edge` is chosen despite it, not
because it escapes it.

---

## The feature ladder, at a fixed metric

Only the feature set varies; pipeline, learner, hyperparameters, folds and
metric are held constant. Personalized folds throughout.

Each row is a complete feature set, not an increment on the row above. The
binary hull and the continuous surface estimate the same thing — a hitter's hot
zone — so they are never combined. Hitter features go to the swing model only,
except in the last row.

| feature set | chase\|zs | zs\|chase | split-half | YoY R² | Zone% \|r\| | next-season |
|---|---|---|---|---|---|---|
| location, count | −0.920 | 0.699 | 0.818 | 0.594 | 0.300 | 0.082 |
| location, count, hull | −0.913 | 0.688 | 0.819 | 0.589 | 0.282 | 0.086 |
| batter frame, handedness, pitch chars | **−0.923** | **0.723** | 0.817 | 0.590 | 0.274 | 0.107 |
| … plus hot-zone surface | −0.895 | 0.677 | 0.838 | 0.624 | 0.184 | 0.154 |
| … same, surface in both models | −0.892 | 0.671 | **0.841** | **0.625** | **0.176** | **0.157** |

The hull contributes nothing. The pitch frame lifts next-season validity
(0.082 → 0.107) with construct validity flat or better. The hot-zone surface
supplies the rest of the gain, and all of the construct cost. Giving the surface
to the take model as well changes nothing (within 0.008 on every column).

Under `chosen_value` the same surface is far more dramatic — next-season
0.150 → **0.289** — but construct validity collapses (−0.878 / +0.657 →
−0.625 / +0.419), the signature of a score drifting from "did he decide well"
toward "is he a good hitter".

---

## The swing model: judge it on conditional means

The swing model estimates a conditional mean, not individual outcomes, and
individual outcomes are nearly irreducible — the same pitch in the same count
yields a home run or a groundout. On pitch-level RMSE it beats a count-only
lookup by only 0.80% with location and count, 0.88% with the full pitch frame.
That is the wrong instrument.

Scored instead on held-out (location × count) bin means — 12 × 12 locations in
the batter frame, bins with at least 200 swings — against a count-only
predictor fitted on the same training seasons:

| features | error | count-only error | noise floor | share of between-bin structure recovered | slope |
|---|---|---|---|---|---|
| location, count | 0.0199–0.0214 | 0.0345–0.0378 | ~0.016 | 83–85% | 1.21–1.33 |
| + batter frame | 0.0175–0.0195 | 0.0345–0.0378 | ~0.016 | 90–94% | 1.19–1.29 |

(Ranges across the three held-out seasons; generic folds.)

What this check can and cannot say:

- **It tests the location-and-count pattern, nothing finer.** Pitches are
  grouped by where they were and the count, so it asks whether the model knows
  how much better a swing is over the middle than at the edge, and how that
  changes with the count. Anything that varies *within* a cell — pitch type,
  velocity, the hitter's hot zone — is averaged away, so it cannot credit
  pitch characteristics or personalization.
- **Individual bins are imprecise.** Each observed bin mean carries ~0.016
  runs of sampling noise, most of the model's remaining error. Which specific
  bins are wrong cannot be told; the "share recovered" depends on that noise
  estimate and is for comparing variants, not a ceiling.
- **The slope is the more robust figure.** It is fitted across ~600 bins, and
  noise in the observed means widens its uncertainty without biasing it. It
  is above 1 in all three held-out seasons and at every feature set, so the
  model understates the location contrast in swing value. It has no
  confidence interval yet, and a calibration check grouped by *predicted*
  value — which would cover features that vary within a cell — has not been
  run.

**Withdrawn:** the earlier "only 1.9% of variance is learnable" and "recovers
90.6% / 94.5% of the learnable signal". Both were in-sample r² over one binning;
the 1.9% was the between-cell variance of that grid, not a ceiling on what any
feature can explain.

**The events are predictable even though the run value is not.** Whiff
probability predicts at **AUC 0.77** on held-out seasons, with velocity and
movement adding clearly over location and count in every season (AUC
0.73 → 0.77; log loss 14.5–14.9% → 17.7–18.3% better than the base rate). That
is the case for modelling whiff / foul / in-play separately rather than
regressing run value directly.

---

## The take model is a called-strike probability

A take ends three ways and the target is the league run value of
`(outcome, count)`, so given the count the only thing location can say is
whether it will be called a strike.

On held-out 2025, regressing the take model's predictions on a fitted
`P(called strike)` within each count gives **median R² 0.991**, with slopes
matching `RE(CS, c) − RE(ball, c)` to within a few thousandths — on 3-2,
−0.611 against an expected −0.614. It is learning an umpire, not a run-value
surface. The explicit structural form can be substituted for interpretability
at no cost in accuracy, and a hitter feature cannot help it: adding the hot
zone to the take model changes nothing.

---

## Personalization is real, and easy to measure wrongly

Hitters cannot cover the whole zone, and they do not cover the same part of it:

- at the **same location**, hitters differ by **1.72 mph** of expected exit
  velocity — about **0.6× the entire league-wide location effect** (87.3–90.2
  mph across the zone)
- the modal "best cell" holds only **21%** of hitter-seasons, with real mass
  across five or six

**Estimator matters enormously.** A per-hitter hot-zone surface reproduces
itself year over year at:

| estimator | YoY r |
|---|---|
| raw 4×4 bins | 0.28–0.33 |
| kernel-smoothed + empirical-Bayes shrunk | 0.63–0.68 |
| same, predicting from a two-season prior | 0.67–0.69 (one-season prior, same hitters: 0.63–0.66) |

Hitter surfaces are nearly three-dimensional — 3 principal components explain
89% of between-hitter variation, 5 explain 96% — which is why pooling recovers
so much.

The binary convex hull is a poor estimator of the same signal and contributes
nothing to the metric: 95% of the pitches it flags are in the strike zone, so it
acts as a coarse location feature.

**And the metric has to be able to see it.** The surface moves `Q_swing` by a
quarter of its own standard deviation but flips the *recommended action* on only
**1.6%** of pitches. A sign-based score therefore barely registers it
(personalized folds, batter-frame location and count, without → with the
surface):

| metric | split-half | YoY R² | next-season r | construct (chase\|zs / zs\|chase) |
|---|---|---|---|---|
| value of action taken | 0.752 → **0.901** | 0.523 → **0.699** | 0.145 → **0.286** | −0.884 / +0.671 → −0.620 / +0.418 |
| chosen minus counterfactual | 0.820 → 0.841 | 0.594 → 0.629 | 0.103 → 0.157 | −0.925 / +0.720 → −0.895 / +0.670 |
| correct decision (±1) | 0.827 → 0.828 | 0.580 → 0.583 | 0.137 → 0.150 | −0.965 / +0.933 → −0.964 / +0.931 |

Under the first, construct validity collapses — the feature pulls the score
toward measuring hitting ability, which is also why it predicts production so
much better. Under the second, reliability, contamination (Zone% 0.279 → 0.185)
and usefulness all improve, for a smaller construct cost.

**A feature and a metric cannot be evaluated independently.**

---

## The data

**2026 is a different measurement regime.** Statcast moved `plate_x`/`plate_z`
from front-of-plate to middle-of-plate and switched `sz_top`/`sz_bot` to the ABS
zone. The location shift is ~1 inch vertically and depends on pitch type (0.7 in
for a four-seamer, 1.5 in for a curveball), so it must be converted from the
pitch trajectory rather than offset by a constant.

**The ABS zone is exactly 27%–53.5% of batter height.** In 2026 `sz_bot/sz_top`
is 0.5047 for every batter with zero spread (27.0/53.5 = 0.5047). The common
zone used throughout applies that band to MLB's listed height, available for
every batter in every season. Listed height is rounded to the inch while ABS
measures it more finely, so the common zone is off by up to ~0.27 in at the top
— far less than the 0.073–0.098 ft within-batter noise of the operator-set
bounds it replaces.

**ABS judges "any part of the ball", not its centre** — the same convention as
the rulebook. The empirical 50% called-strike boundary sits one ball radius
outside the nominal zone. The 2026 change is the nominal zone
(2.64 in lower at the top), not the convention.

**The called zone tightened under ABS.** On the common zone the 50% boundary
went 0.190 → 0.121 ft and the effective called area shrank 8.1%, with 2026
landing on the ball radius — the called boundary converging on the true one.
The lefty strike shrank by only ~20%, as expected when a few pitches per game
are challenged. Pitchers also threw more strikes: Zone% rose from 0.441 (2021)
to 0.476 (2026) on the common zone.

**Run values do not drift** — at most 0.014 runs across six seasons, so one
table fitted on each fold's training years suffices.

**`delta_run_exp` is a deterministic base–out–count lookup** (within-group SD
0.00000000). Grouping by `(outcome, count)` alone averages over base–out states,
whose means span about a quarter of a run — the price of a context-neutral
target.

**`field_error` is worth +0.461 runs against −0.250 for `field_out`**, close to
a single. Folding them together mis-prices 6,524 events by 0.71 runs each.

**Counterfactual support is sufficient.** The thinnest cells are 3-0 off the
plate (~22 league swings per season) but those are also the least ambiguous, so
error there cannot flip a decision. Support correlates **+0.65** with ambiguity —
the close calls have the most data.

**Exclusions.** `automatic_ball`/`automatic_strike` are non-decisions (pitch
clock and intentional walks, ~13.9k rows). 2021 carries 402 pitcher-batters
against 16–30 later. Overseas games are dropped; Toronto is kept.

---

## Method notes worth keeping

- **Select on held-out seasons, and keep the final test untouched.** Every
  decision here was made on the 2023–2025 folds; 2026 is scored once, at the
  end, by a model trained through 2024 and one trained through 2025.
- **One zone definition for every season.** Using each season's own
  `sz_top`/`sz_bot` changes the meaning of "in the zone" at the 2025/2026
  boundary; so does applying the ball radius on some edges and not others.
- **Judge a feature by the best available estimator, not the naive one.** The
  hot zone goes from r ≈ 0.30 to ≈ 0.68 with no new data.
- **A poor estimator of a signal does not necessarily harm the metric**, and a
  good one does not necessarily help it. Measure the metric.
- **Prior-window features drift when the window grows.** With every prior
  season, the hull flag fires on 14.5% of pitches in 2022 and 25.1% in 2025.
  A fixed two-season window holds it at 20–21%.
- **A check grouped by location cannot credit a feature that varies within a
  location.** Pitch characteristics and the hot zone both look inert on the
  bin-mean check while mattering elsewhere.
- **Judge counterfactual support by ambiguity, not raw counts.** Thin cells
  coincide with obvious decisions, where a large gap makes the call robust.
- **Report construct validity as partial correlations.** Chase rate and
  zone-swing rate correlate +0.5 through aggression, so a raw correlation
  against either is confounded and reads as though the metric ignores half the
  construct.
- **Never report RMSE without its floor.** The target is a function of
  `(outcome, count)`, so a count-only lookup is the baseline; the level alone
  mostly reflects which action is being scored.
- Hitter features for season *t* must come from seasons before *t*.
- Realized bat speed or exit velocity on the swing being graded is never a
  feature — that is execution, not decision.
- Contact quality cannot change an umpire's call, so hitter features belong on
  the swing side only. Verified: adding the hull to the take model moved its
  held-out RMSE from 0.04361 to 0.04368, and adding the hot zone changed
  nothing.
