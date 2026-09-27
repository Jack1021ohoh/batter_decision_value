# Findings

What the data established, separate from what we planned to do about it. Every
number here was measured in one of the notebooks and can be reproduced by
running it.

**Every model result is held out.** Each variant is fitted on rolling-origin
folds and each of 2023, 2024 and 2025 is scored by a model that never saw it;
year-over-year and next-season checks use the 2023→24 and 2024→25 pairs.
Generic variants train from 2021; whenever a personalized variant is in a
comparison, every row trains from 2022 and uses 2021 only as prior data. Every
version uses one learner: XGBoost, depth 7, learning rate 0.05, rounds set by
early stopping on 10% of training games.

**2026 is not an untouched test.** It is excluded from everything here, but
earlier versions of the project scored it: they printed a 2026 leaderboard and
averaged the 2025→26 pairs into the headline means that informed the choice of
metric and features. Re-making those decisions on the folds changes the
evidence, not that history. 2026 will be reported as a previously inspected,
out-of-regime evaluation (the first ABS season); the confirmatory test is
reserved for 2027, a season no version has seen.

---

## How models are chosen

The player-metric checks — construct partials, split-half, YoY, Zone% and
next-season — show whether a score is stable and plausible. They cannot show a
model estimates well: a score that tracked batting average would pass every
reliability check, and a feature can raise next-season validity simply by
importing hitting ability. So models are chosen by a rule fixed before the
re-runs:

1. **Accuracy decides.** Held-out squared error of two variants on the same
   pitches, with a 95% interval from resampling games. A swing's error is
   mostly outcome luck, but the luck adds the same amount to both variants, so
   the paired difference is the difference in how far each is from the true
   expected run value. A variant wins if it is better on at least one action
   and worse on neither; an interval spanning 0 is a tie, and the simpler model
   stays.
2. **Calibration checks the winner** — observed against predicted, grouped by
   predicted value, with an interval — because the decision score uses the size
   of each value, not only its sign.
3. **Guardrails.** The player checks, with hitter-resampled intervals on each
   difference, flag a change for discussion and never choose a model.

The metric choice is the exception: there the model is held fixed, so accuracy
cannot differ and the player checks are the only evidence.

---

## Results

The score is reported two ways, each chosen by accuracy (`v4_decomposition.ipynb`):

| output | question | chase\|zs | zs\|chase | split-half | YoY R² | Zone% \|r\| | next-season |
|---|---|---|---|---|---|---|---|
| **generic** | a good decision for a typical hitter? | −0.914 | 0.706 | 0.816 | 0.583 | 0.270 | 0.100 |
| **personalized** | a good decision for *this* hitter? | −0.862 | 0.582 | 0.847 | 0.615 | 0.184 | 0.173 |

Both value a swing by decomposing it — `P(whiff)·RE(whiff) + P(foul)·RE(foul) +
P(in play)·E[value | in play]` — on the full pitch frame, and score it with
`signed_edge`. The personalized model adds the hitter's hot zone to the in-play
value and his whiff and foul tendencies to the outcome classifier. It is the
most accurate model found (swing MSE 2.09% below a count-only predictor, against
1.95% generic); its score is more reliable, less contaminated and more
predictive, and tracks plate discipline less — by design. The two correlate
0.939 across hitter-seasons.

**v1 and v2 as designed** (value of the action taken, v2's folds):

| | chase\|zs | zs\|chase | split-half | YoY R² | Zone% \|r\| | next-season |
|---|---|---|---|---|---|---|
| v1 | −0.890 | 0.678 | 0.763 | 0.538 | 0.116 | 0.129 |
| v2 | −0.803 | 0.548 | 0.794 | 0.540 | 0.058 | 0.129 |

v2's hull does not make a better model: swings 0.018% better, takes 0.27% worse
(it feeds a contact feature to a model of an umpire's call).

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
| `chosen_value` | −0.889 | 0.681 | 0.765 | 0.537 | 0.124 | 0.126 |
| `signed_edge` | −0.920 | 0.700 | 0.819 | 0.596 | 0.299 | 0.085 |
| `regret` | −0.831 | 0.530 | 0.795 | 0.581 | 0.504 | 0.028 |
| `close_weighted` | −0.701 | 0.802 | 0.606 | 0.317 | 0.045 | 0.101 |
| `correct_decision` | **−0.953** | **0.906** | **0.826** | 0.586 | 0.250 | 0.115 |

**By the criterion fixed before the numbers — construct validity first —
`correct_decision` wins**, and still does with the hot zone added (−0.957 /
+0.911). As `close_weighted`'s kernel widens it converges on `correct_decision`
(correlation 0.997 at scale 0.8). The run-value magnitude carries the pitch-mix
bias: `signed_edge` pays about eight times more per pitch for an obvious take
(+0.087 runs) than for getting a genuinely close call right (+0.011).

**`signed_edge` is selected anyway, as a trade.** It keeps the run-value
magnitude every published metric uses — SwRV, SOTO, Nestico's Decision Value,
Creally's wDV, EAGLE all report run value per 100 pitches. A sign-based score
treats a razor-thin call and an obvious blunder alike and barely registers a
feature that changes how much a swing is worth without flipping the decision.

### The contamination problem

Magnitude-weighted scores can be contaminated by pitch mix. Zone% against the
score, with something held fixed (`v3.ipynb`; the v3 model is the direct
regression on the full pitch frame with the hot zone):

| model | score | raw | \| correct-decision rate | \| chase, zone-swing | \| production |
|---|---|---|---|---|---|
| v1 features | `signed_edge` | 0.299 | 0.172 | −0.006 | 0.396 |
| v1 features | `correct_decision` | 0.250 | *(circular)* | 0.160 | 0.337 |
| v3 | `chosen_value` | −0.165 | −0.421 | −0.458 | −0.037 |
| v3 | `signed_edge` | 0.172 | −0.072 | −0.252 | 0.301 |
| v3 | `regret` | 0.555 | 0.599 | 0.533 | 0.591 |
| v3 | `correct_decision` | 0.230 | *(circular)* | 0.039 | 0.328 |

**The answer depends on the control, so no control settles it.** Each is a
screen: the correct-decision rate comes from the same model's Δ (and *is*
`correct_decision`), and chase and zone-swing rates vary with how hard the
pitches a hitter sees are.

**The raw Zone% test is lenient.** Better hitters are thrown fewer strikes
(corr(Zone%, production) = −0.225), so holding production fixed raises every
score's correlation — `signed_edge`'s from 0.172 to 0.301.

**`correct_decision` is cleaner on v1's features but not on v3's**; the hot zone
closes the gap.

**Rescaling cannot fix it** — z-score, OPS+-style ratio and percentile rank
correlate 0.9997 or more.

**Three attempts to remove it at the source failed:**

1. **`regret`** — its largest losses are hittable pitches taken, so more strikes
   mean more regret: the worst contamination of any candidate (0.504).
2. **Departure weighting** — `(swung − P(league swings)) × edge`. The mean
   payout is ~0 in every region and it is the most reliable score tested
   (split-half 0.864, YoY 0.652), but raw Zone% rises to 0.327 and next-season
   validity falls to 0.100.
3. **Opportunity standardization** — a hitter's mean within region × count
   strata (± pitch family), reweighted to the league mix. It raises raw Zone%
   (0.172 → 0.224 / 0.204) and the production-held figure (0.301 → 0.350 /
   0.333) and costs construct validity; only the behaviour-held figure moves
   toward zero. The mix *across* these strata is not what drives the
   contamination; mix at a finer grain is untested.

The contamination is **a real, unsolved limitation that the published metrics
share** — it is what retired SOTO.

---

## The feature ladder

Personalized folds, `signed_edge`, each rung a complete feature set (the hull
and the surface estimate the same thing, so they are never combined). Paired
accuracy is each rung against the one it replaces; negative = better.

| rung | swing MSE (95% interval) | take MSE | chase\|zs | zs\|chase | YoY R² | Zone% \|r\| | next-season |
|---|---|---|---|---|---|---|---|
| location, count (v1) | — | — | −0.921 | 0.703 | 0.594 | 0.298 | 0.084 |
| + hull (v2), vs v1 | −0.023% (−0.037, −0.012) | tie | −0.915 | 0.692 | 0.589 | 0.282 | 0.087 |
| full pitch frame, vs v1 | −0.264% (−0.294, −0.237) | −8.7% | −0.916 | 0.711 | 0.584 | 0.269 | 0.104 |
| + hot-zone surface, vs frame | −0.062% (−0.090, −0.039) | tie | −0.885 | 0.662 | 0.629 | 0.169 | 0.167 |
| surface in both models, vs swing-only | tie | +0.35% | −0.882 | 0.657 | 0.630 | 0.163 | 0.168 |

**Every pitch-frame group makes a better model** (generic folds): the batter
frame cuts take MSE 7.7% and swing MSE 0.12%, handedness takes 0.7%, pitch
characteristics takes 0.5% and swings 0.15%. A location-binned check had
suggested pitch characteristics added nothing to the swing model — it could not
see them, because it averages over everything within a location cell.

**The hot zone belongs in the swing model only.** It improves the swing model
three times as much as the hull; in the take model it makes that model worse.

**The guardrails move with the surface**: YoY, Zone% and next-season improve,
while construct validity slips (−0.916 / +0.711 → −0.885 / +0.662), all with
intervals excluding zero. A better estimate of what a swing is worth to this
hitter partly moves the score toward "is he a good hitter". Under
`chosen_value` the same surface collapses construct validity (−0.602 / +0.403).

---

## The swing and take models

**Both beat a count-only predictor decisively** (v1's features, generic
folds): take MSE 67% lower, swing MSE 1.6% lower, every interval far from zero.
The swing figure is small because most of a swing's squared error is outcome
luck, not because the model learns little; paired comparisons cancel that luck.

**v1's fixed settings under-trained the swing model.** With learning rate 0.01
and 200 rounds its predictions were compressed — calibration slope, grouped by
predicted value, 1.18–1.26 with every interval above 1. Early stopping brings it
to 0.99–1.09 (`v1_baseline.ipynb`). Hyperparameters are pipeline, not design, so
every version uses the early-stopping learner.

**Calibration is close but not exact, and two repairs failed.** Across the
final models, take slopes are 0.99–1.00 and swing slopes 0.94–1.09, furthest
from 1 in 2023, whose fold trains on one season. A linear map fitted on
early-stopping games left swing MSE unchanged and swing calibration worse; one
fitted on the previous held-out season over-corrected and made MSE worse. The
residual varies from fold to fold, so neither map could predict it; the models
are used as fitted.

**Where `Q_swing` is extrapolated it is slightly too generous.** It can only be
checked on pitches someone swung at. On swings at pitches the league swings at
less than 10% of the time, the final models predict about −0.060 runs against
−0.067 observed, so the most obvious chases are penalised slightly less than
they should be.

**The take model is a called-strike probability.** On held-out 2025, regressing
its predictions on a fitted `P(called strike)` within each count gives median R²
0.991, with slopes matching `RE(CS, c) − RE(ball, c)` — on 3-2, −0.612 against
an expected −0.614.

**The events are predictable even though the run value is not.** Whiff
probability predicts at AUC 0.77 held out, with velocity and movement adding
clearly over location and count (0.73 → 0.77; log loss 14.6–14.9% → 17.7–18.3%
better than the base rate).

---

## Personalization is real, and easy to measure wrongly

Hitters cannot cover the whole zone, and they do not cover the same part of it:
at the **same location** they differ by **1.72 mph** of expected exit velocity —
about **0.6× the entire league-wide location effect** (87.3–90.2 mph) — and the
modal "best cell" holds only **21%** of hitter-seasons.

**Estimator matters enormously** (EDA §6):

| estimator | YoY r |
|---|---|
| raw 4×4 bins | 0.28–0.33 |
| kernel-smoothed + empirical-Bayes shrunk | 0.63–0.68 |
| same, predicting from a two-season prior | 0.67–0.69 (one-season prior, same hitters: 0.63–0.66) |

Hitter surfaces are nearly three-dimensional — 3 principal components explain
89% of between-hitter variation — which is why pooling recovers so much.

**And the metric has to be able to see it.** The surface moves `Q_swing` by a
quarter of its own standard deviation but flips the recommended action on only
**2.0%** of pitches. With the model held fixed (batter-frame location and
count, without → with the surface):

| metric | split-half | YoY R² | next-season | construct (chase\|zs / zs\|chase) |
|---|---|---|---|---|
| value of action taken | 0.759 → **0.913** | 0.529 → **0.713** | 0.139 → **0.292** | −0.890 / +0.678 → −0.601 / +0.404 |
| chosen minus counterfactual | 0.822 → 0.846 | 0.592 → 0.635 | 0.103 → 0.165 | −0.925 / +0.720 → −0.888 / +0.657 |
| correct decision (±1) | 0.833 → 0.831 | 0.585 → 0.580 | 0.135 → 0.155 | −0.964 / +0.926 → −0.957 / +0.911 |

**A feature and a metric cannot be evaluated independently.**

---

## Track B: decomposing the swing

`v4_decomposition.ipynb`. The decomposed model is compared with a **direct
twin** sharing everything else — score, pitch features, hitter priors, folds,
learner and the take model itself (asserted identical).

| comparison | swing MSE, paired (95% interval) |
|---|---|
| decomposed vs direct, generic | −0.085% (−0.103, −0.071) — better in every season |
| decomposed vs direct, + hot zone | −0.109% (−0.129, −0.091) |
| + contact priors, direct twin | −0.073% (−0.088, −0.056) |
| + contact priors, decomposed twin | −0.050% (−0.065, −0.035) |

**The decomposition is the better estimator**, generic and personalized, and
more stable (Δ SD across refits 0.0070 against 0.0086 runs). It barely changes
the score: the twins' hitter-season values correlate 0.998. An earlier version
called it a tie, because it compared the twins on the player checks alone, which
cannot see an accuracy difference that leaves the score unchanged.

**Contact priors are more accurate and move the score.** A hitter's whiff-rate
surface reproduces year to year (r ≈ 0.66–0.67; foul rate 0.43–0.46) and
improves both twins. In the decomposed twin it shifts the score toward contact
skill: zone-swing | chase 0.648 → 0.582, YoY 0.632 → 0.615, Zone% 0.162 → 0.184.
That is why the score has two outputs: the priors belong in the answer to "was
this good for *this* hitter", not in "was this good for a typical hitter".

**Benchmarks.** O-Swing% is heavily contaminated (Zone% |r| 0.421) and weakly
predictive (0.077). Z-Swing% − O-Swing% is less contaminated than either output
(0.114) and nearly as predictive as the personalized one (0.146 against 0.173),
though less reliable (YoY 0.557).

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

- **Choose models by paired held-out accuracy; use the player checks as
  guardrails.** Reliability and prediction checks cannot tell a good model from
  a stable wrong one, and pitch-level RMSE looks flat for swings only because
  of outcome luck, which a paired comparison cancels.
- **Check the learner before blaming the data.** A swing model that seemed
  unable to learn the size of location effects had simply stopped too early.
- **A check grouped by location cannot credit a feature that varies within a
  location.** It hid the value of pitch characteristics.
- **A better sub-model need not change the metric, and vice versa.** The
  decomposition improved accuracy while leaving the score unchanged; contact
  priors improved accuracy while moving the score. Measure both.
- **Select on held-out seasons, and keep the test untouched — from the first
  version on.** 2026 was inspected by earlier versions, and removing it from the
  code does not undo that.
- **One zone definition for every season.** Using each season's own
  `sz_top`/`sz_bot` changes the meaning of "in the zone" at the 2025/2026
  boundary; so does applying the ball radius on some edges and not others.
- **Judge a feature by the best available estimator, not the naive one.** The
  hot zone goes from r ≈ 0.30 to ≈ 0.68 with no new data.
- **Prior-window features drift when the window grows.** With every prior
  season, the hull flag fires on 14.5% of pitches in 2022 and 25.1% in 2025; a
  fixed two-season window holds it at 20–21%.
- **Judge counterfactual support by ambiguity, not raw counts.** Thin cells
  coincide with obvious decisions, where a large gap makes the call robust.
- **Report construct validity as partial correlations.** Chase rate and
  zone-swing rate correlate +0.5 through aggression, so a raw correlation
  against either is confounded.
- Hitter features for season *t* must come from seasons before *t*.
- Realized bat speed or exit velocity on the swing being graded is never a
  feature — that is execution, not decision.
- Contact quality cannot change an umpire's call: hitter features in the take
  model made it less accurate (the hull by 0.27%, the hot zone by 0.35%).
