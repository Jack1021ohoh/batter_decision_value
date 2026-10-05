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
2027, a season no version has seen. That does not keep 2026 out of training:
once 2027 is complete, 2026 becomes ordinary data — a training season, and a
held-out fold (train through 2025, score 2026) for model selection, the only
one in the ABS regime. The final model is fitted through 2026 and scores 2027
once, after every decision is locked.

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

The score is reported two ways (`v5_in_play.ipynb`, `v6_shrink.ipynb`):

| output | question | chase\|zs | zs\|chase | split-half | YoY R² | Zone% \|r\| | next-season |
|---|---|---|---|---|---|---|---|
| **generic** | a good decision for a typical hitter? | −0.910 | 0.689 | 0.816 | 0.583 | 0.276 | 0.094 |
| **personalized** | a good decision for *this* hitter? | −0.888 | 0.627 | 0.833 | 0.598 | 0.224 | 0.144 |

Both value a swing by decomposing it — `P(whiff)·RE(whiff) + P(foul)·RE(foul) +
P(in play)·E[value | in play]` — and a ball in play by the odds of each event,
`Σ P(event)·RE(event, count)`, on the full pitch frame, scored with
`signed_edge`. The personalized model adds the hitter's hot zone to the
in-play-event model and his whiff and foul tendencies to the outcome classifier,
and keeps about two thirds of what they add to the swing value (k = 0.68; see
"Shrinking personalization"). It is the most accurate model found (swing MSE
2.16% below a count-only predictor, against 1.96% generic); its score is more
reliable, less contaminated and more predictive, and tracks plate discipline a
little less — by design. The two correlate 0.972 across hitter-seasons. Both are refitted on every
development season and saved in `final_models.ipynb` (see "Final models and
2026").

**Each version**, held out on the same personalized folds and learner. v1 and
v2 score the value of the action taken, as designed; v3–v6 score
`signed_edge`. The model columns compare across every row; the player checks
compare within a metric.

| version | swing MSE vs count-only | swing calibration slope | take MSE vs count-only | chase\|zs | zs\|chase | split-half | YoY R² | Zone% \|r\| | next-season |
|---|---|---|---|---|---|---|---|---|---|
| v1 — location, count | −1.61% | 1.067 | −67.2% | −0.890 | 0.678 | 0.763 | 0.538 | 0.116 | 0.129 |
| v2 — + binary hull (both models) | −1.63% | 1.056 | −67.1% | −0.803 | 0.548 | 0.794 | 0.540 | 0.058 | 0.129 |
| v3 — full pitch frame + hot zone | −1.93% | 1.042 | −70.1% | −0.887 | 0.660 | 0.844 | 0.630 | 0.172 | 0.166 |
| v4 generic | −1.95% | 1.044 | −70.1% | −0.914 | 0.706 | 0.816 | 0.583 | 0.270 | 0.100 |
| v4 personalized | −2.09% | 0.960 | −70.1% | −0.862 | 0.582 | 0.847 | 0.615 | 0.184 | 0.173 |
| v5 generic — + in-play events | −1.96% | 1.017 | −70.1% | −0.910 | 0.689 | 0.816 | 0.583 | 0.276 | 0.094 |
| v5 personalized — + in-play events | −2.12% | 0.956 | −70.1% | −0.864 | 0.576 | 0.844 | 0.606 | 0.198 | 0.159 |
| v6 personalized — shrunk toward generic | −2.16% | 1.007 | −70.1% | −0.888 | 0.627 | 0.833 | 0.598 | 0.224 | 0.144 |

- **v1 → v2: not a better model.** The hull improves the swing model by 0.018%
  of MSE and worsens the take model by 0.27%, since v2 feeds a contact feature
  to a model of an umpire's call. Its score gains reliability and loses
  construct validity (zone-swing | chase 0.678 → 0.548).
- **v1 → v3: better on every model column.** The pitch frame and the hot zone
  together; the step-by-step accuracy is in "The feature ladder". Under
  `signed_edge` the score becomes more reliable, less contaminated and more
  predictive, at a construct cost that comes from the hot zone. v3's row is its
  selected model as that notebook reports it — recalibrated on early-stopping
  games, a step later found not to help; as fitted, its player checks differ by
  at most 0.003.
- **v3 → v4: a better swing model.** Decomposing the swing, with v3's features,
  cuts swing MSE 0.109%; contact priors a further 0.050% (the personalized
  output). The generic output drops hitter features and is about as accurate as
  v3, with the strongest construct validity of any `signed_edge` row.
- **v4 → v5: a ball in play as events.** More accurate for the personalized
  output (−0.038%), a tie for the generic one; both adopt it (see "Valuing a
  ball in play as events").
- **v5 → v6: personalization at two thirds strength.** The personalized swing
  value shrunk toward the generic one is more accurate in every held-out season
  (−0.042%) and calibrated (see "Shrinking personalization"). The generic
  output is unchanged.

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
score, with something held fixed (`v3.ipynb`; "v3" is that notebook's selected
model — the direct regression on the full pitch frame with the hot zone, as
recalibrated there). The two outputs are measured the same way in
`v5_in_play.ipynb` and `v6_shrink.ipynb`, below.

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

**On the outputs** (`v5_in_play.ipynb`; the shrunk row from `v6_shrink.ipynb`):

| output | score | raw | \| correct-decision rate | \| chase, zone-swing | \| production |
|---|---|---|---|---|---|
| generic | `signed_edge` | 0.276 | 0.129 | −0.044 | 0.379 |
| generic | `correct_decision` | 0.245 | *(circular)* | 0.110 | 0.339 |
| personalized, unshrunk | `signed_edge` | 0.198 | −0.047 | −0.193 | 0.331 |
| personalized, unshrunk | `correct_decision` | 0.252 | *(circular)* | 0.067 | 0.355 |
| **personalized (in use, shrunk)** | `signed_edge` | 0.224 | 0.009 | −0.162 | 0.350 |

The same pattern: **without hitter features `signed_edge` is the more
contaminated score** — the generic output's is the highest of any score here,
0.379 with production held fixed — **and with them it is the less
contaminated one** (unshrunk personalized 0.198 raw and 0.331 against 0.252 and
0.355 for `correct_decision`). The shrunk output in use sits between the two it
blends (0.224 and 0.350). A hitter's opportunities feed the magnitude of his
edges unless the model knows what that hitter does with them.

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

Personalized folds, `signed_edge`, direct regression as fitted, each rung a
complete feature set (the hull and the surface estimate the same thing, so they
are never combined). Paired accuracy is each rung against the one it replaces;
negative = better.

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

**The hot zone belongs in the swing model only.** On the same base model
(batter-frame location and count) it improves the swing model three times as
much as the hull — 0.059% of MSE against 0.018% — and beats the hull head to
head. In the take model it makes that model worse (by 0.35%).

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

**Calibration: two repairs failed, the third worked.** A linear map fitted on
early-stopping games left swing MSE unchanged and swing calibration worse in two
of three seasons; one fitted on the previous held-out season over-corrected and
made MSE worse — the residual moved from fold to fold, so neither map could
predict it. What did not move was the personalized model's over-spread (slopes
0.935–0.980 held out, 0.948 on 2026), and shrinking toward the generic model
removed it (0.981–1.039 held out, 0.996 on 2026). The generic output is close
(0.98–1.06, slightly compressed in 2023, whose fold trains on one season); take
slopes are 0.99–1.00.

**Where `Q_swing` is extrapolated it is slightly too generous.** It can only be
checked on pitches someone swung at. On swings at pitches the league swings at
less than 10% of the time, both outputs predict about −0.062 runs against
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

A binary flag over the same signal — v2's convex hull — is a poor estimator of
it: 95% of the pitches it marks are in the strike zone, so it acts as a coarse
location feature rather than as personalization, and improves the model a third
as much as the continuous surface.

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
more stable: refit on resampled games, its Δ varies less (SD 0.0070 against
0.0086 runs for the generic twins, 0.0073 against 0.0087 with the hot zone). It
barely changes the score: the generic twins' hitter-season values correlate
0.998. An earlier version
called it a tie, because it compared the twins on the player checks alone, which
cannot see an accuracy difference that leaves the score unchanged.

**Contact priors are more accurate and move the score.** A hitter's whiff-rate
surface reproduces year to year (r ≈ 0.66–0.67; foul rate 0.43–0.46) and
improves both twins. In the decomposed twin it shifts the score toward contact
skill: zone-swing | chase 0.648 → 0.582, YoY 0.632 → 0.615, Zone% 0.162 → 0.184.
That is why the score has two outputs: the priors belong in the answer to "was
this good for *this* hitter", not in "was this good for a typical hitter".

### Valuing a ball in play as events (`v5_in_play.ipynb`)

The in-play value was first a regression of run value on balls in play. v5
predicts the odds of each event instead — single, double, triple, home run,
out, error — and weights each by its run value in the count, keeping the count
as a feature. Only that branch changes: the take model and the whiff / foul /
in-play classifier are asserted identical between the two.

| output | swing MSE, paired (95% interval) | balls in play only |
|---|---|---|
| personalized | −0.038% (−0.056, −0.018) | −0.078% (−0.122, −0.032) |
| generic | −0.012% (−0.025, +0.002) — a tie | −0.014% (−0.048, +0.021) |

**Better for the personalized output, a tie for the generic one.** The
personalized gain rests mainly on 2023 (−0.090%), whose fold trains on one
season; 2024 and 2025 tie. The guardrails move slightly the wrong way
(personalized YoY 0.615 → 0.606, Zone% 0.184 → 0.198, next-season 0.173 →
0.159), and the personalized swing model is slightly more over-spread in
2024–25. **Both outputs adopt it**: the personalized one by the rule, the
generic one by decision — on a tie, so that the two outputs differ only in
their hitter features. That overrides the rule's tie-keeps-current default and
is recorded as such.

**What it learns.** Log loss improves 1.6–1.8% over the base rates (generic),
1.7–2.1% (personalized). Triples and errors stay near their base rates, as
expected. Personalization moves the event odds as it should: on pitches in a
hitter's hottest zones it raises P(home run) by 1.7 percentage points and
P(double) by 0.7, and lowers P(out) by 2.2.

### Shrinking personalization (`v6_shrink.ipynb`)

The personalized swing model was steadily over-spread: its differences between
pitches about 5% larger than what happened. v6 pulls its swing value part-way
back toward the generic model, shrinking only what personalization adds:

```
q_swing = q_generic + k · (q_personalized − q_generic)
```

k has a closed-form least-squares solution on held-out swings; each held-out
season uses a k fitted on the other two. It is stable — 0.73, 0.66 and 0.67 —
and 0.685 from all three, which the final model uses.

**Adopted by the rule.** The shrunk model is more accurate in every held-out
season (−0.067%, −0.028%, −0.030%; pooled −0.042%, interval −0.050% to −0.032%)
and on 2026 (−0.060%), and calibrated. The hitter priors point the right way but
were believed about 1.5 times too strongly — two seasons of a hitter's contact
are partly luck, and he regresses toward the typical hitter. Construct validity
improves (−0.864 / +0.576 → −0.888 / +0.627); YoY (0.606 → 0.598), Zone%
contamination (0.198 → 0.224) and next-season validity (0.159 → 0.144) give a
little back.

**Benchmarks.** O-Swing% is heavily contaminated (Zone% |r| 0.421) and weakly
predictive (0.077). Z-Swing% − O-Swing% is less contaminated than either output
(0.114) and as predictive as the personalized one (0.146 against 0.144), though
less reliable (YoY 0.557 against 0.598).

---

## Final models and 2026

`final_models.ipynb` refits both outputs once on every development season —
generic on 2021–2025, personalized on 2022–2025 (2026 priors from 2024–25) —
saves them to `models/`, and scores 2026 from the reloaded files. 2026 was
inspected by earlier versions, so this is an **out-of-regime evaluation**, not a
clean test; nothing in the design changes because of it.

| | swing MSE vs count-only | take MSE vs count-only | swing calibration slope (95% interval) |
|---|---|---|---|
| generic | −1.98% | −74.3% | 1.003 (0.979, 1.023) |
| personalized (shrunk) | −2.18% | −74.5% | 0.996 (0.973, 1.017) |

**The models carry over to the first ABS season.** Swing accuracy matches the
held-out 2023–25 seasons; take accuracy is higher (74% against ~70%), as
expected when calls follow a fixed zone. Both are calibrated; without the
shrink the personalized slope would be 0.948. The score's split-half
reliability is 0.800 (generic) and 0.815 (personalized); its construct partials
are lower than held out (−0.893 / +0.659 and −0.855 / +0.558), and the two
outputs' hitter scores correlate 0.963.

---

## The data (EDA)

Full detail in `notebooks/eda.ipynb`, whose seven sections each end in a
decision; these are the results that shape the models.

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
bounds it replaces. Without one zone, cross-era comparisons are confounded: the
ABS zone is ~2.9 in shorter than the operator-set zone it replaced.

**ABS judges "any part of the ball", not its centre** — the same convention as
the rulebook. The empirical 50% called-strike boundary sits one ball radius
outside the nominal zone. The 2026 change is the nominal zone
(2.64 in lower at the top), not the convention.

**The called zone tightened under ABS.** On the common zone the 50% boundary
went 0.190 → 0.121 ft and the effective called area shrank 8.2%, with 2026
landing on the ball radius — the called boundary converging on the true one.
The lefty strike shrank by only ~20%, as expected when a few pitches per game
are challenged. Pitchers also threw more strikes: Zone% rose from 0.441 (2021)
to 0.476 (2026) on the common zone.

**Run values barely drift.** Across all 132 (outcome, count) cells a cell's
value moves by 0.007 runs across six seasons on a pitch-weighted average, and by
at most 0.021 among the cells covering 95% of pitches. The large swings are in
rare cells (a triple or home run on 3-0), where a season holds a handful of
events — sampling noise that a per-season table would feed into the targets.
So one table fitted on each fold's training years is used for training and
held-out seasons alike: the held-out season is then graded against the same
definition of value the model learned, rather than one built from its own
outcomes.

**`delta_run_exp` is a deterministic base–out–count lookup** (within-group SD
0.00000000). Grouping by `(outcome, count)` alone averages over base–out states,
whose means span about a quarter of a run — the price of a context-neutral
target.

**`field_error` is worth +0.460 runs against −0.250 for `field_out`**, close to
a single. Folding them together mis-prices 6,597 events by 0.71 runs each.

**Counterfactual support is sufficient.** The thinnest cells are 3-0 off the
plate (~22 league swings per season) but those are also the least ambiguous, so
error there cannot flip a decision. Support correlates **+0.65** with ambiguity —
the close calls have the most data.

**Exclusions.** `automatic_ball`/`automatic_strike` are non-decisions (pitch
clock and intentional walks, ~14.0k rows). 2021 carries 402 pitcher-batters
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
- **Shrink hitter-specific adjustments, not just hitter estimates.** The hot
  zone was already shrunk toward the league, yet the model built on it still
  overstated hitters' differences; shrinking the personalized prediction toward
  the generic one fixed accuracy and calibration together.
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
