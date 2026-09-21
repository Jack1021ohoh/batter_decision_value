# Findings

What the data established, separate from what we planned to do about it. Every
number here was measured in one of the notebooks and can be reproduced by
running it.

---

## Results

Same pipeline, learner and hyperparameters throughout; only the design differs,
so each row's difference is attributable. Higher is better except Zone%, where
lower means less contaminated by the pitches a hitter happened to be thrown.

| | swing RMSE vs count-only | split-half r | YoY R² | Zone% \|r\| | next-season partial r |
|---|---|---|---|---|---|
| v1 (2021–26) | 0.2985 / 0.3010 | 0.750 | 0.519 | 0.077 | 0.092 |
| v1 (2022–26 window) | 0.2974 / 0.2998 | 0.753 | 0.517 | 0.077 | 0.095 |
| v2 (2022–26) | 0.2973 / 0.2998 | 0.786 | 0.530 | 0.033 | 0.105 |
| v3 (the patch) | 0.2982 / 0.3010 | 0.806 | 0.560 | 0.207 | 0.085 |

**These columns understate what changed.** The check that discriminates between
designs is construct validity — does the metric punish chasing *and* reward
attacking hittable pitches, each measured with the other held fixed. There v3
reaches **−0.924 / +0.871** against v1's **−0.849 / +0.594**.

---

## The metric is most of the model

Swapping only how per-pitch scores aggregate, with features and models
identical, moves construct validity from −0.849/+0.594 to −0.905/+0.831.
Nothing else in the project moves anything comparable.

**The winning score is the simplest one available**: +1 when the hitter picked
the better action, −1 when he did not, with no run-value weighting. Every
magnitude-weighted alternative is more contaminated by pitch mix, because the
magnitude is exactly what varies with the pitches a hitter sees.

| score | contamination (Zone% \|r\|) | why |
|---|---|---|
| value of the action taken | 0.077 | taken balls and swung strikes both pay, so they largely cancel |
| chosen minus counterfactual | 0.276 | pays 5× more for an obvious take than a close call, so it rewards being thrown junk |
| regret, `max(0, −signed)` | 0.468 | its largest error is taking a hittable pitch, so more strikes means more regret |
| **correct decision (±1)** | 0.230 | no magnitude, so none of the above |

A closeness-kernel sweep confirms the mechanism: as the kernel widens, the
weighted score converges on the unweighted one (correlation 0.997) and measures
better the whole way.

**Caveat, and it matters:** discarding magnitude is also what makes this score
unable to benefit from an informative feature. See personalization below.

---

## Pitch-level RMSE is the wrong instrument for the swing model

The swing model estimates a conditional mean, not individual outcomes, and
individual outcomes are nearly irreducible — the same pitch in the same count
yields a home run or a groundout.

| | SD (runs) |
|---|---|
| swing target, pitch to pitch | 0.3015 |
| *within* one location × count cell | 0.3087 |
| *between* cells — the learnable part | **0.0415** |

**Only 1.9% of pitch-level variance is learnable at all.** A model capturing
every bit of available signal would cut RMSE by about 1%, so "0.8% over a
count-only lookup" is close to what success looks like on that measure.

Scored on the cell means it is actually estimating, the model recovers **90.6%**
of the available signal with location and count, and **94.5%** with the full
pitch frame. The features help materially; RMSE cannot show it.

**The events are predictable even though the run value is not.** Whether a
batted ball falls in is near-luck; whether the hitter misses is not. Whiff
probability predicts at **AUC 0.769**, with velocity and movement adding clearly
over location and count (0.735 → 0.769; log loss 14.9% → 18.1% better than base
rate). That is the case for modelling whiff / foul / in-play separately rather
than regressing run value directly.

---

## The take model is a called-strike probability

A take ends three ways and the target is the league run value of
`(outcome, count)`, so given the count the only thing location can say is
whether it will be called a strike.

Regressing the take model's predictions on a fitted `P(called strike)` within
each count gives **median R² 0.991**, with slopes matching
`RE(CS, c) − RE(ball, c)` to three decimals — on 3-2, −0.611 against an expected
−0.614. It is learning an umpire, not a run-value surface. The explicit
structural form can be substituted for interpretability at no cost in accuracy.

---

## Personalization is real, and easy to measure wrongly

Hitters cannot cover the whole zone, and they do not cover the same part of it:

- at the **same location**, hitters differ by **1.78 mph** of expected exit
  velocity — about **0.6× the entire league-wide location effect** (87.3–90.3
  mph across the zone)
- the modal "best cell" holds only **20%** of hitter-seasons, with real mass
  across five or six

**Estimator matters enormously.** A per-hitter hot-zone surface reproduces
itself year over year at:

| estimator | YoY r |
|---|---|
| raw bins, or a convex hull over the top 5% | 0.28–0.33 |
| kernel-smoothed + empirical-Bayes shrunk | 0.64–0.68 |
| same, two-season prior window | 0.71 |

Hitter surfaces are nearly three-dimensional — 3 principal components explain
89% of between-hitter variation, 5 explain 96.5% — which is why pooling recovers
so much.

**And the metric has to be able to see it.** The feature moves `Q_swing` by a
quarter of its own standard deviation but flips the *recommended action* on only
**1.7%** of pitches. A sign-based score therefore cannot register it:

| metric | split-half | YoY R² | next-season r |
|---|---|---|---|
| value of action taken | 0.751 → **0.888** | 0.507 → **0.690** | 0.099 → **0.228** |
| chosen minus counterfactual | 0.818 → 0.836 | 0.587 → 0.618 | 0.069 → 0.105 |
| correct decision (±1) | 0.815 → 0.816 | 0.566 → 0.564 | 0.091 → 0.091 |

Under the first, construct validity *degrades* (−0.864 → −0.643) — the tell that
the feature is pulling the score toward measuring hitting ability, which is also
why it predicts production so much better. Under the second, reliability,
contamination and usefulness all improve together.

**A feature and a metric cannot be evaluated independently.**

---

## The data

**2026 is a different measurement regime.** Statcast moved `plate_x`/`plate_z`
from front-of-plate to middle-of-plate and switched `sz_top`/`sz_bot` to the ABS
zone. The location shift is ~1 inch vertically and depends on pitch type (0.7 in
for a four-seamer, 1.5 in for a curveball), so it must be converted from the
pitch trajectory rather than offset by a constant.

**The ABS zone is exactly 27%–53.5% of batter height.** `sz_top/0.535` and
`sz_bot/0.270` agree to 0.0000 in across all 659 batters — which also recovers
batter height, giving one zone definition valid in every season.

**ABS judges "any part of the ball", not its centre** — the same convention as
the rulebook. The empirical 50% called-strike boundary sits one ball radius
outside the nominal zone. The 2026 change is the nominal zone (2.64 in lower at
the top), not the convention.

**The called zone tightened under ABS.** On a common zone definition the 50%
boundary went 0.172 → 0.122 ft and the effective called area shrank 8.6%, with
2026 landing on the ball radius — the called boundary converging on the true
one. The lefty strike shrank by only ~20%, as expected when a few pitches per
game are challenged.

**Run values do not drift** — at most 0.014 runs across six seasons, so one
table fitted on the training years suffices.

**`delta_run_exp` is a deterministic base–out–count lookup** (within-group SD
0.00000000). Grouping by `(outcome, count)` alone averages over base–out states,
whose means span about a quarter of a run — the price of a context-neutral
target.

**`field_error` is worth +0.461 runs against −0.250 for `field_out`**, close to
a single. Folding them together mis-prices 6,524 events by 0.71 runs each.

**Counterfactual support is sufficient.** The thinnest cells are 3-0 off the
plate (~18 league swings per season) but those are also the least ambiguous, so
error there cannot flip a decision. Support correlates **+0.68** with ambiguity —
the close calls have the most data.

**Exclusions.** `automatic_ball`/`automatic_strike` are non-decisions (pitch
clock and intentional walks, ~13.9k rows). 2021 carries 402 pitcher-batters
against 16–30 later. Overseas games are dropped; Toronto is kept.

---

## Method notes worth keeping

- **Judge a feature by the best available estimator, not the naive one.** The
  hot zone goes from r ≈ 0.30 to ≈ 0.71 with no new data.
- **A poor estimator of a signal does not necessarily harm the metric**, and a
  good one does not necessarily help it. Measure the metric.
- **Prior-window features drift when the window grows.** The hull's flagged area
  nearly doubled from 2022 to 2026 purely from accumulated history. Fix the
  window.
- **Judge counterfactual support by ambiguity, not raw counts.** Thin cells
  coincide with obvious decisions, where a large gap makes the call robust.
- **Report construct validity as partial correlations.** Chase rate and
  zone-swing rate correlate +0.499 through aggression, so a raw correlation
  against either is confounded and reads as though the metric ignores half the
  construct.
- **Never report RMSE without its floor.** The target is a function of
  `(outcome, count)`, so a count-only lookup is the baseline; the level alone
  mostly reflects which action is being scored.
- Hitter features for season *t* must come from seasons before *t*.
- Realized bat speed or exit velocity on the swing being graded is never a
  feature — that is execution, not decision.
- Contact quality cannot change an umpire's call, so hitter features belong on
  the swing side only. Verified: adding one to the take model moved its RMSE
  from 0.04283 to 0.04298, marginally worse.
