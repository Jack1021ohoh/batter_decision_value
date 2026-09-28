# Batter Decision Value

Valuing MLB hitters' swing/take decisions from Statcast pitch data, independently
of how each decision happened to turn out.

At the moment of decision a hitter faces two counterfactual action values —
`Q_swing(s)` and `Q_take(s)`, the expected change in run expectancy from
swinging or taking a given pitch in a given state. Their difference says which
action was better, and the gap between the best action and the chosen one is
the decision's cost:

```
Δ(s)    = Q_swing(s) − Q_take(s)
regret  = max(Q_swing, Q_take) − Q_chosen
```

A hitter can decide correctly and make an out, or decide badly and get a hit.
This framework separates the two.

## What is here

| | |
|---|---|
| Data | 2021–2026 regular seasons, 4.24M pitches, overseas neutral-site games excluded |
| `src/data.py` | loading, caching, cleaning, 2026 harmonization, the common strike zone |
| `src/evaluate.py` | the shared harness — rolling-origin folds, paired held-out accuracy, calibration, guardrail checks |
| `notebooks/eda.ipynb` | exploratory analysis, seven sections, each ending in a decision |
| `notebooks/v1_baseline.ipynb` | location and count — the simplest design worth measuring, and the floor |
| `notebooks/v2_baseline.ipynb` | v1 plus a binary hot-zone flag |
| `notebooks/v3.ipynb` | the metric choice, the pitch-frame and hot-zone ladder, the take-model check, contamination |
| **`notebooks/v4_decomposition.ipynb`** | the swing decomposed into whiff / foul / in play, and the two outputs |

Every model result is **held out**: each of 2023, 2024 and 2025 is scored by a
model that never saw it. 2026 is kept out of model selection, but it is not a
clean test: earlier versions of this project scored it, printed a 2026
leaderboard, and averaged its pairs into the headline figures that informed the
metric and feature choices. It will be reported as a previously inspected,
out-of-regime evaluation — the first ABS season. A confirmatory test needs a
season no version has seen, so 2027 is reserved for that.

**Models are chosen by accuracy, not by how the score looks.** Between two
variants, held-out squared error on the same pitches decides, with a 95%
interval from resampling games; calibration checks the winner; the
player-metric checks (below) are guardrails. They show a score is stable and
plausible, but cannot show a model is right.

### The two outputs (v4)

| output | question | chase \| zone-swing | zone-swing \| chase | split-half | YoY R² | Zone% \|r\| | next-season partial r |
|---|---|---|---|---|---|---|---|
| **generic** | a good decision for a typical hitter? | −0.914 | 0.706 | 0.816 | 0.583 | 0.270 | 0.100 |
| **personalized** | a good decision for *this* hitter? | −0.862 | 0.582 | 0.847 | 0.615 | 0.184 | 0.173 |

Both model a swing as what it produces — a whiff, a foul, or a ball in play —
on the full pitch frame (batter-frame location, handedness, velocity, movement,
pitch type), and score each decision with `signed_edge`, the value of the
chosen action minus the alternative. The personalized model adds each hitter's
hot zone to the value of a ball in play and his whiff and foul tendencies to the
outcome model. It is the most accurate model found; its score is more reliable,
less contaminated and more predictive, and tracks plate discipline less — by
design, since the right decision for a hitter who punishes strikes differs from
the right decision for a typical one. The two correlate 0.939 across
hitter-seasons.

### How the models were chosen

Each step against the one it replaces, on the same held-out swings (negative =
better; takes in brackets where they move):

| step | swing MSE, paired (95% interval) |
|---|---|
| v1 → v3's full pitch frame | −0.264% (−0.294, −0.237); takes −8.7% |
| + hot-zone surface | −0.062% (−0.090, −0.039) |
| v2's binary hull, vs v1 | −0.018%; takes +0.27% as designed (in both models) |
| decompose the swing, vs direct regression | −0.109% (−0.129, −0.091) |
| + contact priors | −0.050% (−0.065, −0.035) |

The swing figures are small because most of a swing's squared error is outcome
luck; the paired comparison cancels it, which is why the intervals sit well
away from zero. Putting hitter features in the take model made it worse
every time — contact skill cannot change an umpire's call.

### Each version

Held out 2023–25 on the same personalized folds and learner. v1 and v2 score
the value of the action taken, because that is their design; v3 and v4 score
`signed_edge`. The model columns compare across every row; the player checks
compare within a metric.

| version | swing MSE vs count-only | swing calibration slope | take MSE vs count-only | chase \| zone-swing | zone-swing \| chase | split-half | YoY R² | Zone% \|r\| | next-season |
|---|---|---|---|---|---|---|---|---|---|
| v1 — location, count | −1.61% | 1.067 | −67.2% | −0.890 | 0.678 | 0.763 | 0.538 | 0.116 | 0.129 |
| v2 — + binary hull (both models) | −1.63% | 1.056 | −67.1% | −0.803 | 0.548 | 0.794 | 0.540 | 0.058 | 0.129 |
| v3 — full pitch frame + hot zone | −1.93% | 1.042 | −70.1% | −0.887 | 0.660 | 0.844 | 0.630 | 0.172 | 0.166 |
| **v4 generic** — decomposed, pitch frame | −1.95% | 1.044 | −70.1% | −0.914 | 0.706 | 0.816 | 0.583 | 0.270 | 0.100 |
| **v4 personalized** — + hot zone, contact priors | −2.09% | 0.960 | −70.1% | −0.862 | 0.582 | 0.847 | 0.615 | 0.184 | 0.173 |

**v1 → v2: not better.** The hull helps the swing model by 0.018% of MSE and
hurts the take model by 0.27%, because v2 feeds a contact feature to a model of
an umpire's call. Its score gains reliability and loses construct validity
(zone-swing | chase 0.678 → 0.548): the flag mostly marks the middle of the
zone, so it pulls the score toward "is he a good hitter" more than it improves
the model.

**v1 → v3: better on every model column.** The full pitch frame cuts take MSE
by 8.7% and swing MSE by 0.26%, and the hot-zone surface — in the swing model
only — cuts swing MSE a further 0.06%. With the metric held at `signed_edge`,
v1's features score YoY 0.594, Zone% 0.298 and next-season 0.084; v3's are
more reliable (0.630), less contaminated (0.172) and more predictive (0.166),
at a construct-validity cost (−0.921 / +0.703 → −0.887 / +0.660) that comes
entirely from the hot zone. v3's row is its selected model as that notebook
reports it — recalibrated on early-stopping games, a step later found not to
help.

**v3 → v4: a better swing model, and a second question.** Modelling the swing as
whiff / foul / in play, with v3's features unchanged, cuts swing MSE by 0.11%.
It barely changes the score: without hitter features, the direct and decomposed
models' hitter values correlate 0.998. Adding the
hitter's whiff and foul tendencies cuts it by a further 0.05% and gives the
**personalized** output — the most accurate model, with the most reliable and
predictive score, but one that tracks plate discipline less (zone-swing | chase
0.660 → 0.582) and is slightly more contaminated than v3 (0.184 against 0.172).
The **generic** output drops hitter features altogether: about as accurate as
v3 (swing MSE 1.95% below count-only, against 1.93% — the decomposition makes
up for the missing hot zone), with the strongest construct validity of any
`signed_edge` row (−0.914 / +0.706), at the price of Zone% 0.270 and
next-season 0.100. Which one to read depends on the question: a good decision
for a typical hitter, or for this one.

**The metric is a trade, not a clean win.** By the criterion fixed before any
numbers, a sign-only score (+1 right, −1 wrong) has better construct validity.
`signed_edge` keeps the run-value magnitude, as every published metric does,
and registers a feature that changes how much a swing is worth without
flipping the decision. The magnitude carries a pitch-mix contamination the
field shares and this project has not solved — see [`FINDINGS.md`](FINDINGS.md).

**Calibration is close, not exact.** v1's original fixed settings left the swing
model's values about a fifth too compressed; every version now uses early
stopping, which fixes most of it. The remainder (swing slopes 0.94–1.09) varies
from fold to fold, and two recalibration maps both failed to improve it, so the
models are used as fitted.

**Against simple benchmarks**, O-Swing% is heavily contaminated (Zone% |r|
0.421). Z-Swing% − O-Swing% is less contaminated than either output (0.114) and
nearly as predictive as the personalized one (0.146 against 0.173), though less
reliable — the benchmark to beat on contamination.

[`FINDINGS.md`](FINDINGS.md) collects what the data established — results, the
metric analysis, the 2026 ABS measurement regime, and the method notes worth
carrying forward.
[`mlb_swing_decision_related_work.md`](mlb_swing_decision_related_work.md)
reviews the public and academic work this builds on (Yee–Deshpande, EAGLE,
SEAGER, SwRV, SOTO, Nestico, Creally, Vock & Vock).

## What the EDA established

Full detail in `notebooks/eda.ipynb`; these are the results that shape the
models.

**The 2026 ABS season is usable, after harmonization.** Statcast moved
`plate_x`/`plate_z` from front-of-plate to middle-of-plate in 2026 and switched
`sz_top`/`sz_bot` to the ABS zone. The location shift is ~1 inch vertically and
depends on pitch type (0.7 in for a four-seamer, 1.5 in for a curveball), so it
is converted exactly from the pitch trajectory rather than offset.

**The ABS zone is 27%–53.5% of batter height, exactly.** In 2026 the ratio
`sz_bot/sz_top` is 0.5047 for every batter, with zero spread. Applying that band
to each batter's listed height gives one zone definition valid in every season.
Without it, cross-era comparisons are confounded, because the ABS zone is
~2.8 in shorter than the operator-set zone it replaced.

**ABS judges "any part of the ball", not its centre** — the same convention as
the rulebook. The empirical 50% called-strike boundary sits one ball radius
outside the nominal zone.

**The called zone tightened under ABS.** On the common zone the 50% boundary
went 0.190 → 0.121 ft and the effective called area shrank 8.1%, with 2026
landing on the ball radius: the called boundary converged on the true one. The
lefty strike shrank by only ~20%, as expected when just a few pitches per game
are challenged.

**Counterfactual support is sufficient.** The thinnest cells are 3-0 off the
plate (~22 league swings per season), but those are also the least ambiguous,
so estimation error there cannot flip a decision. Support correlates +0.65 with
ambiguity — the close calls have the most data.

**Hot zones are real and stable, if estimated properly.** Per-hitter
exit-velocity surfaces reproduce year over year at r ≈ 0.30 from raw bins, but
**r ≈ 0.63–0.68 when kernel-smoothed and shrunk toward the league**, and a
two-season prior beats a one-season one in every season tested. Hitter surfaces
are nearly three-dimensional (89% of between-hitter variance in three
components), which is why pooling recovers so much.

A binary flag over the same signal is a poor estimator of it and barely helps
the model, because 95% of the pitches it marks are in the strike zone, so it
acts as a coarse location feature rather than as personalization. The
continuous surface improves the model three times as much, and shows up in the
score only under a metric that keeps the run-value magnitude: a sign-based
score barely registers it, since the feature flips the recommended action on
just 2.0% of pitches.

## Getting started

```bash
uv sync
brew install libomp          # macOS: LightGBM needs the OpenMP runtime
```

Then fetch the data (slow — a full season per call) and build the cache:

```bash
uv run jupyter lab           # run notebooks/data_fetch.ipynb, editing YEARS
uv run python -c "from src.data import build_cache; build_cache(range(2021, 2027))"
```

```python
from src.data import load_seasons, add_zone_frame
df = add_zone_frame(load_seasons(range(2021, 2027)))
```

`load_seasons()` applies the 2026 harmonization and the cleaning rules, so
every analysis starts from the same definitions.

## Layout

```
src/data.py                 loading, caching, cleaning, 2026 harmonization, the common zone
src/features.py             hitter features: prior-season hull, location surfaces (hot zone, contact priors)
src/baselines.py            the two action models (take, swing), the shared learner, how they score a pitch
src/decomposition.py        the whiff / foul / in-play swing model
src/decision.py             the per-pitch decision scores (signed_edge selected)
src/evaluate.py             folds, paired held-out accuracy, calibration, guardrail checks

notebooks/data_fetch.ipynb  Statcast pulls (Stats API season bounds, overseas games excluded)
notebooks/eda.ipynb         exploratory analysis, §1–§7
notebooks/v1_baseline.ipynb location + count
notebooks/v2_baseline.ipynb v1 + nitro zone, de-leaked
notebooks/v3.ipynb          metric choice, pitch frame, take-model check, hot-zone surface
notebooks/v4_decomposition.ipynb  swing decomposition vs direct twin; generic and personalized outputs

data/                       raw CSVs and parquet cache (gitignored)
FINDINGS.md                 measured results and method notes
AGENTS.md                   guidance for coding agents (CLAUDE.md imports it)
mlb_swing_decision_related_work.md
```

## Data

MLB Statcast via [pybaseball](https://github.com/jldbc/pybaseball), regular
season only. Games outside the US and Canada are excluded — international
series are played at neutral sites with temporary tracking installations.
Toronto is kept; Rogers Centre is a permanent park.
