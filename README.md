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
| `src/evaluate.py` | the shared harness — rolling-origin folds, held-out reliability, validity, contamination and calibration checks |
| `notebooks/eda.ipynb` | exploratory analysis, seven sections, each ending in a decision |
| `notebooks/v1_baseline.ipynb` | location and count — the simplest design worth measuring, and the floor |
| `notebooks/v2_baseline.ipynb` | v1 plus a binary hot-zone flag |
| **`notebooks/v3.ipynb`** | the current model: batter-frame features, a continuous hot-zone surface, and the metric comparison behind both |

Every model result is **held out**. Each variant is fitted on rolling-origin
folds and each of 2023, 2024 and 2025 is scored by a model that never saw it.
2026 has not been scored: it is the final test, run once after every design
decision is locked.

Next is an event-decomposition redesign — modelling whiff, foul and in-play
separately rather than regressing run value directly. v3 supports the premise:
whiff probability predicts at AUC 0.77 on held-out seasons, far better than the
run value of a swing can be predicted pitch by pitch.

### Where the baselines landed

Two readings, answering different questions.

**Each version as designed.** v1 and v2 score the value of the action taken,
because that is their design. v2 needs a prior season for its hitter feature,
so it is compared with v1 refit on the same folds.

| | chase \| zone-swing | zone-swing \| chase | split-half r | YoY R² | Zone% \|r\| | next-season partial r |
|---|---|---|---|---|---|---|
| v1 | −0.884 | 0.673 | 0.758 | 0.526 | 0.082 | 0.135 |
| v1, v2's folds | −0.883 | 0.669 | 0.757 | 0.526 | 0.081 | 0.137 |
| v2 | −0.792 | 0.535 | 0.792 | 0.535 | 0.031 | 0.135 |

**With the metric held fixed at `signed_edge`, so only the features vary.**
This is the comparison that supports any claim about what the added features
buy. Each row is a complete feature set, not an increment on the row above: the
binary hull and the continuous surface estimate the same thing — a hitter's hot
zone — so they are never combined.

| feature set | chase \| zone-swing | zone-swing \| chase | split-half r | YoY R² | Zone% \|r\| | next-season partial r |
|---|---|---|---|---|---|---|
| location, count | −0.920 | 0.699 | 0.818 | 0.594 | 0.300 | 0.082 |
| location, count, hull | −0.913 | 0.688 | 0.819 | 0.589 | 0.282 | 0.086 |
| batter frame, handedness, pitch chars | −0.923 | 0.723 | 0.817 | 0.590 | 0.274 | 0.107 |
| **… plus hot-zone surface** | −0.895 | 0.677 | **0.838** | **0.624** | **0.184** | **0.154** |

Going from location-and-count to the full model **nearly doubles** next-season
predictive validity, cuts pitch-mix contamination by 39% and raises
reliability. It costs some construct validity — whether the metric punishes
chasing *and* rewards attacking hittable pitches, each holding the other fixed —
which slips from −0.920 / +0.699 to −0.895 / +0.677.

Most of the gain comes from personalization, and all of it through the swing
model: giving the hot zone to the take model as well changes nothing. A hitter's
hot zone differs from another's at the *same location* by 1.72 mph of expected
exit velocity, about 0.6× the entire league-wide location effect, and hitters
differ in where their best region sits. The estimator matters enormously: raw
location bins reproduce themselves year over year at r ≈ 0.30, a
kernel-smoothed shrunk surface at r ≈ 0.63–0.68. The binary hull buys nothing.

**The metric is a trade, not a clean win.** By the criterion fixed before any
numbers, a sign-only score (+1 right, −1 wrong) has better construct validity.
`signed_edge` keeps the run-value magnitude, as every published metric does,
and is the only candidate that gains substantially from the hot zone without
its construct validity collapsing. The magnitude carries a pitch-mix
contamination the field shares and this project has not solved — see
[`FINDINGS.md`](FINDINGS.md).

**A caution about the swing model.** It beats a count-only lookup by under 1%
on pitch-level RMSE, which reads as though nothing observable before the pitch
predicts what follows a swing. That reading is wrong: individual swing outcomes
are close to irreducible, and the model estimates their mean. Scored on
held-out (location × count) bin means, the batter-frame model recovers most of
the location structure a count-only predictor misses. The same check suggests
it understates how much better a swing over the middle is than one at the edge:
the calibration slope is above 1 in every held-out season. That check sees only
location and count, and has no confidence interval yet.

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

A binary flag over the same signal is a poor estimator of it and contributes
nothing, because 95% of the pitches it marks are in the strike zone, so it acts
as a coarse location feature rather than as personalization. The continuous
surface is what makes personalization pay, and only under a metric that keeps
the run-value magnitude: a sign-based score barely registers it, since the
feature flips the recommended action on just 1.6% of pitches.

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
src/features.py             hitter features: prior-season hull, continuous hot-zone surface
src/baselines.py            the two action models (take, swing) and how they score a pitch
src/decision.py             the per-pitch decision scores (signed_edge selected)
src/evaluate.py             folds, held-out harness: reliability, validity, contamination, calibration

notebooks/data_fetch.ipynb  Statcast pulls (Stats API season bounds, overseas games excluded)
notebooks/eda.ipynb         exploratory analysis, §1–§7
notebooks/v1_baseline.ipynb location + count
notebooks/v2_baseline.ipynb v1 + nitro zone, de-leaked
notebooks/v3.ipynb          metric choice, pitch frame, take-model check, hot-zone surface

data/                       raw CSVs and parquet cache (gitignored)
FINDINGS.md                 measured results and method notes
mlb_swing_decision_related_work.md
```

## Data

MLB Statcast via [pybaseball](https://github.com/jldbc/pybaseball), regular
season only. Games outside the US and Canada are excluded — international
series are played at neutral sites with temporary tracking installations.
Toronto is kept; Rogers Centre is a permanent park.
