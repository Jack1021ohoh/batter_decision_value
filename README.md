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
| `src/data.py` | loading, caching, cleaning, 2026 harmonization, derived features |
| `src/evaluate.py` | the shared harness — reliability, validity and contamination checks |
| `notebooks/eda.ipynb` | exploratory analysis, seven sections, each ending in a decision |
| `notebooks/v1_baseline.ipynb` | location and count — the simplest design worth measuring, and the floor |
| `notebooks/v2_baseline.ipynb` | v1 plus a binary hot-zone flag |
| **`notebooks/v3.ipynb`** | the current model: batter-frame features, a continuous hot-zone surface, and the metric comparison behind both |

Next is an event-decomposition redesign — modelling whiff, foul and in-play
separately rather than regressing run value directly. v3 supports the premise:
whiff probability predicts at AUC 0.769 while the run value of a swing is
mostly irreducible.

### Where the baselines landed

Two readings, answering different questions.

**Each version as designed.** v1 and v2 score the value of the action taken,
because that is their design; v3 changes the metric as well as the features, so
these rows are not a like-for-like ladder.

| | split-half r | YoY R² | Zone% \|r\| | next-season partial r |
|---|---|---|---|---|
| v1 (2021–26) | 0.750 | 0.519 | 0.077 | 0.092 |
| v2 (2022–26) | 0.786 | 0.530 | 0.033 | 0.105 |

**With the metric held fixed, so only the features vary.** This is the
comparison that supports any claim about what the added features buy.

Each row is a complete feature set, not an increment on the row above: the
last two drop the binary hull, because it and the continuous surface estimate
the same thing — a hitter's hot zone — so carrying both would be redundant
rather than cumulative.

| feature set | split-half r | YoY R² | Zone% \|r\| | next-season partial r |
|---|---|---|---|---|
| location, count | 0.817 | 0.590 | 0.274 | 0.049 |
| location, count, hull | 0.816 | 0.587 | 0.250 | 0.056 |
| batter frame, handedness, pitch chars | 0.815 | 0.583 | 0.269 | 0.070 |
| **… plus hot-zone surface** | **0.835** | **0.613** | **0.192** | **0.110** |

Going from location-and-count to the full model **doubles** next-season
predictive validity, cuts pitch-mix contamination by about 30%, and raises
reliability, with construct validity —
does the metric punish chasing *and* reward attacking hittable pitches — flat
at −0.884 / +0.643 against −0.894 / +0.626.

Most of that comes from personalization. A hitter's hot zone differs from
another's at the *same location* by 1.78 mph of expected exit velocity, about
0.6× the entire league-wide location effect, and hitters differ in where their
best region sits. The estimator matters enormously: a convex hull over the top
5% of balls in play reproduces itself year over year at r ≈ 0.30, a
kernel-smoothed shrunk surface at r ≈ 0.68.

The metric keeps the run-value magnitude, as every published metric does. That
carries a known pitch-mix contamination the field shares and this project did
not solve — four attempts are documented in [`FINDINGS.md`](FINDINGS.md).

**A caution about the swing model.** It beats a count-only lookup by under 1%
on pitch-level RMSE, which reads as though nothing observable before the pitch
predicts what follows a swing. That reading is wrong. Only **1.9%** of the
variance in swing run value is learnable at all — the rest is the difference
between a home run and a groundout on identical pitches — and of that learnable
part the model recovers 90.6% with location and count, 94.5% with the full
pitch frame. RMSE is simply the wrong instrument for a conditional-mean
estimator.

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

**The ABS zone is 27%–53.5% of batter height, exactly.** `sz_top/0.535` and
`sz_bot/0.270` agree to 0.0000 in across all 659 batters. That also recovers
batter height, which gives one zone definition valid in every season — without
it, cross-era comparisons are confounded, because the ABS zone is ~2.8 in
shorter than the operator-set zone it replaced.

**ABS judges "any part of the ball", not its centre** — the same convention as
the rulebook. The empirical 50% called-strike boundary sits one ball radius
outside the nominal zone.

**The called zone tightened under ABS.** On a common zone definition the 50%
boundary went 0.172 → 0.122 ft and the effective called area shrank 8.6%, with
2026 landing on the ball radius: the called boundary converged on the true one.
The lefty strike shrank by only ~20%, as expected when just a few pitches per
game are challenged.

**Counterfactual support is sufficient.** The thinnest cells are 3-0 off the
plate (~18 league swings per season), but those are also the least ambiguous,
so estimation error there cannot flip a decision. Support correlates +0.68 with
ambiguity — the close calls have the most data.

**Hot zones are real and stable, if estimated properly.** Per-hitter
exit-velocity surfaces reproduce year over year at r ≈ 0.30 from raw bins, but
**r ≈ 0.64–0.68 when kernel-smoothed and shrunk toward the league**, and 0.71
with a two-season prior. Hitter surfaces are nearly three-dimensional (89% of
between-hitter variance in three components), which is why pooling recovers so
much.

A binary flag over the same signal is a poor estimator of it but does not
damage the metric — it simply contributes little, because 91% of the pitches it
marks are in the strike zone, so it acts as a coarse location feature rather
than as personalization. The continuous surface is what makes personalization
pay, and only under a metric that keeps the run-value magnitude: a sign-based
score cannot see it, since the feature flips the recommended action on just
1.7% of pitches.

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
src/data.py                 loading, caching, cleaning, 2026 harmonization, derived features
src/features.py             the nitro zone: prior-season hulls, membership test
src/baselines.py            v1/v2 action models and the per-pitch scoring rules
src/evaluate.py             the shared harness — reliability, validity, Zone% contamination

notebooks/data_fetch.ipynb  Statcast pulls (Stats API season bounds, overseas games excluded)
notebooks/eda.ipynb         exploratory analysis, §1–§7
notebooks/v1_baseline.ipynb location + count
notebooks/v2_baseline.ipynb v1 + nitro zone, de-leaked

data/                       raw CSVs and parquet cache (gitignored)
FINDINGS.md                 measured results and method notes
mlb_swing_decision_related_work.md
```

## Data

MLB Statcast via [pybaseball](https://github.com/jldbc/pybaseball), regular
season only. Games outside the US and Canada are excluded — international
series are played at neutral sites with temporary tracking installations.
Toronto is kept; Rogers Centre is a permanent park.
