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

## Status

**Rebuilding.** Three earlier iterations (v1–v3) are retired; see
[Why v3 was retired](#why-v3-was-retired). The current work is:

| Done | |
|---|---|
| Data | 2021–2026 regular seasons, 4.24M pitches, overseas neutral-site games excluded |
| Loader | `src/data.py` — caching, cleaning, 2026 harmonization, derived features |
| EDA | `notebooks/eda.ipynb` — seven sections, each ending in a decision |
| Harness | `src/evaluate.py` — reliability, validity and contamination checks, shared by every variant |
| v1 | `notebooks/v1_baseline.ipynb` — location + count, re-run on the corrected pipeline |
| v2 | `notebooks/v2_baseline.ipynb` — v1 + nitro zone, leak and silent-deletion bug fixed |
| **v3** | `notebooks/v3.ipynb` — the patch: new metric, pitch-frame features, take-model check, personalization test |

| Next | |
|---|---|
| Track B | event-decomposition redesign — though v3 puts two of its premises in doubt |

### Where the baselines landed

Each row is the same pipeline, learner and hyperparameters; only the design
differs. Higher is better except the Zone% column, where lower means less
contaminated by the pitches a hitter happened to see.

| | swing RMSE vs count-only | split-half r | YoY R² | Zone% \|r\| | next-season partial r |
|---|---|---|---|---|---|
| v1 (2021–26) | 0.2985 / 0.3010 | 0.750 | 0.519 | 0.077 | 0.092 |
| v1 (2022–26 window) | 0.2974 / 0.2998 | 0.753 | 0.517 | 0.077 | 0.095 |
| **v2** (2022–26) | 0.2973 / 0.2998 | 0.786 | 0.530 | 0.033 | 0.105 |
| **v3** (the patch) | 0.2982 / 0.3010 | 0.806 | 0.560 | 0.207 | 0.085 |

v3's gain is not in this table. Its construct validity — does the metric punish
chasing *and* reward attacking hittable pitches — is **−0.924 / +0.871**
against v1's −0.849 / +0.594. Changing how per-pitch scores aggregate did that.

The other large effect is personalization, and it interacts with the metric
choice. A hitter's hot zone differs from another's at the *same location* by
1.78 mph of expected exit velocity — about 0.6× the entire league-wide location
effect — and hitters differ in where their best region sits. Adding it takes
next-season predictive correlation from 0.099 to **0.228** under a
magnitude-sensitive score, while a sign-based score cannot register it at all
(it flips the recommended action on only 1.7% of pitches). Which score to
prefer depends on whether you want to rank pure decision-making or produce a
number that carries information about the hitter.

Two things to read off this. The baselines are **reliable but barely useful** —
a partial correlation of ~0.10 against next-season production means the metric
adds little to simply knowing how a hitter already hit, and that is the number
the rebuild has to move. And the **swing model beats a count-only lookup by
under 1%**, against ~44% for the take model: location and count say almost
nothing about what happens when a hitter swings, which is what Track B's event
decomposition is aimed at.

[`FINDINGS.md`](FINDINGS.md) collects what the data established — results, the
metric analysis, the 2026 ABS measurement regime, and the method notes worth
carrying forward.
[`mlb_swing_decision_related_work.md`](mlb_swing_decision_related_work.md)
reviews the public and academic work this builds on (Yee–Deshpande, EAGLE,
SEAGER, SwRV, SOTO, Nestico, Creally, Vock & Vock).

## Why v3 was retired

Year-over-year stability of the player metric fell across iterations — 0.57,
then 0.34, then 0.16 — and the causes turned out to be independent:

- **The player metric dropped swings entirely.** v3 scored the share of a
  hitter's *takes* the model judged hittable. Chases never entered it, and a
  hard threshold at zero counted a miss of 0.001 runs the same as one of 0.08.
- **The per-pitch score double-counted the strike probability.**
  `Q_swing·P(strike) − Q_take·(1−P(strike))` reweights a take value that
  already contains `P(ball)`.
- **The nitro zone leaked its own outcome.** The convex hull was drawn through
  the top 5% of *that same season's* exit velocities, so `in_nitro` partly
  encoded the label — and on held-out data the hull used the future. Rebuilding
  it from prior seasons only shows the leak was the whole problem: the
  de-leaked hull slightly *improves* on v1.
- **Run values were refit per season**, so each year's targets came from its
  own run environment.
- **`field_error` was mapped to `field_out`**, assigning −0.250 runs to an
  event worth +0.461.

The notebooks are not in the working tree. They are in git history at
[`fa48b14`](../../commit/fa48b14) (`git show fa48b14:batter_decision_value_v3.ipynb`).

## What the EDA established

Full detail in `notebooks/eda.ipynb`; these are the results that shape the
rebuild.

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

A caveat the v2 rebuild added: a hull is a poor estimator of that *signal*, but
that does not mean it damages the *metric*. Rebuilt without the leak it edges
v1 on every measure. The hull's real limitation is different — 91% of the
pitches it flags are in the strike zone, so it is mostly acting as a coarse
location feature rather than as personalization.

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
