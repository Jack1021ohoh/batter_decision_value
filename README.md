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
| Data | 2021–2026 regular seasons, 4.28M pitches, overseas neutral-site games excluded |
| `src/data.py` | loading, caching, cleaning, 2026 harmonization, the common strike zone |
| `src/decomposition.py` | the swing model — whiff / foul / in play — and saving / loading fitted models |
| `src/evaluate.py` | the shared harness — rolling-origin folds, paired held-out accuracy, calibration, guardrail checks |
| `notebooks/eda.ipynb` | exploratory analysis, seven sections, each ending in a decision |
| `notebooks/v1_baseline.ipynb` | location and count — the simplest design worth measuring, and the floor |
| `notebooks/v2_baseline.ipynb` | v1 plus a binary hot-zone flag |
| `notebooks/v3.ipynb` | the metric choice, the pitch-frame and hot-zone ladder, the take-model check, contamination |
| `notebooks/v4_decomposition.ipynb` | the swing decomposed into whiff / foul / in play, against a direct regression |
| **`notebooks/v5_in_play.ipynb`** | a ball in play valued as events; the two outputs built on it |
| **`notebooks/v6_shrink.ipynb`** | the personalized swing values shrunk toward the generic ones |
| **`notebooks/final_models.ipynb`** | the two outputs refitted on every development season, saved, and scored on 2026 |

Every model result is **held out**: each of 2023, 2024 and 2025 is scored by a
model that never saw it. 2026 is kept out of model selection, but it is not a
clean test: earlier versions of this project scored it, printed a 2026
leaderboard, and averaged its pairs into the headline figures that informed the
metric and feature choices. It is reported as a previously inspected,
out-of-regime evaluation — the first ABS season. A confirmatory test needs a
season no version has seen, so 2027 is the test season. That does not keep 2026
out of training: once 2027 is complete, 2026 becomes ordinary data — a training
season and an extra held-out fold for model selection, the only one in the ABS
regime — and the final model is fitted through 2026, with every decision
locked before 2027 is scored once.

**Models are chosen by accuracy, not by how the score looks.** Between two
variants, held-out squared error on the same pitches decides, with a 95%
interval from resampling games; calibration checks the winner; the
player-metric checks (below) are guardrails. They show a score is stable and
plausible, but cannot show a model is right.

### The two outputs (v5, v6)

| output | question | chase \| zone-swing | zone-swing \| chase | split-half | YoY R² | Zone% \|r\| | next-season partial r |
|---|---|---|---|---|---|---|---|
| **generic** | a good decision for a typical hitter? | −0.910 | 0.689 | 0.816 | 0.583 | 0.276 | 0.094 |
| **personalized** | a good decision for *this* hitter? | −0.888 | 0.627 | 0.833 | 0.598 | 0.224 | 0.144 |

Both model a swing as what it produces — a whiff, a foul, or a ball in play —
and a ball in play as one of six events (single, double, triple, home run, out,
error), each weighted by its run value in the count. Both use the full pitch
frame (batter-frame location, handedness, velocity, movement, pitch type) and
score each decision with `signed_edge`, the value of the chosen action minus the
alternative. The personalized model adds each hitter's hot zone to the
in-play-event model and his whiff and foul tendencies to the outcome model, and
keeps about two thirds of what those add (k = 0.68; v6): taken at full strength
they overstated hitters' differences, as noisy two-season estimates do. It is
the most accurate model found; its score is more reliable, less contaminated
and more predictive, and tracks plate discipline a little less — by design,
since the right decision for a hitter who punishes strikes differs from the
right decision for a typical one. The two correlate 0.972 across
hitter-seasons.

### The final models

`final_models.ipynb` refits both outputs on every development season — generic
on 2021–2025, personalized on 2022–2025 — saves them to `models/` (gitignored),
and scores 2026 from the reloaded files. On 2026, out of regime, they hold up:

| | swing MSE vs count-only | take MSE vs count-only | swing calibration slope |
|---|---|---|---|
| generic | −1.98% | −74.3% | 1.003 |
| personalized | −2.18% | −74.5% | 0.996 |

The 2026 personalized leaderboard is led by Torres, Seager, Soto, Will Smith and
Nick Kurtz; Báez, Edmundo Sosa and Angel Martínez are at the bottom.

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
| a ball in play as events, vs regression (personalized) | −0.038% (−0.056, −0.018) |
| the same, generic | −0.012% (−0.025, +0.002) — a tie |
| keep two thirds of personalization (k = 0.68), vs all of it | −0.042% (−0.050, −0.032) |

The swing figures are small because most of a swing's squared error is outcome
luck; the paired comparison cancels it, which is why the intervals sit well
away from zero. Putting hitter features in the take model made it worse
every time — contact skill cannot change an umpire's call.

### Each version

Held out 2023–25 on the same personalized folds and learner. v1 and v2 score
the value of the action taken, because that is their design; v3–v6 score
`signed_edge`. The model columns compare across every row; the player checks
compare within a metric.

| version | swing MSE vs count-only | swing calibration slope | take MSE vs count-only | chase \| zone-swing | zone-swing \| chase | split-half | YoY R² | Zone% \|r\| | next-season |
|---|---|---|---|---|---|---|---|---|---|
| v1 — location, count | −1.61% | 1.067 | −67.2% | −0.890 | 0.678 | 0.763 | 0.538 | 0.116 | 0.129 |
| v2 — + binary hull (both models) | −1.63% | 1.056 | −67.1% | −0.803 | 0.548 | 0.794 | 0.540 | 0.058 | 0.129 |
| v3 — full pitch frame + hot zone | −1.93% | 1.042 | −70.1% | −0.887 | 0.660 | 0.844 | 0.630 | 0.172 | 0.166 |
| v4 generic — decomposed, pitch frame | −1.95% | 1.044 | −70.1% | −0.914 | 0.706 | 0.816 | 0.583 | 0.270 | 0.100 |
| v4 personalized — + hot zone, contact priors | −2.09% | 0.960 | −70.1% | −0.862 | 0.582 | 0.847 | 0.615 | 0.184 | 0.173 |
| **v5 generic** (unchanged in v6) — + in-play events | −1.96% | 1.017 | −70.1% | −0.910 | 0.689 | 0.816 | 0.583 | 0.276 | 0.094 |
| v5 personalized — + in-play events | −2.12% | 0.956 | −70.1% | −0.864 | 0.576 | 0.844 | 0.606 | 0.198 | 0.159 |
| **v6 personalized** — shrunk toward generic | −2.16% | 1.007 | −70.1% | −0.888 | 0.627 | 0.833 | 0.598 | 0.224 | 0.144 |

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
models' hitter values correlate 0.998. Adding the hitter's whiff and foul
tendencies cuts it by a further 0.05% and gives the personalized output; the
generic output drops hitter features altogether. Which one to read depends on
the question: a good decision for a typical hitter, or for this one.

**v4 → v5: a ball in play as events.** Predicting the odds of each in-play
event, rather than regressing its run value, makes the personalized model more
accurate (−0.038%) and ties for the generic one. The gain rests mainly on one
fold, and the guardrails move slightly the wrong way (personalized YoY 0.615 →
0.606, Zone% 0.184 → 0.198). Both outputs use it — the generic one by decision
rather than by the rule, so the two outputs differ only in their hitter
features. It also makes personalization readable: a hitter's hot zone raises his
home-run odds by up to 1.7 points and lowers his out odds by 2.2.

**v5 → v6: personalization at two thirds strength.** The personalized swing
model was steadily over-spread — its differences between pitches about 5% too
large. Pulling its swing value two thirds of the way back toward the generic
one, `q_generic + 0.68 · (q_personalized − q_generic)`, makes it more accurate
in every held-out season (−0.042%) and calibrated (slope 0.95 → 1.01). k was
fitted on held-out swings and came out the same in each season (0.66–0.73): the
hitter priors point the right way but were believed about 1.5 times too
strongly, as noisy two-season estimates are. Construct validity improves
(−0.864 / +0.576 → −0.888 / +0.627); reliability and next-season validity give a
little back.

**The metric is a trade, not a clean win.** By the criterion fixed before any
numbers, a sign-only score (+1 right, −1 wrong) has better construct validity.
`signed_edge` keeps the run-value magnitude, as every published metric does,
and registers a feature that changes how much a swing is worth without
flipping the decision. The magnitude carries a pitch-mix contamination the
field shares and this project has not solved. It is worst on the generic
output (Zone% r 0.379 with production held fixed) and smaller on the
personalized one (0.350) — see [`FINDINGS.md`](FINDINGS.md).

**Calibration is close.** v1's original fixed settings left the swing model's
values about a fifth too compressed; every version now uses early stopping,
which fixes most of it. Two linear recalibration maps then failed, because the
remainder moved from fold to fold. What worked was the shrink: the personalized
output's steady over-spread is gone (slopes 0.98–1.04 held out, 0.996 on 2026),
and the generic output is close (0.98–1.06).

**Against simple benchmarks**, O-Swing% is heavily contaminated (Zone% |r|
0.421). Z-Swing% − O-Swing% is less contaminated than either output (0.114) and
as predictive as the personalized one (next-season 0.146 against 0.144), though
less reliable (YoY 0.557 against 0.598) — the benchmark to beat on
contamination and prediction.

[`FINDINGS.md`](FINDINGS.md) collects what the data established — results, the
metric analysis, what the EDA found (including the 2026 ABS measurement
regime), and the method notes worth carrying forward.
[`mlb_swing_decision_related_work.md`](mlb_swing_decision_related_work.md)
reviews the public and academic work this builds on (Yee–Deshpande, EAGLE,
SEAGER, SwRV, SOTO, Nestico, Creally, Vock & Vock).

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

### Using the saved models

After running `notebooks/final_models.ipynb`:

```python
from src.decomposition import load_output
from src.baselines import predict_both
from src import decision as DEC

model = load_output('models', 'personalized')       # or 'generic'; personalized comes back shrunk
scored = DEC.add_scores(predict_both(pitches, model))   # q_take, q_swing, edge, signed_edge
```

`pitches` needs the model's features — see `models/<output>/meta.json`; the
personalized model also needs the hitter priors from `features.season_hot_zone`
and `features.season_contact_priors`. The 2026 scores are already in
`models/pitch_values_2026.parquet` (per pitch, both outputs) and
`models/hitter_scores_2026.parquet` (per qualified hitter, with names).

## Layout

```
src/data.py                       loading, caching, cleaning, 2026 harmonization, the common zone
src/features.py                   hitter features: prior-season hull, location surfaces (hot zone, contact priors)
src/baselines.py                  the two action models (take, swing), the shared learner, how they score a pitch
src/decomposition.py              the whiff / foul / in-play swing model; the shrunk personalized output; save / load
src/decision.py                   the per-pitch decision scores (signed_edge selected)
src/evaluate.py                   folds, paired held-out accuracy, calibration, guardrail checks

notebooks/data_fetch.ipynb        Statcast pulls (Stats API season bounds, overseas games excluded)
notebooks/eda.ipynb               exploratory analysis, §1–§7
notebooks/v1_baseline.ipynb       location + count
notebooks/v2_baseline.ipynb       v1 + nitro zone, de-leaked
notebooks/v3.ipynb                metric choice, pitch frame, take-model check, hot-zone surface
notebooks/v4_decomposition.ipynb  swing decomposition vs direct twin
notebooks/v5_in_play.ipynb        a ball in play valued as events; the two outputs
notebooks/v6_shrink.ipynb         the personalized swing values shrunk toward the generic ones
notebooks/final_models.ipynb      final models: fitted, saved, scored on 2026

models/                           saved final models and 2026 scores (gitignored)
data/                             raw CSVs and parquet cache (gitignored)

FINDINGS.md                       measured results and method notes
AGENTS.md                         guidance for coding agents (CLAUDE.md imports it)
mlb_swing_decision_related_work.md
```

## Data

MLB Statcast via [pybaseball](https://github.com/jldbc/pybaseball), regular
season only. Games outside the US and Canada are excluded — international
series are played at neutral sites with temporary tracking installations.
Toronto is kept; Rogers Centre is a permanent park.
