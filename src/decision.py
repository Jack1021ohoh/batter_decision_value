"""Per-pitch decision scores.

Each function turns the counterfactual pair produced by `baselines.predict_both`
-- `q_swing`, `q_take`, `edge = q_swing - q_take` -- into one number per pitch.
Aggregating that number over a hitter's season gives his decision value.

The choice between them is not cosmetic. They disagree about which hitters
decide well, and they are contaminated by pitch mix in different directions:

* `chosen_value` scores the action taken. A taken ball and a swung-at strike
  both pay, so the two largely cancel and it is the least sensitive to which
  pitches a hitter saw -- but conceptually it credits him for the pitches he
  was thrown, not only for his choices.
* `signed_edge` scores how much better the chosen action was than the
  alternative. It pays about eight times more per pitch for an obvious take
  than for a genuinely close call, so it can reward being thrown junk.
* `regret` is `max(0, -signed_edge)`: the same quantity with the positive half
  discarded. It cannot be inflated by easy correct decisions, because correct
  decisions score zero -- but its largest error is taking a hittable pitch, so
  a hitter thrown more strikes accumulates more of it.
* `close_weighted` credits correct decisions like `signed_edge` but weights by
  how close the call was rather than by the raw gap, which is an attempt to
  cancel the two biases against each other. As its scale grows the weights all
  approach 1 and it converges on `correct_decision` (correlation 0.997 across
  hitter-seasons at scale 0.8).
* `correct_decision` is simply +1 when the hitter picked the better action and
  -1 when he did not, with no run-value weighting at all. It measures best on
  construct validity -- the only check that tests whether a metric tracks swing
  decisions rather than merely tracking something stable -- because dropping the
  magnitude is what removes the pitch-mix bias that the magnitude carries.

Sign convention: every function returns a score where **higher is better**, so
`evaluate.player_metric` can aggregate any of them without special-casing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Scale of the "closeness" kernel in runs. A decision whose two action values
#: differ by more than about this much is not really a decision.
CLOSE_SCALE = 0.05


def _signed(df: pd.DataFrame) -> np.ndarray:
    """Q_chosen - Q_alternative: positive when the hitter picked the better action."""
    return np.where(df['swing'], df['edge'], -df['edge'])


def chosen_value(df: pd.DataFrame) -> pd.Series:
    """Value of the action actually taken. The baseline design's score."""
    return pd.Series(np.where(df['swing'], df['q_swing'], df['q_take']),
                     index=df.index, name='chosen_value')


def signed_edge(df: pd.DataFrame) -> pd.Series:
    """How much better the chosen action was than the alternative."""
    return pd.Series(_signed(df), index=df.index, name='signed_edge')


def regret(df: pd.DataFrame) -> pd.Series:
    """Negated distance from the best available action; 0 when the hitter was right.

    Returned negated so that higher is better, matching the other scores.
    """
    return pd.Series(-np.maximum(0.0, -_signed(df)), index=df.index, name='regret')


def close_weighted(df: pd.DataFrame, scale: float = CLOSE_SCALE) -> pd.Series:
    """Signed correctness, weighted by how close the decision was.

    `signed_edge` pays in proportion to the gap between the two action values,
    so it pays most where the decision was obvious. This pays in proportion to
    how *hard* the call was: the weight `exp(-|edge| / scale)` is near 1 when
    the two actions are nearly equal and decays as one becomes clearly better.

    A hitter is therefore credited for getting close calls right and penalised
    for getting them wrong, while obvious pitches -- which say little about his
    judgement either way -- contribute almost nothing regardless of how many of
    them he happened to see.
    """
    weight = np.exp(-np.abs(df['edge'].to_numpy()) / scale)
    return pd.Series(np.sign(_signed(df)) * weight, index=df.index, name='close_weighted')


def correct_decision(df: pd.DataFrame) -> pd.Series:
    """+1 if the hitter picked the better action, -1 if not. No magnitude.

    Discarding the size of the gap is the point. `signed_edge` pays in
    proportion to that gap, so it pays most on the most obvious pitches and a
    hitter thrown more junk scores higher for the same judgement. Scoring every
    decision equally removes that, at the cost of treating a razor-thin call and
    an obvious one alike.

    Measured across the candidates this is the strongest on construct validity
    and passes the Zone% contamination veto, while giving up only a little
    reliability to `signed_edge`.
    """
    return pd.Series(np.sign(_signed(df)), index=df.index, name='correct_decision')


#: Every candidate, for iterating in the comparison.
SCORES = {
    'chosen_value': chosen_value,
    'signed_edge': signed_edge,
    'regret': regret,
    'close_weighted': close_weighted,
    'correct_decision': correct_decision,
}

#: The selected score. `signed_edge` keeps the run-value magnitude, which is
#: what every published metric does (SwRV, SOTO, Nestico, Creally, EAGLE) and
#: what a decision metric needs if it is to register a feature that shifts
#: Q_swing without flipping the decision.
#:
#: `correct_decision` measures cleaner on pitch-mix contamination, but it buys
#: that by discarding magnitude -- treating a nearly indifferent call and an
#: obvious blunder alike -- and is consequently blind to the single most
#: valuable feature found in this project. Four attempts to get both
#: properties (regret, close_weighted, a departure-weighted score, and
#: opportunity standardization) all failed; see FINDINGS.md.
DEFAULT_SCORE = 'signed_edge'


def add_scores(df: pd.DataFrame, which=None) -> pd.DataFrame:
    """Attach the named per-pitch scores as columns."""
    df = df.copy()
    for name in (which or SCORES):
        df[name] = SCORES[name](df)
    return df
