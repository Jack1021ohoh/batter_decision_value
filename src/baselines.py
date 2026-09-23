"""Baseline swing-decision models.

A baseline is only useful if it differs from the thing it is benchmarking in
exactly one respect. These keep each baseline's *design* -- its feature set and
its player metric -- and run it on the same pipeline, learner and
hyperparameters as everything else, so a difference in the scorecard is
attributable to the design rather than to how the data was prepared.

What each baseline's design was, and what it measured, is recorded in
FINDINGS.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb

SEED = 1126

#: v1's feature set. `plate_x_mid`/`plate_z_mid` are the harmonized location
#: from `data.to_middle_of_plate()` -- harmonization is a *data* correction and
#: must apply. Batter-frame mirroring and zone normalization are *design*
#: changes belonging to Track A2 and deliberately do not appear here.
V1_FEATURES = ['plate_x_mid', 'plate_z_mid', 'count']

_COMMON = {'objective': 'reg:squarederror', 'eval_metric': 'rmse',
           'random_state': SEED, 'tree_method': 'hist'}
V1_TAKE_PARAMS = {**_COMMON, 'max_depth': 8, 'learning_rate': 0.03}
V1_SWING_PARAMS = {**_COMMON, 'max_depth': 7, 'learning_rate': 0.01}
V1_ROUNDS = 200


#: Hitter-specific features. They describe how a hitter makes contact, which
#: cannot change what an umpire calls, so they belong on the swing side only.
#: `fit_v1` keeps them out of the take model unless told otherwise.
SWING_ONLY_FEATURES = {'in_nitro', 'hot_zone'}


@dataclass
class ActionModels:
    """A take model and a swing model, each with its own feature list."""
    take: xgb.Booster
    swing: xgb.Booster
    take_features: list[str]
    swing_features: list[str]

    def features_for(self, action: str) -> list[str]:
        return self.take_features if action == 'take' else self.swing_features

    def predict(self, df: pd.DataFrame, action: str) -> np.ndarray:
        booster = self.take if action == 'take' else self.swing
        return booster.predict(_dmatrix(df, self.features_for(action)))


def _dmatrix(df: pd.DataFrame, features: list[str], label=None) -> xgb.DMatrix:
    X = df[features].copy()
    if 'count' in X:
        X['count'] = X['count'].astype('category')
    return xgb.DMatrix(X, label=label, enable_categorical=True)


def fit_v1(train: pd.DataFrame, target: str = 'target',
           features: list[str] | None = None,
           take_params: dict | None = None, swing_params: dict | None = None,
           rounds: int = V1_ROUNDS, swing_only: bool = True) -> ActionModels:
    """Fit the two action models: one on takes, one on swings.

    `features` is a parameter so each variant is the same call with a different
    list rather than a copied function.

    With `swing_only=True` (the default) any feature in `SWING_ONLY_FEATURES` is
    given to the swing model and withheld from the take model. The take model
    is a called-strike probability -- regressing its predictions on a fitted
    P(called strike) gives a median R^2 of 0.991 within count -- and a hitter's
    hot zone cannot change an umpire's call; letting it in lets `Q_take` vary
    by hitter, so part of any personalization gain could arrive through the
    wrong branch. `swing_only=False` reproduces the earlier behaviour, where
    both models shared one list, for comparison.
    """
    features = list(features or V1_FEATURES)
    swing_features = features
    take_features = ([f for f in features if f not in SWING_ONLY_FEATURES]
                     if swing_only else features)
    take = train[~train['swing']]
    swing = train[train['swing']]

    return ActionModels(
        take=xgb.train(take_params or V1_TAKE_PARAMS,
                       _dmatrix(take, take_features, take[target]), rounds),
        swing=xgb.train(swing_params or V1_SWING_PARAMS,
                        _dmatrix(swing, swing_features, swing[target]), rounds),
        take_features=take_features,
        swing_features=swing_features,
    )


def predict_chosen(df: pd.DataFrame, models: ActionModels,
                   out: str = 'y_pred') -> pd.DataFrame:
    """Score each pitch with the model matching the action actually taken.

    This is v1's defining choice and its central weakness: the value of the
    chosen action says nothing about what the alternative was worth, so a
    hitter who simply sees more hittable pitches scores well. Track A1 replaces
    it with `edge = Q_swing - Q_take`, which is why this stays a named
    function rather than an inline expression.
    """
    df = df.copy()
    df[out] = np.nan
    for action, mask in (('take', ~df['swing']), ('swing', df['swing'])):
        if mask.any():
            df.loc[mask, out] = models.predict(df.loc[mask], action)
    return df


def predict_both(df: pd.DataFrame, models: ActionModels) -> pd.DataFrame:
    """Score every pitch under *both* actions.

    Not used by v1, but the counterfactual pair is what every later variant
    needs, and it costs one extra pass.
    """
    df = df.copy()
    df['q_take'] = models.predict(df, 'take')
    df['q_swing'] = models.predict(df, 'swing')
    df['edge'] = df['q_swing'] - df['q_take']
    return df
