"""Models that predict home margin of victory, trained walk-forward by season."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "result"


def make_model(kind: str):
    if kind == "ridge":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=30.0)),
            ]
        )
    if kind == "lgbm":
        return LGBMRegressor(
            n_estimators=500,
            learning_rate=0.015,
            num_leaves=7,
            min_child_samples=50,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_lambda=10.0,
            verbose=-1,
            random_state=42,
        )
    raise ValueError(f"unknown model kind: {kind}")


def walk_forward(
    df: pl.DataFrame,
    feature_cols: list[str],
    model_kind: str,
    first_test_season: int,
    min_games_played: int = 0,
) -> pd.DataFrame:
    """Train on all completed games from seasons < s, predict season s. Repeat for every s.

    Rows with fewer than `min_games_played` games for either team in the current season are
    still predicted but can be filtered downstream (early-season form is noisy).
    """
    pdf = df.to_pandas()
    labeled = pdf[pdf[TARGET].notna()]
    seasons = sorted(s for s in pdf["season"].unique() if s >= first_test_season)

    outputs = []
    for s in seasons:
        train = labeled[labeled["season"] < s]
        test = pdf[pdf["season"] == s]
        if train.empty or test.empty:
            continue
        model = make_model(model_kind)
        model.fit(train[feature_cols], train[TARGET])
        preds = model.predict(test[feature_cols])
        out = test[
            [
                "game_id",
                "season",
                "week",
                "game_type",
                "gameday",
                "home_team",
                "away_team",
                "spread_line",
                "home_spread_odds",
                "away_spread_odds",
                TARGET,
                "home_games_played",
                "away_games_played",
            ]
        ].copy()
        out["pred"] = preds
        outputs.append(out)

    res = pd.concat(outputs, ignore_index=True)
    res["min_games_played"] = np.minimum(res["home_games_played"], res["away_games_played"])
    return res


def fit_full(df: pl.DataFrame, feature_cols: list[str], model_kind: str):
    pdf = df.to_pandas()
    labeled = pdf[pdf[TARGET].notna()]
    model = make_model(model_kind)
    model.fit(labeled[feature_cols], labeled[TARGET])
    return model
