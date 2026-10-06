"""Grade model predictions against the closing spread and simulate a flat-stake betting strategy."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import DEFAULT_ODDS, EDGE_THRESHOLDS

BREAKEVEN_110 = 110 / 210  # 52.38%


def american_to_profit(odds: pd.Series) -> pd.Series:
    """Profit per 1 unit staked for a winning bet at the given American odds."""
    odds = odds.fillna(DEFAULT_ODDS).astype(float)
    return np.where(odds < 0, 100.0 / odds.abs(), odds / 100.0)


def grade_bets(preds: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Bet home when model - spread > threshold, away when < -threshold. Flat 1 unit."""
    df = preds[preds["result"].notna() & preds["spread_line"].notna()].copy()
    df["edge"] = df["pred"] - df["spread_line"]
    df["cover_margin"] = df["result"] - df["spread_line"]  # >0 home covers

    df["side"] = np.where(
        df["edge"] > threshold, "home", np.where(df["edge"] < -threshold, "away", None)
    )
    df = df[df["side"].notna()].copy()

    signed = np.where(df["side"] == "home", df["cover_margin"], -df["cover_margin"])
    df["outcome"] = np.where(signed > 0, "win", np.where(signed < 0, "loss", "push"))

    odds = np.where(df["side"] == "home", df["home_spread_odds"], df["away_spread_odds"])
    profit_if_win = american_to_profit(pd.Series(odds, index=df.index))
    df["pnl"] = np.select(
        [df["outcome"] == "win", df["outcome"] == "loss"], [profit_if_win, -1.0], default=0.0
    )
    return df


def summarize(bets: pd.DataFrame) -> dict:
    n = len(bets)
    wins = int((bets["outcome"] == "win").sum())
    losses = int((bets["outcome"] == "loss").sum())
    pushes = int((bets["outcome"] == "push").sum())
    decided = wins + losses
    win_rate = wins / decided if decided else float("nan")
    roi = bets["pnl"].sum() / n if n else float("nan")
    # One-sided normal approximation: P(win rate this high | true rate = breakeven)
    if decided:
        se = math.sqrt(BREAKEVEN_110 * (1 - BREAKEVEN_110) / decided)
        z = (win_rate - BREAKEVEN_110) / se
        p = 0.5 * math.erfc(z / math.sqrt(2))
    else:
        z, p = float("nan"), float("nan")
    return {
        "bets": n,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate": win_rate,
        "roi": roi,
        "units": bets["pnl"].sum(),
        "z": z,
        "p_value": p,
    }


def accuracy_table(preds: pd.DataFrame) -> pd.DataFrame:
    """Per-season MAE of model vs. closing line, plus straight-up accuracy."""
    df = preds[preds["result"].notna()].copy()
    df["model_ae"] = (df["pred"] - df["result"]).abs()
    df["vegas_ae"] = (df["spread_line"] - df["result"]).abs()
    df["model_su"] = np.sign(df["pred"]) == np.sign(df["result"])
    df["vegas_su"] = np.sign(df["spread_line"]) == np.sign(df["result"])
    g = df.groupby("season").agg(
        games=("game_id", "size"),
        model_mae=("model_ae", "mean"),
        vegas_mae=("vegas_ae", "mean"),
        model_su=("model_su", "mean"),
        vegas_su=("vegas_su", "mean"),
    )
    total = pd.DataFrame(
        {
            "games": [len(df)],
            "model_mae": [df["model_ae"].mean()],
            "vegas_mae": [df["vegas_ae"].mean()],
            "model_su": [df["model_su"].mean()],
            "vegas_su": [df["vegas_su"].mean()],
        },
        index=["ALL"],
    )
    return pd.concat([g, total])


def threshold_table(preds: pd.DataFrame, thresholds=EDGE_THRESHOLDS) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        s = summarize(grade_bets(preds, t))
        s["threshold"] = t
        rows.append(s)
    return pd.DataFrame(rows).set_index("threshold")


def season_table(preds: pd.DataFrame, threshold: float) -> pd.DataFrame:
    bets = grade_bets(preds, threshold)
    rows = []
    for season, grp in bets.groupby("season"):
        s = summarize(grp)
        s["season"] = season
        rows.append(s)
    return pd.DataFrame(rows).set_index("season")[
        ["bets", "wins", "losses", "pushes", "win_rate", "roi", "units"]
    ]


def edge_calibration(preds: pd.DataFrame, bins=(0, 1, 2, 3, 4, 6, 100)) -> pd.DataFrame:
    """Does a bigger model edge translate to a higher cover rate? Buckets by |edge|."""
    bets = grade_bets(preds, 0.0)
    bets["abs_edge"] = bets["edge"].abs()
    bets["bucket"] = pd.cut(bets["abs_edge"], bins=bins, right=False)
    g = (
        bets[bets["outcome"] != "push"]
        .groupby("bucket", observed=True)
        .agg(
            bets=("outcome", "size"),
            cover_rate=("outcome", lambda s: (s == "win").mean()),
            roi=("pnl", "mean"),
        )
    )
    return g
