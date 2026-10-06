"""Feature engineering.

Pipeline:
  pbp + schedules -> team_game_stats (one row per team per game, post-game box score)
                  -> team_form (EWMA of each stat, shifted one game so it is strictly pre-game)
  schedules       -> elo (pre-game ratings)
  all of the above -> game-level feature matrix (home minus away diffs + context)

Every feature attached to a game uses only information available before kickoff.
"""

from __future__ import annotations

import math

import polars as pl

from .config import (
    DATA_PROCESSED,
    ELO_HOME_ADV,
    ELO_INIT,
    ELO_K,
    ELO_POINTS_PER_PT,
    ELO_SEASON_REGRESS,
    EWMA_HALFLIVES,
)

# Per-team-per-game stats computed from play-by-play (offense perspective).
OFF_STATS = [
    "epa_play",
    "success",
    "pass_epa",
    "rush_epa",
    "pass_rate",
    "explosive",
    "turnover_rate",
    "sack_rate",
    "cpoe",
    "pass_oe",
    "plays",
]
# Stats that get a defensive mirror (how the team's defense fared against opponents).
DEF_STATS = [
    "epa_play",
    "success",
    "pass_epa",
    "rush_epa",
    "explosive",
    "turnover_rate",
    "sack_rate",
]
# Game-outcome stats taken from the schedule.
RESULT_STATS = ["margin", "points_for", "points_against", "st_epa"]

CONTEXT_FEATURES = [
    "rest_diff",
    "div_game",
    "neutral",
    "dome",
    "week",
    "playoff",
    "home_games_played",
    "elo_spread",
]


# --------------------------------------------------------------------------- #
# Team-game box scores
# --------------------------------------------------------------------------- #
def team_game_stats(pbp: pl.DataFrame, schedules: pl.DataFrame) -> pl.DataFrame:
    plays = pbp.filter(
        ((pl.col("pass") == 1) | (pl.col("rush") == 1)),
        pl.col("epa").is_not_null(),
        pl.col("posteam").is_not_null(),
        pl.col("two_point_attempt") != 1,
    )

    agg = [
        pl.len().alias("plays"),
        pl.col("epa").mean().alias("epa_play"),
        pl.col("success").mean().alias("success"),
        pl.col("epa").filter(pl.col("pass") == 1).mean().alias("pass_epa"),
        pl.col("epa").filter(pl.col("rush") == 1).mean().alias("rush_epa"),
        pl.col("pass").mean().alias("pass_rate"),
        (pl.col("yards_gained") >= 15).mean().alias("explosive"),
        ((pl.col("interception") + pl.col("fumble_lost")).sum() / pl.len()).alias("turnover_rate"),
        pl.col("sack").filter(pl.col("pass") == 1).mean().alias("sack_rate"),
        pl.col("cpoe").mean().alias("cpoe"),
        pl.col("pass_oe").mean().alias("pass_oe"),
    ]

    off = plays.group_by("game_id", "posteam").agg(agg).rename({"posteam": "team"})
    deff = (
        plays.group_by("game_id", "defteam")
        .agg(agg)
        .rename({"defteam": "team"})
        .select(["game_id", "team"] + DEF_STATS)
        .rename({s: f"def_{s}" for s in DEF_STATS})
    )

    st = (
        pbp.filter(
            pl.col("special") == 1, pl.col("epa").is_not_null(), pl.col("posteam").is_not_null()
        )
        .group_by("game_id", "posteam")
        .agg(pl.col("epa").sum().alias("st_epa"))
        .rename({"posteam": "team"})
    )

    # Long schedule: one row per team per game with points and context.
    base = schedules.select(
        "game_id",
        "season",
        "week",
        "game_type",
        "gameday",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
    )
    home = base.select(
        "game_id",
        "season",
        "week",
        "game_type",
        "gameday",
        pl.col("home_team").alias("team"),
        pl.col("away_team").alias("opponent"),
        pl.col("home_score").alias("points_for"),
        pl.col("away_score").alias("points_against"),
        pl.lit(1).alias("is_home"),
    )
    away = base.select(
        "game_id",
        "season",
        "week",
        "game_type",
        "gameday",
        pl.col("away_team").alias("team"),
        pl.col("home_team").alias("opponent"),
        pl.col("away_score").alias("points_for"),
        pl.col("home_score").alias("points_against"),
        pl.lit(0).alias("is_home"),
    )
    long = pl.concat([home, away]).with_columns(
        (pl.col("points_for") - pl.col("points_against")).alias("margin")
    )

    out = (
        long.join(off, on=["game_id", "team"], how="left")
        .join(deff, on=["game_id", "team"], how="left")
        .join(st, on=["game_id", "team"], how="left")
        .sort("team", "gameday", "game_id")
    )
    return out


# --------------------------------------------------------------------------- #
# Pre-game form features (EWMA, shifted)
# --------------------------------------------------------------------------- #
def form_stat_columns() -> list[str]:
    return OFF_STATS + [f"def_{s}" for s in DEF_STATS] + RESULT_STATS


def team_form(team_games: pl.DataFrame) -> pl.DataFrame:
    """Shifted EWMA of every box-score stat. Row for game g uses games strictly before g."""
    stats = form_stat_columns()
    tg = team_games.sort("team", "gameday", "game_id")

    exprs = []
    for hl in EWMA_HALFLIVES:
        for s in stats:
            exprs.append(
                pl.col(s)
                .ewm_mean(half_life=hl, ignore_nulls=True, adjust=True)
                .shift(1)
                .over("team")
                .alias(f"{s}_ewm{hl}")
            )
    exprs.append(pl.int_range(pl.len()).over("team", "season").alias("games_played"))

    return tg.with_columns(exprs).select(
        ["game_id", "team", "season", "gameday", "games_played"]
        + [f"{s}_ewm{hl}" for hl in EWMA_HALFLIVES for s in stats]
    )


# --------------------------------------------------------------------------- #
# Elo
# --------------------------------------------------------------------------- #
def elo_ratings(schedules: pl.DataFrame) -> pl.DataFrame:
    """Sequential Elo. Returns pre-game ratings per game (available for unplayed games too)."""
    return _run_elo(schedules)[0]


def elo_current(schedules: pl.DataFrame) -> dict[str, float]:
    """Rating of every team after the last completed game in `schedules`."""
    return _run_elo(schedules)[1]


def _run_elo(schedules: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, float]]:
    games = (
        schedules.sort("gameday", "game_id")
        .select("game_id", "season", "home_team", "away_team", "result", "location")
        .to_dicts()
    )

    ratings: dict[str, float] = {}
    last_season: dict[str, int] = {}
    rows = []

    for g in games:
        h, a, season = g["home_team"], g["away_team"], g["season"]
        for t in (h, a):
            if t not in ratings:
                ratings[t] = ELO_INIT
                last_season[t] = season
            elif last_season[t] != season:
                ratings[t] = ratings[t] + ELO_SEASON_REGRESS * (ELO_INIT - ratings[t])
                last_season[t] = season

        hfa = 0.0 if g["location"] == "Neutral" else ELO_HOME_ADV
        h_pre, a_pre = ratings[h], ratings[a]
        diff = h_pre + hfa - a_pre
        rows.append(
            {
                "game_id": g["game_id"],
                "home_elo": h_pre,
                "away_elo": a_pre,
                "elo_spread": diff / ELO_POINTS_PER_PT,
            }
        )

        result = g["result"]
        if result is None:
            continue
        expected_home = 1.0 / (1.0 + 10 ** (-diff / 400.0))
        actual_home = 1.0 if result > 0 else 0.0 if result < 0 else 0.5
        winner_diff = diff if result >= 0 else -diff
        mov_mult = math.log(abs(result) + 1.0) * (2.2 / (winner_diff * 0.001 + 2.2))
        delta = ELO_K * mov_mult * (actual_home - expected_home)
        ratings[h] = h_pre + delta
        ratings[a] = a_pre - delta

    return pl.DataFrame(rows), ratings


# --------------------------------------------------------------------------- #
# Game-level matrix
# --------------------------------------------------------------------------- #
def build_game_features(
    schedules: pl.DataFrame, form: pl.DataFrame, elo: pl.DataFrame
) -> pl.DataFrame:
    stat_cols = [c for c in form.columns if c.endswith(tuple(f"_ewm{h}" for h in EWMA_HALFLIVES))]

    home_form = form.select(
        pl.col("game_id"),
        pl.col("team").alias("home_team"),
        pl.col("games_played").alias("home_games_played"),
        *[pl.col(c).alias(f"home_{c}") for c in stat_cols],
    )
    away_form = form.select(
        pl.col("game_id"),
        pl.col("team").alias("away_team"),
        pl.col("games_played").alias("away_games_played"),
        *[pl.col(c).alias(f"away_{c}") for c in stat_cols],
    )

    df = (
        schedules.join(home_form, on=["game_id", "home_team"], how="left")
        .join(away_form, on=["game_id", "away_team"], how="left")
        .join(elo, on="game_id", how="left")
    )

    df = df.with_columns(
        [(pl.col(f"home_{c}") - pl.col(f"away_{c}")).alias(f"diff_{c}") for c in stat_cols]
        + [
            (pl.col("home_rest") - pl.col("away_rest")).alias("rest_diff"),
            (pl.col("location") == "Neutral").cast(pl.Int8).alias("neutral"),
            pl.col("roof").is_in(["dome", "closed"]).cast(pl.Int8).alias("dome"),
            (pl.col("game_type") != "REG").cast(pl.Int8).alias("playoff"),
        ]
    )
    return df.sort("gameday", "game_id")


def feature_columns(df: pl.DataFrame, include_market: bool = False) -> list[str]:
    cols = [c for c in df.columns if c.startswith("diff_")] + CONTEXT_FEATURES
    if include_market:
        cols = cols + ["spread_line"]
    return cols


def features_path():
    return DATA_PROCESSED / "game_features.parquet"


def build_and_save(pbp: pl.DataFrame, schedules: pl.DataFrame) -> pl.DataFrame:
    tg = team_game_stats(pbp, schedules)
    form = team_form(tg)
    elo = elo_ratings(schedules)
    df = build_game_features(schedules, form, elo)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    tg.write_parquet(DATA_PROCESSED / "team_games.parquet")
    df.write_parquet(features_path())
    return df


def load_features() -> pl.DataFrame:
    return pl.read_parquet(features_path())
