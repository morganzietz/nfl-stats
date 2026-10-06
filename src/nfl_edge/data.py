"""Ingest nflverse schedules and play-by-play into local parquet files."""

from __future__ import annotations

import nflreadpy as nfl
import polars as pl

from .config import DATA_RAW, FIRST_SEASON, TEAM_ALIASES

PBP_COLUMNS = [
    "game_id",
    "play_id",
    "season",
    "week",
    "game_date",
    "posteam",
    "defteam",
    "home_team",
    "away_team",
    "play_type",
    "epa",
    "success",
    "wpa",
    "pass",
    "rush",
    "qb_dropback",
    "yards_gained",
    "down",
    "ydstogo",
    "penalty",
    "special",
    "two_point_attempt",
    "touchdown",
    "interception",
    "fumble_lost",
    "sack",
    "air_yards",
    "cpoe",
    "pass_oe",
    "series_success",
    "qb_epa",
    "qb_scramble",
]

SCHEDULE_COLUMNS = [
    "game_id",
    "season",
    "game_type",
    "week",
    "gameday",
    "weekday",
    "gametime",
    "away_team",
    "home_team",
    "away_score",
    "home_score",
    "result",
    "total",
    "location",
    "overtime",
    "away_rest",
    "home_rest",
    "away_moneyline",
    "home_moneyline",
    "spread_line",
    "away_spread_odds",
    "home_spread_odds",
    "total_line",
    "under_odds",
    "over_odds",
    "div_game",
    "roof",
    "surface",
    "temp",
    "wind",
    "away_qb_id",
    "home_qb_id",
    "away_qb_name",
    "home_qb_name",
    "away_coach",
    "home_coach",
]


def _alias(col: str) -> pl.Expr:
    return pl.col(col).replace(TEAM_ALIASES)


def schedules_path():
    return DATA_RAW / "schedules.parquet"


def pbp_path():
    return DATA_RAW / "pbp.parquet"


def ingest_schedules() -> pl.DataFrame:
    df = nfl.load_schedules().select(SCHEDULE_COLUMNS)
    df = df.with_columns(
        _alias("home_team").alias("home_team"),
        _alias("away_team").alias("away_team"),
        pl.col("gameday").str.to_date().alias("gameday"),
    ).filter(pl.col("season") >= FIRST_SEASON)
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    df.write_parquet(schedules_path())
    return df


def ingest_pbp(seasons: list[int] | None = None) -> pl.DataFrame:
    if seasons is None:
        latest = int(nfl.load_schedules()["season"].max())
        seasons = list(range(FIRST_SEASON, latest + 1))
    frames = []
    for s in seasons:
        df = nfl.load_pbp([s])
        cols = [c for c in PBP_COLUMNS if c in df.columns]
        frames.append(df.select(cols))
    pbp = pl.concat(frames, how="diagonal_relaxed")
    pbp = pbp.with_columns(
        _alias("posteam").alias("posteam"),
        _alias("defteam").alias("defteam"),
        _alias("home_team").alias("home_team"),
        _alias("away_team").alias("away_team"),
    )
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    pbp.write_parquet(pbp_path())
    return pbp


def load_schedules() -> pl.DataFrame:
    return pl.read_parquet(schedules_path())


def load_pbp() -> pl.DataFrame:
    return pl.read_parquet(pbp_path())
