"""Leakage and correctness tests for feature construction."""

import polars as pl

from nfl_edge import features as f


def _fake_team_games(schedule: pl.DataFrame) -> pl.DataFrame:
    """Minimal team-game table with a single stat we control."""
    home = schedule.select(
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
    away = schedule.select(
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
    tg = pl.concat([home, away]).with_columns(
        (pl.col("points_for") - pl.col("points_against")).alias("margin")
    )
    for s in f.form_stat_columns():
        if s not in tg.columns:
            tg = tg.with_columns(pl.col("margin").cast(pl.Float64).alias(s))
    return tg


def test_form_is_strictly_pregame(tiny_schedule):
    tg = _fake_team_games(tiny_schedule)
    form = f.team_form(tg)

    # Team A's first game: no history -> null form.
    a1 = form.filter(pl.col("team") == "A", pl.col("game_id") == "2020_01_A_B")
    assert a1["margin_ewm5"][0] is None

    # Team A's second game: form equals its week-1 margin exactly (one observation).
    a2 = form.filter(pl.col("team") == "A", pl.col("game_id") == "2020_02_D_A")
    assert a2["margin_ewm5"][0] == -7.0  # A lost 17-24 away in week 1


def test_future_results_do_not_change_past_features(tiny_schedule):
    tg = _fake_team_games(tiny_schedule)
    base = f.team_form(tg)

    # Perturb the week-2 result massively and recompute: week-1 and week-2 features must not move.
    perturbed = tg.with_columns(
        pl.when(pl.col("game_id") == "2020_02_D_A")
        .then(pl.col("margin") + 1000)
        .otherwise(pl.col("margin"))
        .alias("margin")
    )
    alt = f.team_form(perturbed)
    for gid in ("2020_01_A_B", "2020_01_C_D", "2020_02_B_C", "2020_02_D_A"):
        b = base.filter(pl.col("game_id") == gid).sort("team")["margin_ewm5"].to_list()
        a = alt.filter(pl.col("game_id") == gid).sort("team")["margin_ewm5"].to_list()
        assert a == b, gid
    # ...but week-3 features for A must change.
    b3 = base.filter(pl.col("game_id") == "2020_03_A_C", pl.col("team") == "A")["margin_ewm5"][0]
    a3 = alt.filter(pl.col("game_id") == "2020_03_A_C", pl.col("team") == "A")["margin_ewm5"][0]
    assert a3 != b3


def test_games_played_resets_each_season(tiny_schedule):
    s2 = tiny_schedule.with_columns(
        pl.col("season") + 1,
        pl.col("gameday").dt.offset_by("1y"),
        (pl.col("game_id").str.replace("2020", "2021")),
    )
    tg = _fake_team_games(pl.concat([tiny_schedule, s2]))
    form = f.team_form(tg)
    a = form.filter(pl.col("team") == "A").sort("gameday")
    assert a["games_played"].to_list() == [0, 1, 2, 0, 1, 2]


def test_elo_uses_only_prior_games_and_neutral_site(tiny_schedule):
    elo = f.elo_ratings(tiny_schedule)
    first = elo.filter(pl.col("game_id") == "2020_01_A_B").row(0, named=True)
    assert first["home_elo"] == first["away_elo"] == 1500.0
    assert first["elo_spread"] > 0  # home-field advantage only

    # Neutral site: elo_spread is the raw rating gap, no HFA.
    neutral = elo.filter(pl.col("game_id") == "2020_02_D_A").row(0, named=True)
    assert abs(neutral["elo_spread"] - (neutral["home_elo"] - neutral["away_elo"]) / 25.0) < 1e-9

    # Unplayed game still receives pre-game ratings.
    assert elo.filter(pl.col("game_id") == "2020_03_A_C").height == 1

    # Winner's rating went up: B beat A in week 1 at home.
    wk2_b = elo.filter(pl.col("game_id") == "2020_02_B_C").row(0, named=True)
    assert wk2_b["away_elo"] > 1500.0


def test_game_matrix_context_flags(tiny_schedule):
    tg = _fake_team_games(tiny_schedule)
    df = f.build_game_features(tiny_schedule, f.team_form(tg), f.elo_ratings(tiny_schedule))
    row = df.filter(pl.col("game_id") == "2020_02_D_A").row(0, named=True)
    assert row["neutral"] == 1
    dome = df.filter(pl.col("game_id") == "2020_01_C_D").row(0, named=True)
    assert dome["dome"] == 1
    assert set(f.feature_columns(df)) >= set(f.CONTEXT_FEATURES)
    assert "spread_line" not in f.feature_columns(df)
    assert "spread_line" in f.feature_columns(df, include_market=True)
