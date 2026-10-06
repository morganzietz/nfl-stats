"""Command line interface: nfl-edge ingest | features | backtest | predict."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import typer
from rich.console import Console
from rich.table import Table

from . import backtest as bt
from . import data
from . import features as feat
from .config import FIRST_TEST_SEASON, REPORTS_DIR
from .model import fit_full, walk_forward

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


def _print_df(df: pd.DataFrame, title: str, fmt: dict | None = None):
    fmt = fmt or {}
    table = Table(title=title, show_lines=False)
    table.add_column(df.index.name or "")
    for c in df.columns:
        table.add_column(str(c), justify="right")
    for i, idx in enumerate(df.index):
        cells = []
        for c in df.columns:
            v = df[c].iat[i]  # .iat keeps the column dtype; iterrows would upcast ints to float
            if c in fmt:
                cells.append(fmt[c].format(v))
            elif isinstance(v, (float, np.floating)):
                cells.append(f"{v:.3f}")
            else:
                cells.append(str(v))
        table.add_row(str(idx), *cells)
    console.print(table)


@app.command()
def ingest():
    """Download schedules and play-by-play from nflverse into data/raw."""
    s = data.ingest_schedules()
    console.print(f"schedules: {s.shape[0]} games, seasons {s['season'].min()}-{s['season'].max()}")
    p = data.ingest_pbp()
    console.print(f"pbp: {p.shape[0]:,} plays")


@app.command()
def features():
    """Build the game-level feature matrix into data/processed."""
    df = feat.build_and_save(data.load_pbp(), data.load_schedules())
    n_feat = len(feat.feature_columns(df))
    console.print(f"features: {df.shape[0]} games x {n_feat} features -> {feat.features_path()}")


@app.command()
def backtest(
    model: str = typer.Option("ridge", help="ridge | lgbm"),
    market: bool = typer.Option(False, help="Include the closing spread as a feature."),
    first_season: int = typer.Option(FIRST_TEST_SEASON),
    threshold: float = typer.Option(2.0, help="Edge threshold (points) for the per-season table."),
    min_games: int = typer.Option(
        0, help="Skip bets when either team has fewer games this season."
    ),
):
    """Walk-forward backtest: train on prior seasons, predict each season, grade vs the spread."""
    df = feat.load_features()
    cols = feat.feature_columns(df, include_market=market)
    preds = walk_forward(df, cols, model, first_season)
    preds = preds[preds["min_games_played"] >= min_games]

    tag = f"{model}_{'market' if market else 'pure'}"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    preds.to_csv(REPORTS_DIR / f"preds_{tag}.csv", index=False)

    pct = {
        "model_su": "{:.1%}",
        "vegas_su": "{:.1%}",
        "win_rate": "{:.1%}",
        "roi": "{:+.1%}",
        "cover_rate": "{:.1%}",
        "p_value": "{:.3f}",
        "units": "{:+.1f}",
    }
    _print_df(bt.accuracy_table(preds), f"[{tag}] MAE vs closing line", pct)
    _print_df(bt.threshold_table(preds), f"[{tag}] Flat-stake ATS by edge threshold", pct)
    _print_df(bt.season_table(preds, threshold), f"[{tag}] By season, threshold={threshold}", pct)
    _print_df(bt.edge_calibration(preds), f"[{tag}] Cover rate by |edge| bucket", pct)
    console.print(f"predictions saved to reports/preds_{tag}.csv")


@app.command()
def predict(
    season: int = typer.Option(..., help="Season to predict."),
    week: int = typer.Option(..., help="Week to predict."),
    model: str = typer.Option("ridge"),
    market: bool = typer.Option(False),
):
    """Fit on all completed games and print picks for one week."""
    df = feat.load_features()
    cols = feat.feature_columns(df, include_market=market)
    train = df.filter(~((pl.col("season") == season) & (pl.col("week") >= week)))
    m = fit_full(train, cols, model)
    target = df.filter(pl.col("season") == season, pl.col("week") == week).to_pandas()
    target["pred"] = m.predict(target[cols])
    target["edge"] = target["pred"] - target["spread_line"]

    def _pick(r):
        if pd.isna(r.spread_line):
            return "no line"
        if r.edge > 0:
            return f"{r.home_team} {-r.spread_line:+.1f}"
        return f"{r.away_team} {r.spread_line:+.1f}"

    target["pick"] = target.apply(_pick, axis=1)
    target["gameday"] = pd.to_datetime(target["gameday"]).dt.strftime("%Y-%m-%d")
    out = target[
        ["gameday", "away_team", "home_team", "spread_line", "pred", "edge", "pick", "result"]
    ]
    out = out.sort_values("edge", key=lambda s: s.abs().fillna(-1), ascending=False).set_index(
        "gameday"
    )
    _print_df(
        out,
        f"{season} week {week} picks ({model}, {'market' if market else 'pure'})",
        {"spread_line": "{:+.1f}", "pred": "{:+.1f}", "edge": "{:+.1f}"},
    )


if __name__ == "__main__":
    app()
