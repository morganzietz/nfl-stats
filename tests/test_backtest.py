import numpy as np
import pandas as pd
import pytest

from nfl_edge import backtest as bt


def _preds():
    return pd.DataFrame(
        {
            "game_id": ["g1", "g2", "g3", "g4", "g5"],
            "season": [2020] * 5,
            "result": [10, -3, 0, 7, 1],
            "spread_line": [3.0, -3.0, 0.0, 7.0, -2.5],
            "pred": [7.0, -1.0, 4.0, 2.0, -2.0],  # edges: +4, +2, +4, -5, +0.5
            "home_spread_odds": [-110, -105, -110, np.nan, -110],
            "away_spread_odds": [-110, -115, -110, +100, -110],
        }
    )


def test_american_to_profit():
    out = bt.american_to_profit(pd.Series([-110, +150, np.nan, -200]))
    assert np.allclose(out, [100 / 110, 1.5, 100 / 110, 0.5])


def test_grade_bets_sides_and_outcomes():
    bets = bt.grade_bets(_preds(), threshold=1.0).set_index("game_id")
    # g1: edge +4 -> home; result 10 vs line 3 -> cover -> win at -110
    assert bets.loc["g1", "side"] == "home" and bets.loc["g1", "outcome"] == "win"
    assert bets.loc["g1", "pnl"] == pytest.approx(100 / 110)
    # g2: edge +2 -> home; result -3 vs line -3 -> push
    assert bets.loc["g2", "outcome"] == "push" and bets.loc["g2", "pnl"] == 0
    # g3: edge +4 -> home; result 0 vs line 0 -> push
    assert bets.loc["g3", "outcome"] == "push"
    # g4: edge -5 -> away at +100; result 7 vs line 7 -> push
    assert bets.loc["g4", "side"] == "away" and bets.loc["g4", "outcome"] == "push"
    # g5: edge +0.5 below threshold -> no bet
    assert "g5" not in bets.index


def test_grade_bets_loss_pnl():
    p = _preds()
    p.loc[0, "result"] = -20  # g1 home bet now loses
    bets = bt.grade_bets(p, threshold=1.0).set_index("game_id")
    assert bets.loc["g1", "outcome"] == "loss" and bets.loc["g1", "pnl"] == -1.0


def test_summarize_breakeven_math():
    bets = pd.DataFrame(
        {"outcome": ["win"] * 11 + ["loss"] * 10, "pnl": [100 / 110] * 11 + [-1.0] * 10}
    )
    s = bt.summarize(bets)
    assert s["bets"] == 21 and s["wins"] == 11 and s["losses"] == 10
    assert s["units"] == pytest.approx(0.0)
    assert s["roi"] == pytest.approx(0.0)


def test_accuracy_table_has_total_row():
    t = bt.accuracy_table(_preds())
    assert "ALL" in t.index and t.loc["ALL", "games"] == 5
