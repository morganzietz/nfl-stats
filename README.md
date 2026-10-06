# nfl-edge

An NFL point-spread model benchmarked honestly against the closing Vegas line.

The question this project answers is the hard one: **can a model built from public
play-by-play data beat the market?** Every prediction is made walk-forward (train on prior
seasons only, predict the next season), every feature is strictly pre-game, and every pick is
graded against the actual closing spread and spread odds.

## Results (walk-forward, 2012 through 2026 week 4, 3,892 games)

| Model | Features | MAE vs result | Vegas MAE | ATS win rate (edge > 3) | ROI |
|---|---|---|---|---|---|
| Ridge | team form + Elo + context | 10.31 | 9.97 | 52.3% (n=1,334) | +1.6% |
| LightGBM | same | 10.36 | 9.97 | 51.0% (n=1,362) | -1.0% |
| Ridge | same + closing spread | 10.04 | 9.97 | 52.0% (n=2,144) | +1.3% |

Breakeven at standard -110 odds is 52.4%. None of these edges are statistically distinguishable
from zero (one-sided p > 0.5), which is exactly what an efficient market predicts. The pure model
gets within ~0.35 points of the closing line's accuracy using nothing but public data, and the
market-aware model essentially learns to trust the line. See "What I'd do next" for where real
edge might hide.

## How it works

```
nflverse schedules + play-by-play
        │
        ▼
team_game_stats      one row per team per game: EPA/play, success rate, pass/rush EPA,
                     explosive rate, turnover rate, sack rate, CPOE, pass rate over expected,
                     and the defensive mirror of each
        │
        ▼
team_form            exponentially weighted moving average of each stat (half-lives of 5
                     and 12 games), shifted one game so the row for game g only sees games
                     strictly before g
        │
elo_ratings          sequential Elo with margin-of-victory multiplier, 1/3 regression to
                     the mean each offseason, home-field advantage zeroed on neutral sites
        │
        ▼
game features        home-minus-away differences of every form stat, plus rest differential,
                     divisional flag, dome, neutral site, week, playoff flag, games played
                     this season, and the Elo-implied spread. 52 features.
        │
        ▼
walk-forward         for each season s >= 2012: fit on all completed games with season < s,
                     predict season s. Ridge (alpha=30) or LightGBM (shallow, heavily
                     regularized).
        │
        ▼
backtest             edge = model - closing spread. Bet the side the model favors when |edge|
                     exceeds a threshold. Grade with actual home/away spread odds. Report MAE,
                     ATS record, ROI, a z-test against breakeven, and cover rate by edge bucket.
```

### Leakage controls

- Form features are computed with `.ewm_mean().shift(1)` partitioned by team. A test perturbs a
  future game's result by 1,000 points and asserts no earlier feature moves.
- Elo is updated only after a game's result is consumed; unplayed games receive pre-game ratings.
- Training data for season s never includes any game from season s, including earlier weeks.
  (A weekly-refit variant is a planned improvement.)
- The spread is excluded from features unless `--market` is passed, so the pure model's skill
  is measured independently of the market.

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). On macOS, LightGBM also needs
`brew install libomp`.

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
```

## Usage

```bash
nfl-edge ingest                              # download 2006-present schedules and pbp (~30s)
nfl-edge features                            # build the game feature matrix (~2s)
nfl-edge backtest --model ridge              # walk-forward backtest, pure features
nfl-edge backtest --model lgbm --market      # LightGBM with the closing line as a feature
nfl-edge predict --season 2026 --week 5      # picks for a week, fit on everything before it
pytest                                       # leakage and grading tests
```

Backtest predictions are written to `reports/preds_<model>_<features>.csv` for further analysis.

## Data

All data comes from [nflverse](https://github.com/nflverse) via `nflreadpy`. Schedules include
closing spread, total, moneyline, and spread odds for every game since 1999. Play-by-play includes
EPA, success, CPOE, and pass-rate-over-expected. Franchise moves (OAK to LV, SD to LAC, STL to LA)
are mapped so team history is continuous.

## What I'd do next

In rough order of expected value:

1. **Quarterback adjustments.** The single biggest reason a form model lags the market is that
   the market knows who is starting at QB and the EWMA doesn't. nflverse ships `home_qb_id` per
   game; a QB-level EPA prior swapped in on injury would close much of the gap.
2. **Weekly refit.** Currently season s is predicted from seasons < s. Refitting each week adds
   the current season's early games to training.
3. **Opening lines and closing line value.** nflverse has only closing lines. Pulling opening
   lines from The Odds API would let the model be judged on CLV, the standard professional metric.
4. **Totals model.** The same pipeline predicts points scored; the totals market is thinner.
5. **Deploy.** Ship the Next.js site and API so the picks board is public.
