from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

# Seasons of play-by-play to ingest. Earlier seasons warm up EWMA/Elo state.
FIRST_SEASON = 2006
# First season evaluated in the walk-forward backtest.
FIRST_TEST_SEASON = 2012

# Franchise relocations: map old abbreviations onto current ones so team history is continuous.
TEAM_ALIASES = {"OAK": "LV", "SD": "LAC", "STL": "LA"}

# Elo parameters (FiveThirtyEight-style).
ELO_INIT = 1500.0
ELO_K = 20.0
ELO_HOME_ADV = 48.0  # Elo points; ~2 points of spread at 25 Elo/pt
ELO_SEASON_REGRESS = 1 / 3  # regress toward mean each new season
ELO_POINTS_PER_PT = 25.0  # Elo-diff to point-spread conversion

# EWMA half-lives (in games) for team form features.
EWMA_HALFLIVES = (5, 12)

# Betting evaluation.
DEFAULT_ODDS = -110
EDGE_THRESHOLDS = (0.0, 1.0, 2.0, 3.0, 4.0)
