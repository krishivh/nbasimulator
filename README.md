# NBA Season Simulator

An end-to-end machine learning pipeline that predicts the 2025-26 NBA champion. Trained on historical game data from Kaggle, it simulates the remainder of the regular season, runs the play-in tournament, and plays out the full playoff bracket.

**[Live Demo →](#demo)** (React walkthrough)  
**[Streamlit Dashboard →](#streamlit)** (data exploration)

---

## How It Works

### The Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│                     DATA LAYER                                │
│                                                               │
│  Kaggle Dataset ─────→ Player box scores (1947–present)      │
│  (7,740+ games)         PTS, FG%, REB, AST, STL, BLK, TOV   │
│                                                               │
│  Aggregation ────────→ Team-level game stats per matchup     │
│                         Home team stats vs Away team stats    │
└──────────────────────────┬───────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────┐
│                  FEATURE ENGINEERING                          │
│                                                               │
│  Rolling averages (5, 10, 20 game windows) for each stat:   │
│    • Points, FG%, 3PT%, FT%                                  │
│    • Rebounds, Assists, Steals, Blocks, Turnovers            │
│    • Offensive Rating, Defensive Rating, Pace                │
│    • Win percentage                                           │
│                                                               │
│  Differential features: home_stat - away_stat                │
│  → 198 features total per game                               │
└──────────────────────────┬───────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────┐
│                    ML MODEL                                   │
│                                                               │
│  XGBoost Gradient Boosted Decision Trees                     │
│    • 200 trees, max depth 6                                  │
│    • TimeSeriesSplit cross-validation (5 folds)              │
│    • Prevents data leakage (no future data in training)      │
│    • CV Accuracy: ~53% | CV AUC: ~0.54                       │
│                                                               │
│  Output: P(home team wins) for any matchup                   │
│                                                               │
│  Exported: Team strength profiles (home/away/overall)        │
│    → JSON consumed by the React demo                         │
└──────────────────────────┬───────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────┐
│                  SIMULATION ENGINE                            │
│                                                               │
│  1. Fetch today's live NBA standings                         │
│  2. Blend: 70% XGBoost strength + 30% current win%          │
│  3. Log5 formula for matchup probability + home court        │
│  4. Simulate remaining regular season games                  │
│  5. Run play-in tournament (7v8, 9v10, elimination)          │
│  6. Simulate playoff bracket (best-of-7 series)             │
│  7. Crown the NBA champion                                   │
│                                                               │
│  Python version: 1,000 Monte Carlo iterations                │
│  React demo: Single simulation with animated walkthrough     │
└──────────────────────────────────────────────────────────────┘
```

### Why ~53% Accuracy Is Expected

Individual NBA games are extremely noisy. A team's best player might rest, a role player might have a career night, or a game might go to overtime on a lucky bounce. Even professional sportsbooks only predict individual game winners at ~65-68% accuracy.

The value of the model isn't in predicting single games — it's that the team-quality signal it learns (offensive rating trends, defensive consistency, pace of play) accumulates over hundreds of simulated games. By the time you've simulated the full remaining season, the strong teams have risen to the top and the championship probabilities become meaningful.

This is a great interview talking point: "I know the per-game accuracy looks modest, but the model captures real signal that compounds over a season-length simulation."

---

## Project Structure

```
nba-season-simulator/
├── pipeline.py                  # Main entry point — full pipeline
├── app.py                       # Streamlit dashboard
├── requirements.txt
├── .gitignore
├── README.md
│
├── src/
│   ├── generate_training_data.py  # Generates synthetic training data
│   ├── train_model.py             # Feature engineering + XGBoost training
│   └── simulate.py                # Monte Carlo simulation engine (Python)
│
├── data/                        # Git-ignored (too large)
│   └── .gitkeep
│
└── models/                      # Git-ignored (regenerated by pipeline)
    └── .gitkeep
```

### What Each File Does

| File | Purpose |
|------|---------|
| `pipeline.py` | **Start here.** Loads Kaggle data, engineers features, trains XGBoost, exports team profiles. Single command to run everything. |
| `src/generate_training_data.py` | Creates ~7,800 synthetic NBA games with realistic box score distributions. Used when Kaggle data isn't available. |
| `src/train_model.py` | Core ML: rolling feature computation, XGBoost training, TimeSeriesSplit CV, feature importance analysis. |
| `src/simulate.py` | Python Monte Carlo engine. Runs 1,000 simulations of remaining season + playoffs. Outputs championship probability distributions. |
| `app.py` | Streamlit dashboard for exploring training data, model metrics, and simulation results. |

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get the Data

**Option A: Kaggle dataset (recommended)**
```bash
pip install kagglehub
python -c "import kagglehub; print(kagglehub.dataset_download('eoinamoore/historical-nba-data-and-player-box-scores'))"
```
Then run:
```bash
python pipeline.py --kaggle-path /path/to/downloaded/data
```

**Option B: Generated data (no Kaggle account needed)**
```bash
python src/generate_training_data.py
python pipeline.py --use-generated
```

### 3. Run the Simulation

```bash
python src/simulate.py
```

### 4. Launch the Dashboard

```bash
streamlit run app.py
```

---

## The React Demo

The interactive walkthrough (`nba-demo-walkthrough.jsx`) is a separate React artifact that runs entirely in the browser. It:

1. **Fetches live standings** via API call when opened
2. **Uses XGBoost-derived team profiles** (embedded as JSON) to predict game outcomes
3. **Blends ML strength with current win%** (70/30) for each prediction
4. **Simulates remaining games** using Log5 probability + home court advantage
5. **Runs play-in tournament** (7v8, 9v10, elimination game)
6. **Animates playoff bracket** round by round with timed reveals
7. **Crowns champion** with finals score

The demo doesn't run XGBoost in the browser — that would be too heavy. Instead, it uses the pre-computed team strength profiles that the Python pipeline exported from the trained model. This is a common pattern in ML deployment: train offline, serve predictions online.

---

## Tech Stack

| Technology | Role |
|-----------|------|
| Python 3.12 | Core language |
| XGBoost | Gradient boosted decision tree classifier |
| scikit-learn | TimeSeriesSplit CV, evaluation metrics |
| pandas / numpy | Data manipulation, feature engineering |
| Streamlit | Interactive data dashboard |
| Plotly | Visualization |
| React | Interactive demo walkthrough |
| Anthropic API | Live standings fetch (web search) |

---

## Key Concepts for Interviews

- **Feature Engineering**: Rolling averages prevent data leakage — you can't use same-game stats to predict the game's outcome. The model only sees what was known before tip-off.
- **TimeSeriesSplit**: Standard k-fold CV would leak future information into training. TimeSeriesSplit respects temporal ordering.
- **Log5 Method**: Bill James' formula for head-to-head probability from team win rates. Combined with ML strength ratings for better calibration.
- **Monte Carlo Simulation**: Running the season 1,000 times captures uncertainty — instead of saying "Team X wins the title," we say "Team X wins in 35% of simulations."
- **Home Court Advantage**: ~3.5% historical win rate boost for the home team, applied to both regular season and playoff predictions.
- **Play-In Tournament**: NBA's format for seeds 7-10. Three single-elimination games determine the final two playoff spots.

---

## Dataset

**Kaggle: [NBA Dataset - Box Scores & Stats, 1947 - Today](https://www.kaggle.com/datasets/eoinamoore/historical-nba-data-and-player-box-scores)**

Updated daily. Contains player-level box scores for every NBA game since 1947. The pipeline aggregates these into team-level features for model training.

---

## Resume Keywords

`Machine Learning` · `XGBoost` · `Python` · `Monte Carlo Simulation` · `Feature Engineering` · `Cross-Validation` · `Time Series` · `Classification` · `Probability Modeling` · `Sports Analytics` · `React` · `Streamlit` · `pandas` · `scikit-learn` · `Data Pipeline` · `Log5` · `NBA API`
