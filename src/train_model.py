"""
Feature Engineering & Model Training Pipeline

This module:
1. Loads raw game data
2. Engineers rolling/aggregate features per team
3. Trains an XGBoost classifier to predict game outcomes
4. Evaluates model performance with cross-validation
5. Saves the trained model for use in simulation
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier as XGBClassifier
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


def compute_rolling_stats(df, team_col, stat_cols, windows=[5, 10, 20]):
    """
    Compute rolling averages for a team's stats over recent games.
    This is the core feature engineering step - we create features that
    capture a team's recent form rather than using single-game stats
    (which we wouldn't have at prediction time).
    """
    features = {}

    for team in df[team_col].unique():
        team_home = df[df['home_team'] == team].copy()
        team_away = df[df['away_team'] == team].copy()

        # Combine into a single timeline for the team
        home_games = team_home[['date', 'home_win'] + [f'home_{s}' for s in stat_cols]].copy()
        home_games.columns = ['date', 'win'] + stat_cols
        home_games['is_home'] = 1

        away_games = team_away[['date', 'home_win'] + [f'away_{s}' for s in stat_cols]].copy()
        away_games.columns = ['date', 'win'] + stat_cols
        away_games['win'] = 1 - away_games['win']  # Flip for away team
        away_games['is_home'] = 0

        team_games = pd.concat([home_games, away_games]).sort_values('date').reset_index(drop=True)

        for window in windows:
            for stat in stat_cols + ['win']:
                col_name = f'{stat}_roll_{window}'
                team_games[col_name] = team_games[stat].rolling(window, min_periods=1).mean()

        features[team] = team_games

    return features


def build_features(df):
    """
    Build the feature matrix for ML training.

    Features include:
    - Rolling averages of key stats (5, 10, 20 game windows)
    - Win percentage differentials
    - Home/away splits
    - Offensive/defensive rating trends
    """
    stat_cols = ['pts', 'fg_pct', 'fg3_pct', 'ft_pct', 'reb', 'ast',
                 'stl', 'blk', 'tov', 'off_rating', 'def_rating', 'pace']

    print("Computing rolling statistics per team...")
    rolling_stats = compute_rolling_stats(df, 'home_team', stat_cols)

    feature_rows = []

    for idx, row in df.iterrows():
        home = row['home_team']
        away = row['away_team']
        date = row['date']

        # Get most recent rolling stats for each team BEFORE this game
        home_stats = rolling_stats.get(home)
        away_stats = rolling_stats.get(away)

        if home_stats is None or away_stats is None:
            continue

        # Find the last entry before this game date
        home_prior = home_stats[home_stats['date'] < date]
        away_prior = away_stats[away_stats['date'] < date]

        if len(home_prior) < 5 or len(away_prior) < 5:
            continue  # Need minimum history

        home_latest = home_prior.iloc[-1]
        away_latest = away_prior.iloc[-1]

        features = {}

        # Rolling stat features for both teams
        for window in [5, 10, 20]:
            for stat in stat_cols + ['win']:
                h_col = f'{stat}_roll_{window}'
                a_col = f'{stat}_roll_{window}'

                h_val = home_latest.get(h_col, np.nan)
                a_val = away_latest.get(a_col, np.nan)

                features[f'home_{stat}_r{window}'] = h_val
                features[f'away_{stat}_r{window}'] = a_val
                # Differential features (home - away) are very predictive
                features[f'diff_{stat}_r{window}'] = h_val - a_val if pd.notna(h_val) and pd.notna(a_val) else np.nan

        # Conference matchup
        features['same_conf'] = 1 if row['home_conf'] == row['away_conf'] else 0

        # Target
        features['home_win'] = row['home_win']
        features['game_id'] = row['game_id']
        features['date'] = row['date']
        features['home_team'] = home
        features['away_team'] = away

        feature_rows.append(features)

    return pd.DataFrame(feature_rows)


def train_model(features_df):
    """Train XGBoost classifier with time-series aware cross-validation."""

    # Separate features and target
    meta_cols = ['game_id', 'date', 'home_team', 'away_team', 'home_win']
    feature_cols = [c for c in features_df.columns if c not in meta_cols]

    X = features_df[feature_cols].copy()
    y = features_df['home_win'].copy()

    # Handle any remaining NaN
    X = X.fillna(X.median())

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    print(f"Home win rate: {y.mean():.3f}")

    # XGBoost model with tuned hyperparameters
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
    )

    # Time-series cross-validation (respects temporal ordering)
    print("\nRunning TimeSeriesSplit cross-validation...")
    tscv = TimeSeriesSplit(n_splits=5)

    cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy')
    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"Per fold: {[f'{s:.4f}' for s in cv_scores]}")

    auc_scores = cross_val_score(model, X, y, cv=tscv, scoring='roc_auc')
    print(f"CV AUC: {auc_scores.mean():.4f} (+/- {auc_scores.std():.4f})")

    # Train final model on all data
    print("\nTraining final model on all data...")
    model.fit(X, y)

    # Feature importance
    importance = pd.Series(model.feature_importances_, index=feature_cols)
    top_features = importance.nlargest(15)
    print("\nTop 15 Most Important Features:")
    for feat, imp in top_features.items():
        print(f"  {feat}: {imp:.4f}")

    return model, feature_cols, cv_scores, auc_scores, top_features


def main():
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(model_dir, exist_ok=True)

    # Load data
    print("Loading game data...")
    df = pd.read_csv(os.path.join(data_dir, 'nba_games.csv'))
    print(f"Loaded {len(df)} games across {df['season'].nunique()} seasons")

    # Build features
    print("\nBuilding feature matrix...")
    features_df = build_features(df)
    print(f"Feature matrix: {features_df.shape}")

    # Train model
    model, feature_cols, cv_scores, auc_scores, top_features = train_model(features_df)

    # Save model and metadata
    joblib.dump(model, os.path.join(model_dir, 'xgb_model.pkl'))
    joblib.dump(feature_cols, os.path.join(model_dir, 'feature_cols.pkl'))

    # Save evaluation metrics
    metrics = {
        'cv_accuracy_mean': float(cv_scores.mean()),
        'cv_accuracy_std': float(cv_scores.std()),
        'cv_auc_mean': float(auc_scores.mean()),
        'cv_auc_std': float(auc_scores.std()),
        'top_features': {k: float(v) for k, v in top_features.items()},
        'n_features': len(feature_cols),
        'n_training_games': len(features_df),
    }

    import json
    with open(os.path.join(model_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModel saved to {model_dir}/")
    print("Done!")

    return model, feature_cols, metrics


if __name__ == '__main__':
    main()
