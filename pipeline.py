"""
pipeline.py — Main data pipeline
=================================
This is the SINGLE entry point for the project. It:

1. Loads the Kaggle dataset (historical NBA game data + player box scores)
2. Aggregates player-level data into team-level game features
3. Engineers rolling features (5/10/20 game windows)
4. Trains an XGBoost classifier with TimeSeriesSplit CV
5. Computes per-team strength profiles from the trained model
6. Exports everything the React demo needs as JSON

USAGE:
    # First, download the Kaggle dataset:
    #   pip install kagglehub
    #   python -c "import kagglehub; kagglehub.dataset_download('eoinamoore/historical-nba-data-and-player-box-scores')"
    #
    # Then run this pipeline:
    python pipeline.py --kaggle-path /path/to/kaggle/data

    # Or use the generated training data (if Kaggle isn't available):
    python pipeline.py --use-generated
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier as XGBClassifier
import joblib
import json
import os
import argparse
import warnings
warnings.filterwarnings('ignore')


# ─── STEP 1: Load Kaggle Data ───────────────────────────────────────────

def load_kaggle_data(kaggle_path):
    """
    Load and process the Kaggle NBA dataset.
    Expected files in kaggle_path:
      - PlayerStatistics.csv (player box scores per game)
      - Games.csv or similar (game-level info)
    
    The Kaggle dataset 'eoinamoore/historical-nba-data-and-player-box-scores'
    contains PlayerStatistics.csv with columns like:
      PLAYER_NAME, TEAM_ABBREVIATION, GAME_ID, GAME_DATE, 
      PTS, FGM, FGA, FG3M, FG3A, FTM, FTA, OREB, DREB, REB,
      AST, STL, BLK, TOV, PF, PLUS_MINUS, MIN, etc.
    """
    print("Loading Kaggle dataset...")
    
    # Try different possible file names
    possible_files = [
        'PlayerStatistics.csv', 'player_statistics.csv',
        'player_stats.csv', 'box_scores.csv',
    ]
    
    player_file = None
    for f in possible_files:
        path = os.path.join(kaggle_path, f)
        if os.path.exists(path):
            player_file = path
            break
    
    if not player_file:
        # List what's actually in the directory
        files = os.listdir(kaggle_path)
        print(f"  Files found in {kaggle_path}: {files}")
        # Try the first CSV we find
        csvs = [f for f in files if f.endswith('.csv')]
        if csvs:
            player_file = os.path.join(kaggle_path, csvs[0])
            print(f"  Using: {csvs[0]}")
        else:
            raise FileNotFoundError(f"No CSV files found in {kaggle_path}")
    
    df = pd.read_csv(player_file)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"  Columns: {list(df.columns)[:20]}...")
    
    return df


def aggregate_to_team_games(player_df):
    """
    Aggregate player-level box scores into team-level game stats.
    This is the key transformation from raw Kaggle data to ML features.
    """
    print("Aggregating player stats to team-level game stats...")
    
    # Normalize column names (Kaggle datasets vary)
    col_map = {}
    for col in player_df.columns:
        col_map[col.upper().replace(' ', '_')] = col
    
    # Identify key columns
    team_col = None
    for candidate in ['TEAM_ABBREVIATION', 'TEAM', 'TEAM_ABB', 'TM']:
        for orig, mapped in col_map.items():
            if candidate in orig:
                team_col = mapped
                break
    
    game_col = None
    for candidate in ['GAME_ID', 'GAMEID', 'GAME']:
        for orig, mapped in col_map.items():
            if candidate in orig:
                game_col = mapped
                break
    
    date_col = None
    for candidate in ['GAME_DATE', 'DATE', 'GAMEDATE']:
        for orig, mapped in col_map.items():
            if candidate in orig:
                date_col = mapped
                break
    
    if not all([team_col, game_col, date_col]):
        print(f"  Warning: Could not identify all required columns.")
        print(f"  team_col={team_col}, game_col={game_col}, date_col={date_col}")
        print(f"  Available: {list(player_df.columns)}")
        return None
    
    # Stat columns to aggregate
    stat_cols_map = {
        'PTS': None, 'FGM': None, 'FGA': None, 'FG3M': None, 'FG3A': None,
        'FTM': None, 'FTA': None, 'OREB': None, 'DREB': None, 'REB': None,
        'AST': None, 'STL': None, 'BLK': None, 'TOV': None, 'PF': None,
    }
    
    for stat in list(stat_cols_map.keys()):
        for orig, mapped in col_map.items():
            if stat == orig or stat in orig:
                stat_cols_map[stat] = mapped
                break
    
    available_stats = {k: v for k, v in stat_cols_map.items() if v is not None}
    print(f"  Found stat columns: {list(available_stats.keys())}")
    
    # Aggregate: sum player stats per team per game
    agg_dict = {v: 'sum' for v in available_stats.values()}
    if date_col not in agg_dict:
        agg_dict[date_col] = 'first'
    
    team_games = player_df.groupby([game_col, team_col]).agg(agg_dict).reset_index()
    
    # Rename columns to standard names
    rename = {v: k.lower() for k, v in available_stats.items()}
    rename[team_col] = 'team'
    rename[game_col] = 'game_id'
    rename[date_col] = 'date'
    team_games = team_games.rename(columns=rename)
    
    # Compute percentages
    if 'fgm' in team_games.columns and 'fga' in team_games.columns:
        team_games['fg_pct'] = team_games['fgm'] / team_games['fga'].clip(lower=1)
    if 'fg3m' in team_games.columns and 'fg3a' in team_games.columns:
        team_games['fg3_pct'] = team_games['fg3m'] / team_games['fg3a'].clip(lower=1)
    if 'ftm' in team_games.columns and 'fta' in team_games.columns:
        team_games['ft_pct'] = team_games['ftm'] / team_games['fta'].clip(lower=1)
    
    # Approximate pace and ratings
    if 'pts' in team_games.columns and 'fga' in team_games.columns:
        team_games['pace'] = team_games['fga'] + 0.44 * team_games.get('fta', 0) - team_games.get('oreb', 0) + team_games.get('tov', 0)
        team_games['off_rating'] = team_games['pts'] / team_games['pace'].clip(lower=1) * 100
    
    print(f"  Aggregated to {len(team_games)} team-game records")
    return team_games


def build_matchup_dataset(team_games):
    """
    Pair team-game records into home vs away matchups for ML training.
    Each game has two rows in team_games (one per team); we join them.
    """
    print("Building matchup dataset...")
    
    # Each game_id should have exactly 2 teams
    game_teams = team_games.groupby('game_id')['team'].apply(list).reset_index()
    game_teams = game_teams[game_teams['team'].apply(len) == 2]
    
    matchups = []
    for _, row in game_teams.iterrows():
        gid = row['game_id']
        teams = row['team']
        
        # Assign home/away (first alphabetically = home, or use actual if available)
        home, away = sorted(teams)
        
        home_stats = team_games[(team_games['game_id'] == gid) & (team_games['team'] == home)].iloc[0]
        away_stats = team_games[(team_games['game_id'] == gid) & (team_games['team'] == away)].iloc[0]
        
        home_win = 1 if home_stats.get('pts', 0) > away_stats.get('pts', 0) else 0
        
        game = {'game_id': gid, 'date': home_stats['date'], 'home_team': home, 'away_team': away, 'home_win': home_win}
        
        # Add prefixed stats
        for col in home_stats.index:
            if col not in ['game_id', 'team', 'date']:
                game[f'home_{col}'] = home_stats[col]
                game[f'away_{col}'] = away_stats[col]
        
        matchups.append(game)
    
    df = pd.DataFrame(matchups)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    print(f"  Built {len(df)} matchup records")
    return df


# ─── STEP 2: Feature Engineering ────────────────────────────────────────

def compute_rolling_features(df, windows=[5, 10, 20]):
    """
    For each game, compute rolling averages of each team's recent stats.
    These become the ML features — we can't use same-game stats at prediction time.
    """
    print("Computing rolling features...")
    
    stat_cols = [c.replace('home_', '') for c in df.columns 
                 if c.startswith('home_') and c != 'home_win' and c != 'home_team'
                 and c != 'home_conf' and df[c].dtype in ['float64', 'int64', 'float32', 'int32']]
    
    # Build per-team timelines
    team_histories = {}
    all_teams = set(df['home_team'].unique()) | set(df['away_team'].unique())
    
    for team in all_teams:
        home_mask = df['home_team'] == team
        away_mask = df['away_team'] == team
        
        records = []
        for idx in df.index:
            row = df.loc[idx]
            if row['home_team'] == team:
                record = {'date': row['date'], 'win': row['home_win']}
                for s in stat_cols:
                    record[s] = row.get(f'home_{s}', np.nan)
                records.append(record)
            elif row['away_team'] == team:
                record = {'date': row['date'], 'win': 1 - row['home_win']}
                for s in stat_cols:
                    record[s] = row.get(f'away_{s}', np.nan)
                records.append(record)
        
        tdf = pd.DataFrame(records).sort_values('date')
        for w in windows:
            for s in stat_cols + ['win']:
                tdf[f'{s}_r{w}'] = tdf[s].rolling(w, min_periods=max(1, w // 2)).mean()
        
        team_histories[team] = tdf
    
    # Now build feature rows for each game
    print("  Building feature matrix from rolling stats...")
    feature_rows = []
    
    for idx, row in df.iterrows():
        home = row['home_team']
        away = row['away_team']
        date = row['date']
        
        home_hist = team_histories.get(home)
        away_hist = team_histories.get(away)
        
        if home_hist is None or away_hist is None:
            continue
        
        home_prior = home_hist[home_hist['date'] < date]
        away_prior = away_hist[away_hist['date'] < date]
        
        if len(home_prior) < 5 or len(away_prior) < 5:
            continue
        
        hl = home_prior.iloc[-1]
        al = away_prior.iloc[-1]
        
        features = {'game_id': row['game_id'], 'date': date, 
                     'home_team': home, 'away_team': away, 'home_win': row['home_win']}
        
        for w in windows:
            for s in stat_cols + ['win']:
                hk = f'{s}_r{w}'
                features[f'home_{s}_r{w}'] = hl.get(hk, np.nan)
                features[f'away_{s}_r{w}'] = al.get(hk, np.nan)
                hv = hl.get(hk, np.nan)
                av = al.get(hk, np.nan)
                if pd.notna(hv) and pd.notna(av):
                    features[f'diff_{s}_r{w}'] = hv - av
        
        feature_rows.append(features)
    
    result = pd.DataFrame(feature_rows)
    print(f"  Feature matrix: {result.shape}")
    return result


# ─── STEP 3: Train Model ────────────────────────────────────────────────

def train_model(features_df):
    """Train XGBoost with TimeSeriesSplit cross-validation."""
    print("\nTraining XGBoost model...")
    
    meta_cols = ['game_id', 'date', 'home_team', 'away_team', 'home_win']
    feature_cols = [c for c in features_df.columns if c not in meta_cols]
    
    X = features_df[feature_cols].fillna(features_df[feature_cols].median())
    y = features_df['home_win']
    
    print(f"  Features: {len(feature_cols)}, Samples: {len(X)}")
    print(f"  Home win rate: {y.mean():.3f}")
    
    model = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, random_state=42,
    )
    
    tscv = TimeSeriesSplit(n_splits=5)
    acc_scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy')
    auc_scores = cross_val_score(model, X, y, cv=tscv, scoring='roc_auc')
    
    print(f"  CV Accuracy: {acc_scores.mean():.4f} (+/- {acc_scores.std():.4f})")
    print(f"  CV AUC:      {auc_scores.mean():.4f} (+/- {auc_scores.std():.4f})")
    
    model.fit(X, y)
    
    # Feature importance
    importance = pd.Series(model.feature_importances_, index=feature_cols)
    top_feats = importance.nlargest(15)
    
    return model, feature_cols, {
        'cv_accuracy': float(acc_scores.mean()),
        'cv_accuracy_std': float(acc_scores.std()),
        'cv_auc': float(auc_scores.mean()),
        'cv_auc_std': float(auc_scores.std()),
        'n_features': len(feature_cols),
        'n_training_games': len(X),
        'top_features': {k: float(v) for k, v in top_feats.items()},
    }


# ─── STEP 4: Export Team Profiles for React Demo ────────────────────────

def compute_team_profiles(features_df, model, feature_cols):
    """
    Compute per-team strength profiles from the trained model.
    For each team, we compute:
      - Average predicted win probability when playing at home
      - Average predicted win probability when playing away
      - Overall strength rating
      - Recent form (last 20 games rolling stats)
    
    This is exported as JSON for the React demo to use.
    """
    print("\nComputing team strength profiles...")
    
    meta_cols = ['game_id', 'date', 'home_team', 'away_team', 'home_win']
    X = features_df[[c for c in features_df.columns if c not in meta_cols]].fillna(0)
    
    # Get predicted probabilities for all games
    probs = model.predict_proba(X)[:, 1]  # P(home_win)
    features_df = features_df.copy()
    features_df['pred_home_win_prob'] = probs
    
    # Recent games only (last season worth)
    recent = features_df.tail(len(features_df) // 3)
    
    profiles = {}
    all_teams = set(recent['home_team'].unique()) | set(recent['away_team'].unique())
    
    for team in all_teams:
        home_games = recent[recent['home_team'] == team]
        away_games = recent[recent['away_team'] == team]
        
        # When this team is home, how likely does the model say they win?
        avg_home_win_prob = home_games['pred_home_win_prob'].mean() if len(home_games) > 0 else 0.5
        # When this team is away, prob of away win = 1 - P(home_win)
        avg_away_win_prob = (1 - away_games['pred_home_win_prob']).mean() if len(away_games) > 0 else 0.5
        
        overall_strength = (avg_home_win_prob + avg_away_win_prob) / 2
        
        # Actual win rate
        home_wins = home_games['home_win'].sum() if len(home_games) > 0 else 0
        away_wins = (1 - away_games['home_win']).sum() if len(away_games) > 0 else 0
        total_games = len(home_games) + len(away_games)
        actual_win_rate = (home_wins + away_wins) / max(total_games, 1)
        
        profiles[team] = {
            'strength': round(float(overall_strength), 4),
            'home_win_prob': round(float(avg_home_win_prob), 4),
            'away_win_prob': round(float(avg_away_win_prob), 4),
            'actual_win_rate': round(float(actual_win_rate), 4),
            'games_analyzed': int(total_games),
        }
    
    return profiles


def export_for_react(profiles, metrics, output_dir):
    """
    Export everything the React demo needs:
      - team_profiles.json: per-team ML strength ratings
      - model_metrics.json: model evaluation metrics
    """
    print("\nExporting for React demo...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'team_profiles.json'), 'w') as f:
        json.dump(profiles, f, indent=2)
    
    with open(os.path.join(output_dir, 'model_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"  Saved team_profiles.json ({len(profiles)} teams)")
    print(f"  Saved model_metrics.json")
    print(f"\n  Copy these files to your React app's public/ directory,")
    print(f"  then update the React app to fetch and use them.")


# ─── MAIN ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='NBA Season Simulator Pipeline')
    parser.add_argument('--kaggle-path', type=str, default=None,
                        help='Path to downloaded Kaggle dataset directory')
    parser.add_argument('--use-generated', action='store_true',
                        help='Use the generated training data instead of Kaggle')
    parser.add_argument('--output', type=str, default='exports',
                        help='Output directory for React demo files')
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    if args.kaggle_path:
        # ── Kaggle path: full pipeline ──
        player_df = load_kaggle_data(args.kaggle_path)
        team_games = aggregate_to_team_games(player_df)
        
        if team_games is not None:
            matchups = build_matchup_dataset(team_games)
            matchups.to_csv(os.path.join(base_dir, 'data', 'processed_games.csv'), index=False)
            print(f"  Saved processed_games.csv")
        else:
            print("  Failed to aggregate. Falling back to generated data.")
            matchups = pd.read_csv(os.path.join(base_dir, 'data', 'nba_games.csv'))
    
    elif args.use_generated:
        # ── Use generated data ──
        print("Using generated training data...")
        matchups = pd.read_csv(os.path.join(base_dir, 'data', 'nba_games.csv'))
        matchups['date'] = pd.to_datetime(matchups['date'])
    
    else:
        print("Please specify --kaggle-path or --use-generated")
        print("\nTo download Kaggle data:")
        print("  pip install kagglehub")
        print("  python -c \"import kagglehub; print(kagglehub.dataset_download('eoinamoore/historical-nba-data-and-player-box-scores'))\"")
        print("  python pipeline.py --kaggle-path /path/to/downloaded/data")
        return
    
    # Feature engineering
    features_df = compute_rolling_features(matchups)
    
    # Train model
    model, feature_cols, metrics = train_model(features_df)
    
    # Save model
    model_dir = os.path.join(base_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, 'xgb_model.pkl'))
    joblib.dump(feature_cols, os.path.join(model_dir, 'feature_cols.pkl'))
    
    # Compute team profiles
    profiles = compute_team_profiles(features_df, model, feature_cols)
    
    # Export for React
    export_for_react(profiles, metrics, os.path.join(base_dir, args.output))
    
    # Print summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Training games: {metrics['n_training_games']}")
    print(f"Features: {metrics['n_features']}")
    print(f"CV Accuracy: {metrics['cv_accuracy']:.4f}")
    print(f"CV AUC: {metrics['cv_auc']:.4f}")
    print(f"\nTop teams by ML strength:")
    for team, profile in sorted(profiles.items(), key=lambda x: x[1]['strength'], reverse=True)[:10]:
        print(f"  {team}: {profile['strength']:.4f} (actual: {profile['actual_win_rate']:.3f})")
    print(f"\nExports saved to: {os.path.join(base_dir, args.output)}/")
    print(f"Model saved to: {model_dir}/")


if __name__ == '__main__':
    main()
