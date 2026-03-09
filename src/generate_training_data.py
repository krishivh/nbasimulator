"""
Generate realistic NBA game training data based on historical patterns.
This creates a dataset of game-level features and outcomes for ML training.

In production, you'd pull this from the Kaggle dataset or nba_api.
This generator creates statistically realistic data following real NBA distributions.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

np.random.seed(42)

# All 30 NBA teams with realistic strength ratings (based on recent seasons)
TEAMS = {
    # Eastern Conference
    'ATL': {'conf': 'East', 'div': 'Southeast', 'base_strength': 0.50},
    'BOS': {'conf': 'East', 'div': 'Atlantic', 'base_strength': 0.68},
    'BKN': {'conf': 'East', 'div': 'Atlantic', 'base_strength': 0.30},
    'CHA': {'conf': 'East', 'div': 'Southeast', 'base_strength': 0.48},
    'CHI': {'conf': 'East', 'div': 'Central', 'base_strength': 0.42},
    'CLE': {'conf': 'East', 'div': 'Central', 'base_strength': 0.62},
    'DET': {'conf': 'East', 'div': 'Central', 'base_strength': 0.76},
    'IND': {'conf': 'East', 'div': 'Central', 'base_strength': 0.25},
    'MIA': {'conf': 'East', 'div': 'Southeast', 'base_strength': 0.52},
    'MIL': {'conf': 'East', 'div': 'Central', 'base_strength': 0.44},
    'NYK': {'conf': 'East', 'div': 'Atlantic', 'base_strength': 0.64},
    'ORL': {'conf': 'East', 'div': 'Southeast', 'base_strength': 0.53},
    'PHI': {'conf': 'East', 'div': 'Atlantic', 'base_strength': 0.55},
    'TOR': {'conf': 'East', 'div': 'Atlantic', 'base_strength': 0.58},
    'WAS': {'conf': 'East', 'div': 'Southeast', 'base_strength': 0.27},
    # Western Conference
    'DAL': {'conf': 'West', 'div': 'Southwest', 'base_strength': 0.35},
    'DEN': {'conf': 'West', 'div': 'Northwest', 'base_strength': 0.61},
    'GSW': {'conf': 'West', 'div': 'Pacific', 'base_strength': 0.52},
    'HOU': {'conf': 'West', 'div': 'Southwest', 'base_strength': 0.63},
    'LAC': {'conf': 'West', 'div': 'Pacific', 'base_strength': 0.47},
    'LAL': {'conf': 'West', 'div': 'Pacific', 'base_strength': 0.60},
    'MEM': {'conf': 'West', 'div': 'Southwest', 'base_strength': 0.39},
    'MIN': {'conf': 'West', 'div': 'Northwest', 'base_strength': 0.62},
    'NOP': {'conf': 'West', 'div': 'Southwest', 'base_strength': 0.31},
    'OKC': {'conf': 'West', 'div': 'Northwest', 'base_strength': 0.76},
    'PHX': {'conf': 'West', 'div': 'Pacific', 'base_strength': 0.57},
    'POR': {'conf': 'West', 'div': 'Northwest', 'base_strength': 0.47},
    'SAC': {'conf': 'West', 'div': 'Pacific', 'base_strength': 0.23},
    'SAS': {'conf': 'West', 'div': 'Southwest', 'base_strength': 0.72},
    'UTA': {'conf': 'West', 'div': 'Northwest', 'base_strength': 0.30},
}

TEAM_ABBREVS = list(TEAMS.keys())

def generate_team_game_stats(strength, is_home, opp_strength):
    """Generate realistic box score stats for one team in a game."""
    # Home court advantage ~3.5 points historically
    home_boost = 0.03 if is_home else -0.03
    effective = strength + home_boost

    # Pace (possessions per game) - NBA average ~100
    pace = np.random.normal(100, 3)

    # Offensive/defensive ratings influenced by team strength
    off_rating = np.random.normal(110 + (effective - 0.5) * 20, 4)
    def_rating = np.random.normal(110 - (effective - 0.5) * 15 + (opp_strength - 0.5) * 10, 4)

    # Points scored based on offensive rating and pace
    pts = max(80, int(np.random.normal(off_rating * pace / 100, 8)))

    # Field goals
    fga = int(np.random.normal(88, 5))
    fg_pct = np.clip(np.random.normal(0.46 + (effective - 0.5) * 0.04, 0.03), 0.35, 0.58)
    fgm = int(fga * fg_pct)

    # Three pointers
    fg3a = int(np.random.normal(37, 5))
    fg3_pct = np.clip(np.random.normal(0.36 + (effective - 0.5) * 0.02, 0.04), 0.25, 0.48)
    fg3m = int(fg3a * fg3_pct)

    # Free throws
    fta = int(np.random.normal(22, 5))
    ft_pct = np.clip(np.random.normal(0.78, 0.04), 0.65, 0.92)
    ftm = int(fta * ft_pct)

    # Recalculate points from components
    pts = fgm * 2 + fg3m * 1 + ftm  # fg3m already counted in fgm for 2pts, add extra 1

    # Rebounds
    oreb = int(np.random.normal(10, 2.5))
    dreb = int(np.random.normal(34, 3))
    reb = oreb + dreb

    # Other stats
    ast = int(np.random.normal(25 + (effective - 0.5) * 5, 4))
    stl = int(np.random.normal(7.5, 2))
    blk = int(np.random.normal(5, 1.5))
    tov = int(np.random.normal(14 - (effective - 0.5) * 3, 3))
    pf = int(np.random.normal(20, 3))

    return {
        'pts': max(pts, 85),
        'fgm': max(fgm, 28), 'fga': max(fga, 70),
        'fg_pct': round(fgm / max(fga, 1), 3),
        'fg3m': max(fg3m, 5), 'fg3a': max(fg3a, 20),
        'fg3_pct': round(fg3m / max(fg3a, 1), 3),
        'ftm': max(ftm, 8), 'fta': max(fta, 10),
        'ft_pct': round(ftm / max(fta, 1), 3),
        'oreb': max(oreb, 3), 'dreb': max(dreb, 22),
        'reb': max(reb, 30),
        'ast': max(ast, 15), 'stl': max(stl, 2),
        'blk': max(blk, 1), 'tov': max(tov, 5),
        'pf': max(pf, 10),
        'pace': round(pace, 1),
        'off_rating': round(off_rating, 1),
        'def_rating': round(def_rating, 1),
    }


def generate_season(season_year, strength_variation=0.05):
    """Generate a full season of games with realistic scheduling."""
    games = []

    # Add per-season strength variation
    season_strengths = {}
    for team, info in TEAMS.items():
        variation = np.random.normal(0, strength_variation)
        season_strengths[team] = np.clip(info['base_strength'] + variation, 0.15, 0.85)

    # Generate ~1230 games per season (82 games × 30 teams / 2)
    # Simplified: each team plays each other team ~2.7 times
    season_start = datetime(season_year, 10, 22)
    game_id = 0

    matchups = []
    for i, t1 in enumerate(TEAM_ABBREVS):
        for j, t2 in enumerate(TEAM_ABBREVS):
            if i < j:
                # Each pair plays 2-4 times
                n_games = np.random.choice([2, 3, 3, 4],
                    p=[0.15, 0.35, 0.35, 0.15])
                for g in range(n_games):
                    home = t1 if g % 2 == 0 else t2
                    away = t2 if g % 2 == 0 else t1
                    matchups.append((home, away))

    np.random.shuffle(matchups)

    # Distribute games over ~170 days
    for idx, (home, away) in enumerate(matchups):
        game_date = season_start + timedelta(days=int(idx * 170 / len(matchups)))

        home_strength = season_strengths[home]
        away_strength = season_strengths[away]

        home_stats = generate_team_game_stats(home_strength, True, away_strength)
        away_stats = generate_team_game_stats(away_strength, False, home_strength)

        # Determine winner based on points (add some noise for upsets)
        home_score = home_stats['pts']
        away_score = away_stats['pts']

        # Ensure no ties
        if home_score == away_score:
            # OT - add 5-15 points to a random team
            ot_pts = np.random.randint(5, 16)
            if np.random.random() > 0.5:
                home_score += ot_pts
                home_stats['pts'] = home_score
            else:
                away_score += ot_pts
                away_stats['pts'] = away_score

        home_win = 1 if home_score > away_score else 0

        game = {
            'game_id': f"{season_year}_{game_id:04d}",
            'season': f"{season_year}-{str(season_year+1)[-2:]}",
            'date': game_date.strftime('%Y-%m-%d'),
            'home_team': home,
            'away_team': away,
            'home_conf': TEAMS[home]['conf'],
            'away_conf': TEAMS[away]['conf'],
            'home_win': home_win,
            # Home team stats
            'home_pts': home_stats['pts'],
            'home_fgm': home_stats['fgm'], 'home_fga': home_stats['fga'],
            'home_fg_pct': home_stats['fg_pct'],
            'home_fg3m': home_stats['fg3m'], 'home_fg3a': home_stats['fg3a'],
            'home_fg3_pct': home_stats['fg3_pct'],
            'home_ftm': home_stats['ftm'], 'home_fta': home_stats['fta'],
            'home_ft_pct': home_stats['ft_pct'],
            'home_oreb': home_stats['oreb'], 'home_dreb': home_stats['dreb'],
            'home_reb': home_stats['reb'],
            'home_ast': home_stats['ast'], 'home_stl': home_stats['stl'],
            'home_blk': home_stats['blk'], 'home_tov': home_stats['tov'],
            'home_pf': home_stats['pf'],
            'home_pace': home_stats['pace'],
            'home_off_rating': home_stats['off_rating'],
            'home_def_rating': home_stats['def_rating'],
            # Away team stats
            'away_pts': away_stats['pts'],
            'away_fgm': away_stats['fgm'], 'away_fga': away_stats['fga'],
            'away_fg_pct': away_stats['fg_pct'],
            'away_fg3m': away_stats['fg3m'], 'away_fg3a': away_stats['fg3a'],
            'away_fg3_pct': away_stats['fg3_pct'],
            'away_ftm': away_stats['ftm'], 'away_fta': away_stats['fta'],
            'away_ft_pct': away_stats['ft_pct'],
            'away_oreb': away_stats['oreb'], 'away_dreb': away_stats['dreb'],
            'away_reb': away_stats['reb'],
            'away_ast': away_stats['ast'], 'away_stl': away_stats['stl'],
            'away_blk': away_stats['blk'], 'away_tov': away_stats['tov'],
            'away_pf': away_stats['pf'],
            'away_pace': away_stats['pace'],
            'away_off_rating': away_stats['off_rating'],
            'away_def_rating': away_stats['def_rating'],
        }
        games.append(game)
        game_id += 1

    return pd.DataFrame(games)


def main():
    print("Generating NBA training data...")
    all_seasons = []

    # Generate 5 seasons of data for training
    for year in range(2019, 2025):
        print(f"  Generating {year}-{year+1} season...")
        season_df = generate_season(year)
        all_seasons.append(season_df)
        print(f"    -> {len(season_df)} games")

    df = pd.concat(all_seasons, ignore_index=True)
    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'nba_games.csv')
    df.to_csv(output_path, index=False)
    print(f"\nTotal: {len(df)} games saved to {output_path}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nHome win rate: {df['home_win'].mean():.3f}")
    print(f"Avg home points: {df['home_pts'].mean():.1f}")
    print(f"Avg away points: {df['away_pts'].mean():.1f}")


if __name__ == '__main__':
    main()
