"""
NBA Season & Playoff Monte Carlo Simulation Engine

Simulates the remainder of the 2025-26 NBA season N times,
generates playoff brackets, and simulates the playoffs to determine
championship probabilities for each team.
"""
import pandas as pd
import numpy as np
import joblib
import json
import os
from collections import defaultdict

# Current 2025-26 standings as of March 2, 2026
CURRENT_STANDINGS = {
    'East': {
        'DET': {'wins': 45, 'losses': 14},
        'BOS': {'wins': 40, 'losses': 20},
        'NYK': {'wins': 39, 'losses': 22},
        'CLE': {'wins': 38, 'losses': 24},
        'TOR': {'wins': 35, 'losses': 25},
        'PHI': {'wins': 33, 'losses': 27},
        'MIA': {'wins': 32, 'losses': 29},
        'ORL': {'wins': 31, 'losses': 28},
        'ATL': {'wins': 31, 'losses': 31},
        'CHA': {'wins': 30, 'losses': 31},
        'MIL': {'wins': 26, 'losses': 33},
        'CHI': {'wins': 25, 'losses': 36},
        'WAS': {'wins': 16, 'losses': 43},
        'IND': {'wins': 15, 'losses': 46},
        'BKN': {'wins': 15, 'losses': 45},
    },
    'West': {
        'OKC': {'wins': 47, 'losses': 15},
        'SAS': {'wins': 43, 'losses': 17},
        'MIN': {'wins': 38, 'losses': 23},
        'HOU': {'wins': 37, 'losses': 22},
        'DEN': {'wins': 37, 'losses': 24},
        'LAL': {'wins': 36, 'losses': 24},
        'PHX': {'wins': 34, 'losses': 26},
        'MIA': {'wins': 32, 'losses': 29},
        'GSW': {'wins': 31, 'losses': 29},
        'POR': {'wins': 29, 'losses': 33},
        'LAC': {'wins': 28, 'losses': 31},
        'MEM': {'wins': 23, 'losses': 36},
        'DAL': {'wins': 21, 'losses': 39},
        'NOP': {'wins': 19, 'losses': 43},
        'UTA': {'wins': 18, 'losses': 42},
        'SAC': {'wins': 14, 'losses': 48},
    }
}

# Fix: MIA is in East, not West. Remove duplicate
CURRENT_STANDINGS['West'].pop('MIA', None)

# Team strength ratings derived from current win%
def get_team_strength(team_abbrev):
    """Get team strength from current season win percentage."""
    for conf in CURRENT_STANDINGS.values():
        if team_abbrev in conf:
            record = conf[team_abbrev]
            total = record['wins'] + record['losses']
            return record['wins'] / total if total > 0 else 0.5
    return 0.5


def get_team_conference(team_abbrev):
    """Get which conference a team belongs to."""
    for conf_name, teams in CURRENT_STANDINGS.items():
        if team_abbrev in teams:
            return conf_name
    return None


def predict_game_winner(home_team, away_team, model=None, use_simple=True):
    """
    Predict win probability for a single game.
    
    Uses a logistic model based on team strengths with home court advantage.
    In production with the full ML model, this would use the trained XGBoost.
    """
    home_strength = get_team_strength(home_team)
    away_strength = get_team_strength(away_team)

    # Log5 method (Bill James) for matchup probability
    # P(home wins) = (p_h * (1 - p_a)) / (p_h * (1 - p_a) + p_a * (1 - p_h))
    # Then adjust for home court advantage (~60% historical home win rate)
    HOME_ADVANTAGE = 0.035  # ~3.5% boost

    p_h = home_strength + HOME_ADVANTAGE
    p_a = away_strength

    # Log5 formula
    if p_h + p_a == 0:
        return 0.5

    log5_prob = (p_h * (1 - p_a)) / (p_h * (1 - p_a) + p_a * (1 - p_h))

    # Clamp to reasonable range
    return np.clip(log5_prob, 0.05, 0.95)


def generate_remaining_schedule():
    """
    Generate remaining regular season games.
    NBA regular season ends ~April 13, 2026.
    Each team plays 82 games total.
    """
    remaining_games = []

    all_teams = []
    for conf, teams in CURRENT_STANDINGS.items():
        for team, record in teams.items():
            games_played = record['wins'] + record['losses']
            games_remaining = 82 - games_played
            all_teams.append((team, conf, games_remaining))

    # Generate approximate remaining matchups
    # Pair teams that still need games
    np.random.seed(None)  # Different each simulation

    teams_needing_games = {t: rem for t, c, rem in all_teams if rem > 0}

    game_id = 0
    attempts = 0
    max_attempts = 5000

    while any(v > 0 for v in teams_needing_games.values()) and attempts < max_attempts:
        available = [t for t, g in teams_needing_games.items() if g > 0]
        if len(available) < 2:
            break

        t1, t2 = np.random.choice(available, 2, replace=False)

        # Randomly assign home/away
        if np.random.random() > 0.5:
            home, away = t1, t2
        else:
            home, away = t2, t1

        remaining_games.append({
            'game_id': f'sim_{game_id}',
            'home_team': home,
            'away_team': away,
        })

        teams_needing_games[t1] -= 1
        teams_needing_games[t2] -= 1
        game_id += 1
        attempts += 1

    return remaining_games


def simulate_remaining_season(n_sims=1000):
    """
    Monte Carlo simulation of the remaining regular season.
    Returns championship probabilities and final standings distributions.
    """
    print(f"Running {n_sims} season simulations...")

    # Track results
    final_standings = defaultdict(lambda: defaultdict(list))
    playoff_appearances = defaultdict(int)
    conference_winners = defaultdict(int)
    champions = defaultdict(int)

    seed_counts = defaultdict(lambda: defaultdict(int))

    for sim in range(n_sims):
        if (sim + 1) % 100 == 0:
            print(f"  Simulation {sim + 1}/{n_sims}...")

        # Copy current standings
        sim_standings = {}
        for conf, teams in CURRENT_STANDINGS.items():
            for team, record in teams.items():
                sim_standings[team] = {
                    'wins': record['wins'],
                    'losses': record['losses'],
                    'conf': conf
                }

        # Generate and simulate remaining games
        remaining = generate_remaining_schedule()

        for game in remaining:
            home = game['home_team']
            away = game['away_team']

            win_prob = predict_game_winner(home, away)

            if np.random.random() < win_prob:
                sim_standings[home]['wins'] += 1
                sim_standings[away]['losses'] += 1
            else:
                sim_standings[away]['wins'] += 1
                sim_standings[home]['losses'] += 1

        # Determine final standings by conference
        for conf in ['East', 'West']:
            conf_teams = [(t, s) for t, s in sim_standings.items() if s['conf'] == conf]
            conf_teams.sort(key=lambda x: x[1]['wins'], reverse=True)

            for rank, (team, stats) in enumerate(conf_teams):
                total = stats['wins'] + stats['losses']
                win_pct = stats['wins'] / total if total > 0 else 0
                final_standings[team]['wins'].append(stats['wins'])
                final_standings[team]['losses'].append(stats['losses'])
                final_standings[team]['win_pct'].append(win_pct)

                seed = rank + 1
                seed_counts[team][seed] += 1

                if seed <= 10:  # Play-in eligible
                    playoff_appearances[team] += 1

        # Simulate playoffs
        champion = simulate_playoffs(sim_standings)
        if champion:
            champions[champion] += 1

    # Compile results
    results = {
        'standings': {},
        'playoff_probs': {},
        'championship_probs': {},
        'seed_distributions': {},
    }

    all_team_abbrevs = set()
    for conf_teams in CURRENT_STANDINGS.values():
        all_team_abbrevs.update(conf_teams.keys())

    for team in all_team_abbrevs:
        wins = final_standings[team]['wins']
        if not wins:
            continue
        results['standings'][team] = {
            'avg_wins': np.mean(wins),
            'avg_losses': np.mean(final_standings[team]['losses']),
            'avg_win_pct': np.mean(final_standings[team]['win_pct']),
            'win_range': (int(np.percentile(wins, 5)), int(np.percentile(wins, 95))),
            'conf': sim_standings[team]['conf'],
        }
        results['playoff_probs'][team] = playoff_appearances[team] / n_sims
        results['championship_probs'][team] = champions[team] / n_sims
        results['seed_distributions'][team] = {
            str(k): v / n_sims for k, v in sorted(seed_counts[team].items())
        }

    return results


def simulate_playoffs(standings):
    """
    Simulate the NBA playoffs given final standings.
    Implements play-in tournament + standard 7-game series format.
    """
    for conf in ['East', 'West']:
        conf_teams = [(t, s) for t, s in standings.items() 
                      if isinstance(s, dict) and 'conf' in s and s['conf'] == conf]
        conf_teams.sort(key=lambda x: x[1]['wins'], reverse=True)

        if len(conf_teams) < 10:
            continue

        # Seeds 1-6 are locked in
        seeds = {i+1: conf_teams[i][0] for i in range(6)}

        # Play-in tournament
        team_7 = conf_teams[6][0]
        team_8 = conf_teams[7][0]
        team_9 = conf_teams[8][0]
        team_10 = conf_teams[9][0]

        # 7 vs 8 (winner gets 7 seed)
        winner_78 = simulate_single_game(team_7, team_8)
        loser_78 = team_8 if winner_78 == team_7 else team_7

        # 9 vs 10 (loser eliminated)
        winner_910 = simulate_single_game(team_9, team_10)

        # Loser of 7/8 vs winner of 9/10 (winner gets 8 seed)
        winner_final = simulate_single_game(loser_78, winner_910)

        seeds[7] = winner_78
        seeds[8] = winner_final

        standings[f'{conf}_seeds'] = seeds

    # Playoff bracket - First round
    east_seeds = standings.get('East_seeds', {})
    west_seeds = standings.get('West_seeds', {})

    if not east_seeds or not west_seeds:
        return None

    # East playoffs
    east_r1_winners = []
    for matchup in [(1, 8), (2, 7), (3, 6), (4, 5)]:
        higher = east_seeds.get(matchup[0])
        lower = east_seeds.get(matchup[1])
        if higher and lower:
            winner = simulate_series(higher, lower)
            east_r1_winners.append(winner)

    # East semis
    east_semis_winners = []
    for i in range(0, len(east_r1_winners), 2):
        if i + 1 < len(east_r1_winners):
            winner = simulate_series(east_r1_winners[i], east_r1_winners[i+1])
            east_semis_winners.append(winner)

    # East finals
    east_champion = None
    if len(east_semis_winners) == 2:
        east_champion = simulate_series(east_semis_winners[0], east_semis_winners[1])

    # West playoffs
    west_r1_winners = []
    for matchup in [(1, 8), (2, 7), (3, 6), (4, 5)]:
        higher = west_seeds.get(matchup[0])
        lower = west_seeds.get(matchup[1])
        if higher and lower:
            winner = simulate_series(higher, lower)
            west_r1_winners.append(winner)

    west_semis_winners = []
    for i in range(0, len(west_r1_winners), 2):
        if i + 1 < len(west_r1_winners):
            winner = simulate_series(west_r1_winners[i], west_r1_winners[i+1])
            west_semis_winners.append(winner)

    west_champion = None
    if len(west_semis_winners) == 2:
        west_champion = simulate_series(west_semis_winners[0], west_semis_winners[1])

    # NBA Finals
    if east_champion and west_champion:
        champion = simulate_series(east_champion, west_champion)
        return champion

    return None


def simulate_single_game(team_a, team_b):
    """Simulate a single game between two teams."""
    prob_a = predict_game_winner(team_a, team_b)
    return team_a if np.random.random() < prob_a else team_b


def simulate_series(higher_seed, lower_seed, best_of=7):
    """
    Simulate a best-of-7 playoff series.
    Higher seed has home court advantage (games 1,2,5,7 at home).
    """
    wins_higher = 0
    wins_lower = 0
    needed = (best_of // 2) + 1

    # Home court pattern: H,H,A,A,H,A,H
    home_pattern = [higher_seed, higher_seed, lower_seed, lower_seed,
                    higher_seed, lower_seed, higher_seed]

    game_num = 0
    while wins_higher < needed and wins_lower < needed and game_num < best_of:
        home_team = home_pattern[game_num]
        away_team = lower_seed if home_team == higher_seed else higher_seed

        # Playoff intensity - slightly more variance, slight home boost
        prob = predict_game_winner(home_team, away_team)

        if np.random.random() < prob:
            if home_team == higher_seed:
                wins_higher += 1
            else:
                wins_lower += 1
        else:
            if away_team == higher_seed:
                wins_higher += 1
            else:
                wins_lower += 1

        game_num += 1

    return higher_seed if wins_higher >= needed else lower_seed


def main():
    results = simulate_remaining_season(n_sims=1000)

    # Print results
    print("\n" + "="*70)
    print("NBA 2025-26 SEASON SIMULATION RESULTS (1000 simulations)")
    print("="*70)

    print("\n--- PROJECTED FINAL STANDINGS ---")
    for conf in ['East', 'West']:
        print(f"\n{conf}ern Conference:")
        conf_teams = [(t, s) for t, s in results['standings'].items() if s['conf'] == conf]
        conf_teams.sort(key=lambda x: x[1]['avg_wins'], reverse=True)

        print(f"  {'Team':<6} {'Avg W':>6} {'Avg L':>6} {'Win%':>6} {'Range (5-95%)':>15}")
        print(f"  {'-'*42}")
        for team, stats in conf_teams:
            lo, hi = stats['win_range']
            print(f"  {team:<6} {stats['avg_wins']:>6.1f} {stats['avg_losses']:>6.1f} "
                  f"{stats['avg_win_pct']:>6.3f} {lo:>6}-{hi}")

    print("\n--- CHAMPIONSHIP PROBABILITIES ---")
    champ_probs = sorted(results['championship_probs'].items(),
                         key=lambda x: x[1], reverse=True)
    for team, prob in champ_probs[:10]:
        bar = '█' * int(prob * 100)
        print(f"  {team:<6} {prob*100:>5.1f}% {bar}")

    print("\n--- PLAYOFF PROBABILITIES ---")
    playoff_probs = sorted(results['playoff_probs'].items(),
                          key=lambda x: x[1], reverse=True)
    for team, prob in playoff_probs:
        if prob > 0.01:
            print(f"  {team:<6} {prob*100:>5.1f}%")

    # Save results
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    with open(os.path.join(output_dir, 'simulation_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir}/simulation_results.json")

    return results


if __name__ == '__main__':
    main()
