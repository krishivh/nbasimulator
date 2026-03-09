"""
NBA Season Simulator Dashboard
===============================
Interactive Streamlit dashboard that visualizes:
- Current standings
- ML model performance
- Monte Carlo simulation results
- Projected standings & championship probabilities
- Playoff bracket visualization
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config
st.set_page_config(
    page_title="NBA Season Simulator",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1D428A;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1D428A 0%, #C8102E 100%);
        border-radius: 12px;
        padding: 1.5rem;
        color: white;
        text-align: center;
    }
    .stMetric > div {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- Data Loading ---
@st.cache_data
def load_simulation_results():
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    with open(os.path.join(data_dir, 'simulation_results.json'), 'r') as f:
        return json.load(f)

@st.cache_data
def load_model_metrics():
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    with open(os.path.join(model_dir, 'metrics.json'), 'r') as f:
        return json.load(f)

@st.cache_data
def load_training_data():
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    return pd.read_csv(os.path.join(data_dir, 'nba_games.csv'))

# Team colors for visualization
TEAM_COLORS = {
    'ATL': '#E03A3E', 'BOS': '#007A33', 'BKN': '#000000', 'CHA': '#1D1160',
    'CHI': '#CE1141', 'CLE': '#860038', 'DAL': '#00538C', 'DEN': '#0E2240',
    'DET': '#C8102E', 'GSW': '#1D428A', 'HOU': '#CE1141', 'IND': '#002D62',
    'LAC': '#C8102E', 'LAL': '#552583', 'MEM': '#5D76A9', 'MIA': '#98002E',
    'MIL': '#00471B', 'MIN': '#0C2340', 'NOP': '#0C2340', 'NYK': '#006BB6',
    'OKC': '#007AC1', 'ORL': '#0077C0', 'PHI': '#006BB6', 'PHX': '#1D1160',
    'POR': '#E03A3E', 'SAC': '#5A2D81', 'SAS': '#C4CED4', 'TOR': '#CE1141',
    'UTA': '#002B5C', 'WAS': '#002B5C',
}

TEAM_NAMES = {
    'ATL': 'Hawks', 'BOS': 'Celtics', 'BKN': 'Nets', 'CHA': 'Hornets',
    'CHI': 'Bulls', 'CLE': 'Cavaliers', 'DAL': 'Mavericks', 'DEN': 'Nuggets',
    'DET': 'Pistons', 'GSW': 'Warriors', 'HOU': 'Rockets', 'IND': 'Pacers',
    'LAC': 'Clippers', 'LAL': 'Lakers', 'MEM': 'Grizzlies', 'MIA': 'Heat',
    'MIL': 'Bucks', 'MIN': 'Timberwolves', 'NOP': 'Pelicans', 'NYK': 'Knicks',
    'OKC': 'Thunder', 'ORL': 'Magic', 'PHI': '76ers', 'PHX': 'Suns',
    'POR': 'Trail Blazers', 'SAC': 'Kings', 'SAS': 'Spurs', 'TOR': 'Raptors',
    'UTA': 'Jazz', 'WAS': 'Wizards',
}

# Load data
try:
    results = load_simulation_results()
    metrics = load_model_metrics()
    training_data = load_training_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Run the simulation first: `python src/simulate.py`")
    st.stop()

# --- Header ---
st.markdown('<div class="main-header">🏀 NBA Season Simulator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">2025-26 Season • Monte Carlo Simulation with ML Predictions</div>', unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", [
    "🏆 Championship Odds",
    "📊 Projected Standings",
    "🤖 Model Performance",
    "🏀 Playoff Bracket",
    "📈 Data Explorer",
])

# ============================================================
# PAGE 1: Championship Odds
# ============================================================
if page == "🏆 Championship Odds":
    st.header("Championship Probabilities")
    st.caption("Based on 1,000 Monte Carlo simulations of the remaining season + playoffs")

    champ_probs = results['championship_probs']

    # Top metrics
    sorted_probs = sorted(champ_probs.items(), key=lambda x: x[1], reverse=True)
    top3 = sorted_probs[:3]

    col1, col2, col3 = st.columns(3)
    for i, (col, (team, prob)) in enumerate(zip([col1, col2, col3], top3)):
        with col:
            medal = ["🥇", "🥈", "🥉"][i]
            st.metric(
                label=f"{medal} {team} {TEAM_NAMES.get(team, team)}",
                value=f"{prob*100:.1f}%",
            )

    # Championship probability chart
    teams_with_odds = [(t, p) for t, p in sorted_probs if p > 0.001]
    df_champ = pd.DataFrame(teams_with_odds, columns=['Team', 'Probability'])
    df_champ['Probability'] = df_champ['Probability'] * 100
    df_champ['Color'] = df_champ['Team'].map(TEAM_COLORS)
    df_champ['Full Name'] = df_champ['Team'].map(lambda x: f"{x} {TEAM_NAMES.get(x, x)}")

    fig = px.bar(
        df_champ,
        x='Full Name',
        y='Probability',
        color='Team',
        color_discrete_map=TEAM_COLORS,
        title='Championship Probability by Team',
        labels={'Probability': 'Win Championship (%)', 'Full Name': ''},
    )
    fig.update_layout(
        showlegend=False,
        xaxis_tickangle=-45,
        height=500,
        template='plotly_white',
        font=dict(size=12),
    )
    fig.update_traces(
        texttemplate='%{y:.1f}%',
        textposition='outside',
    )
    st.plotly_chart(fig, use_container_width=True)

    # Conference breakdown
    st.subheader("Conference Breakdown")
    col_east, col_west = st.columns(2)

    with col_east:
        st.markdown("### Eastern Conference")
        east_teams = [(t, p) for t, p in sorted_probs
                      if results['standings'].get(t, {}).get('conf') == 'East' and p > 0]
        for team, prob in east_teams:
            bar_width = int(prob * 300)
            st.markdown(
                f"**{team}** {TEAM_NAMES.get(team, '')} — {prob*100:.1f}%"
            )
            st.progress(min(prob * 2.5, 1.0))

    with col_west:
        st.markdown("### Western Conference")
        west_teams = [(t, p) for t, p in sorted_probs
                      if results['standings'].get(t, {}).get('conf') == 'West' and p > 0]
        for team, prob in west_teams:
            st.markdown(
                f"**{team}** {TEAM_NAMES.get(team, '')} — {prob*100:.1f}%"
            )
            st.progress(min(prob * 2.5, 1.0))


# ============================================================
# PAGE 2: Projected Standings
# ============================================================
elif page == "📊 Projected Standings":
    st.header("Projected Final Standings")

    conf_choice = st.radio("Conference", ["Eastern", "Western"], horizontal=True)
    conf_key = "East" if conf_choice == "Eastern" else "West"

    conf_teams = [(t, s) for t, s in results['standings'].items() if s['conf'] == conf_key]
    conf_teams.sort(key=lambda x: x[1]['avg_wins'], reverse=True)

    # Build dataframe
    rows = []
    for rank, (team, stats) in enumerate(conf_teams, 1):
        lo, hi = stats['win_range']
        playoff_prob = results['playoff_probs'].get(team, 0) * 100
        champ_prob = results['championship_probs'].get(team, 0) * 100

        rows.append({
            'Seed': rank,
            'Team': f"{team} {TEAM_NAMES.get(team, '')}",
            'Proj. Wins': round(stats['avg_wins'], 1),
            'Proj. Losses': round(stats['avg_losses'], 1),
            'Win %': f"{stats['avg_win_pct']:.3f}",
            'Win Range (90%)': f"{lo}-{hi}",
            'Playoff %': f"{playoff_prob:.1f}%",
            'Title %': f"{champ_prob:.1f}%",
        })

    df_standings = pd.DataFrame(rows)
    st.dataframe(df_standings, use_container_width=True, hide_index=True)

    # Win distribution chart for selected team
    st.subheader("Win Distribution by Team")
    team_options = [f"{t} {TEAM_NAMES.get(t, '')}" for t, _ in conf_teams]
    selected = st.selectbox("Select team", team_options)
    selected_abbrev = selected.split()[0]

    # Show projected wins range as a gauge chart
    stats = results['standings'][selected_abbrev]
    lo, hi = stats['win_range']

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=stats['avg_wins'],
        title={'text': f"Projected Wins - {selected}"},
        gauge={
            'axis': {'range': [0, 82]},
            'bar': {'color': TEAM_COLORS.get(selected_abbrev, '#1D428A')},
            'steps': [
                {'range': [0, lo], 'color': '#f0f0f0'},
                {'range': [lo, hi], 'color': '#e0e8f0'},
                {'range': [hi, 82], 'color': '#f0f0f0'},
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': stats['avg_wins']
            }
        }
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGE 3: Model Performance
# ============================================================
elif page == "🤖 Model Performance":
    st.header("ML Model Performance")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("CV Accuracy", f"{metrics['cv_accuracy_mean']:.1%}")
    with col2:
        st.metric("CV AUC", f"{metrics['cv_auc_mean']:.3f}")
    with col3:
        st.metric("Training Games", f"{metrics['n_training_games']:,}")
    with col4:
        st.metric("Features", metrics['n_features'])

    st.subheader("Model Details")

    st.markdown("""
    **Algorithm:** XGBoost Gradient Boosted Decision Trees

    **Training Data:** ~7,800 NBA games across 6 seasons (2019-2025)

    **Evaluation:** TimeSeriesSplit cross-validation (5 folds) to prevent data leakage

    **Feature Engineering:** Rolling averages (5, 10, 20 game windows) of:
    - Points, FG%, 3PT%, FT%
    - Rebounds, Assists, Steals, Blocks, Turnovers
    - Offensive/Defensive Rating, Pace
    - Home vs Away differential features
    """)

    # Feature importance chart
    st.subheader("Top Feature Importances")
    feat_imp = metrics['top_features']
    df_feat = pd.DataFrame([
        {'Feature': k.replace('diff_', '').replace('_r', ' (last ').replace('home_', 'Home ').replace('away_', 'Away ') + ' games)', 'Importance': v}
        for k, v in feat_imp.items()
    ])

    fig = px.bar(
        df_feat,
        x='Importance',
        y='Feature',
        orientation='h',
        title='XGBoost Feature Importance (Top 15)',
        color='Importance',
        color_continuous_scale='Blues',
    )
    fig.update_layout(
        height=500,
        template='plotly_white',
        yaxis={'categoryorder': 'total ascending'},
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Technical Architecture")
    st.markdown("""
    ```
    ┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
    │  Raw Game    │────▶│   Feature    │────▶│    XGBoost      │
    │  Data (CSV)  │     │  Engineering │     │   Classifier    │
    └─────────────┘     └──────────────┘     └────────┬────────┘
                                                       │
                                                       ▼
    ┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
    │  Streamlit   │◀────│  Simulation  │◀────│  Win Probability│
    │  Dashboard   │     │   Engine     │     │   Predictions   │
    └─────────────┘     └──────────────┘     └─────────────────┘
    ```
    """)


# ============================================================
# PAGE 4: Playoff Bracket
# ============================================================
elif page == "🏀 Playoff Bracket":
    st.header("Projected Playoff Picture")

    for conf_name, conf_key in [("Eastern Conference", "East"), ("Western Conference", "West")]:
        st.subheader(conf_name)

        conf_teams = [(t, s) for t, s in results['standings'].items() if s['conf'] == conf_key]
        conf_teams.sort(key=lambda x: x[1]['avg_wins'], reverse=True)

        # Top 6 = direct playoff, 7-10 = play-in
        col_playoff, col_playin = st.columns([3, 2])

        with col_playoff:
            st.markdown("**Direct Playoff Qualifiers (Seeds 1-6)**")
            for i, (team, stats) in enumerate(conf_teams[:6], 1):
                prob = results['playoff_probs'].get(team, 0) * 100
                champ = results['championship_probs'].get(team, 0) * 100
                st.markdown(
                    f"**{i}.** {team} {TEAM_NAMES.get(team, '')} — "
                    f"{stats['avg_wins']:.0f}-{stats['avg_losses']:.0f} "
                    f"(🏆 {champ:.1f}%)"
                )

        with col_playin:
            st.markdown("**Play-In Tournament (Seeds 7-10)**")
            for i, (team, stats) in enumerate(conf_teams[6:10], 7):
                prob = results['playoff_probs'].get(team, 0) * 100
                st.markdown(
                    f"**{i}.** {team} {TEAM_NAMES.get(team, '')} — "
                    f"{stats['avg_wins']:.0f}-{stats['avg_losses']:.0f} "
                    f"(Playoff: {prob:.0f}%)"
                )

        # First round matchup preview
        st.markdown("**Projected First Round Matchups:**")
        matchups = [(1, 8), (2, 7), (3, 6), (4, 5)]
        cols = st.columns(4)
        for col, (high, low) in zip(cols, matchups):
            with col:
                if high-1 < len(conf_teams) and low-1 < len(conf_teams):
                    h_team = conf_teams[high-1][0]
                    l_team = conf_teams[low-1][0]
                    h_champ = results['championship_probs'].get(h_team, 0) * 100
                    l_champ = results['championship_probs'].get(l_team, 0) * 100
                    st.markdown(f"""
                    **({high}) vs ({low})**
                    {h_team} {TEAM_NAMES.get(h_team, '')}
                    vs
                    {l_team} {TEAM_NAMES.get(l_team, '')}
                    """)

        st.markdown("---")


# ============================================================
# PAGE 5: Data Explorer
# ============================================================
elif page == "📈 Data Explorer":
    st.header("Training Data Explorer")

    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Games", f"{len(training_data):,}")
    with col2:
        st.metric("Seasons", training_data['season'].nunique())
    with col3:
        st.metric("Home Win Rate", f"{training_data['home_win'].mean():.1%}")

    # Points distribution
    st.subheader("Points Distribution")
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=training_data['home_pts'],
        name='Home Team',
        opacity=0.7,
        marker_color='#1D428A'
    ))
    fig.add_trace(go.Histogram(
        x=training_data['away_pts'],
        name='Away Team',
        opacity=0.7,
        marker_color='#C8102E'
    ))
    fig.update_layout(
        barmode='overlay',
        title='Points Scored Distribution',
        xaxis_title='Points',
        yaxis_title='Frequency',
        template='plotly_white',
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Team performance
    st.subheader("Team Win Rates by Season")
    season_filter = st.selectbox("Season", sorted(training_data['season'].unique(), reverse=True))
    season_data = training_data[training_data['season'] == season_filter]

    # Calculate win rates per team
    team_wins = {}
    for team in TEAM_NAMES.keys():
        home_wins = len(season_data[(season_data['home_team'] == team) & (season_data['home_win'] == 1)])
        away_wins = len(season_data[(season_data['away_team'] == team) & (season_data['home_win'] == 0)])
        home_games = len(season_data[season_data['home_team'] == team])
        away_games = len(season_data[season_data['away_team'] == team])
        total = home_games + away_games
        if total > 0:
            team_wins[team] = (home_wins + away_wins) / total

    if team_wins:
        df_wins = pd.DataFrame([
            {'Team': t, 'Win Rate': w, 'Color': TEAM_COLORS.get(t, '#666')}
            for t, w in sorted(team_wins.items(), key=lambda x: x[1], reverse=True)
        ])

        fig = px.bar(
            df_wins,
            x='Team',
            y='Win Rate',
            color='Team',
            color_discrete_map=TEAM_COLORS,
            title=f'Team Win Rates - {season_filter}',
        )
        fig.update_layout(
            showlegend=False,
            xaxis_tickangle=-45,
            height=400,
            template='plotly_white',
        )
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="50%")
        st.plotly_chart(fig, use_container_width=True)

    # Raw data sample
    st.subheader("Raw Data Sample")
    st.dataframe(training_data.head(20), use_container_width=True)


# --- Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 0.85rem;'>
    Built with Python • XGBoost • Streamlit • Plotly<br>
    Monte Carlo Simulation (1,000 iterations) • 2025-26 NBA Season<br>
    <em>Data as of March 2, 2026</em>
</div>
""", unsafe_allow_html=True)
