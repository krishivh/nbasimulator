# Setup Instructions

## Prerequisites
- Python 3.10+
- pip
- A Kaggle account (for the dataset)

## Step-by-Step Setup

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/nba-season-simulator.git
cd nba-season-simulator
```

### 2. Create virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
# or: venv\Scripts\activate  # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Kaggle dataset
```bash
pip install kagglehub
python -c "import kagglehub; print(kagglehub.dataset_download('eoinamoore/historical-nba-data-and-player-box-scores'))"
```
Note the path it prints — you'll use it in the next step.

### 5. Run the pipeline
```bash
# With Kaggle data:
python pipeline.py --kaggle-path /path/kagglehub/printed

# Or with generated data (if Kaggle isn't available):
python src/generate_training_data.py
python pipeline.py --use-generated
```

### 6. Run the simulation
```bash
python src/simulate.py
```

### 7. Launch Streamlit dashboard
```bash
streamlit run app.py
```

## Outputs

After running the pipeline:
- `models/xgb_model.pkl` — Trained XGBoost model
- `models/feature_cols.pkl` — Feature column names
- `models/metrics.json` — Model evaluation metrics
- `exports/team_profiles.json` — Team strength ratings for the React demo
- `data/simulation_results.json` — Monte Carlo simulation results
