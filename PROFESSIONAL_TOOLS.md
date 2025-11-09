# Professional ML/Data Science Tools Guide

This guide covers the professional tools integrated into the Prosumer Energy Advisor project for production-grade ML workflows.

## 📋 Table of Contents

1. [Data Profiling (ydata-profiling)](#1-data-profiling)
2. [Experiment Tracking (wandb)](#2-experiment-tracking)
3. [Hyperparameter Optimization (Optuna)](#3-hyperparameter-optimization)
4. [Interactive Dashboard (Streamlit)](#4-interactive-dashboard)
5. [Time Series Forecasting (Darts)](#5-time-series-forecasting)
6. [Complete Workflow Example](#6-complete-workflow)

---

## 1️⃣ Data Profiling

### Overview
**ydata-profiling** (formerly pandas-profiling) generates comprehensive data analysis reports automatically.

### Installation
```bash
pip install ydata-profiling
```

### Usage

#### Quick Profile
```python
from data_profiling import generate_profile_report
from fetch_data import get_data
import yaml

# Load config and data
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

df = get_data(cfg)

# Generate report
generate_profile_report(df)
# Opens: outputs/data_profile.html
```

#### Command Line
```bash
python data_profiling.py
```

### Features
- **Univariate Analysis**: Distribution, missing values, unique values
- **Correlations**: Pearson, Spearman correlations
- **Missing Data**: Heatmaps and dendrograms
- **Interactions**: Scatter plots between variables
- **Alerts**: Data quality warnings

### Output
- HTML report with interactive visualizations
- Exportable to JSON for programmatic access
- Comparison reports for train/test splits

---

## 2️⃣ Experiment Tracking

### Overview
**Weights & Biases (wandb)** provides MLOps capabilities for tracking experiments, models, and metrics.

### Installation
```bash
pip install wandb
wandb login  # One-time setup
```

### Usage

#### Basic Tracking
```python
from experiment_tracking import WandbExperimentTracker

# Initialize
tracker = WandbExperimentTracker(
    project_name="prosumer-energy-advisor",
    config=cfg
)

# Start run
tracker.start_run(
    run_name="experiment_rf_v1",
    tags=["training", "random-forest"],
    notes="Testing new features"
)

# Log metrics
tracker.log_metrics({
    "accuracy": 0.85,
    "f1_score": 0.82,
    "train_loss": 0.15
})

# Log model
tracker.log_model("model.joblib", "prosumer_model_v1")

# Finish
tracker.finish()
```

#### Integrated Training
```python
from experiment_tracking import train_with_tracking

# Automatically tracks entire training process
train_with_tracking(df_train, cfg)
```

### Features
- **Automatic Logging**: Metrics, hyperparameters, system info
- **Visualizations**: Real-time charts and plots
- **Model Versioning**: Track and compare model versions
- **Collaboration**: Share experiments with team
- **Artifacts**: Store datasets, models, predictions

### Dashboard
Access at: https://wandb.ai/your-username/prosumer-energy-advisor

---

## 3️⃣ Hyperparameter Optimization

### Overview
**Optuna** provides efficient hyperparameter optimization using Bayesian methods.

### Installation
```bash
pip install optuna
```

### Usage

#### Optimize RandomForest
```python
from hyperparameter_tuning import optimize_random_forest

# Prepare data
X_train = df[feature_cols].fillna(0)
y_train = df['recommended_action'].map(action_mapping)

# Optimize
result = optimize_random_forest(
    X_train, y_train,
    n_trials=50,  # Number of trials
    cv_folds=3    # Cross-validation folds
)

# Best parameters
print(result['params'])
# {'n_estimators': 150, 'max_depth': 12, ...}

# Use optimized parameters
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(**result['params'])
model.fit(X_train, y_train)
```

#### Visualize Results
```python
from hyperparameter_tuning import visualize_optimization

visualize_optimization(result['study'])
# Creates: outputs/optimization_history.html
#          outputs/param_importances.html
#          outputs/parallel_coordinate.html
```

#### Command Line
```bash
python hyperparameter_tuning.py
```

### Features
- **Efficient Search**: Bayesian optimization (TPE sampler)
- **Pruning**: Early stopping of unpromising trials
- **Parallel Execution**: Multi-process optimization
- **Visualization**: Interactive plots of optimization process
- **Study Management**: Save and resume optimization

### Optimization Strategies

#### Grid Search Alternative
```python
# Optuna is smarter than grid search
# Automatically focuses on promising regions
result = optimize_random_forest(X, y, n_trials=100)
```

#### Multi-Objective Optimization
```python
# Optimize for both accuracy and speed
def objective(trial):
    params = {...}
    model = RandomForestClassifier(**params)
    
    # Measure both metrics
    accuracy = cross_val_score(model, X, y).mean()
    train_time = measure_training_time(model, X, y)
    
    return accuracy, -train_time  # Maximize accuracy, minimize time
```

---

## 4️⃣ Interactive Dashboard

### Overview
**Streamlit** creates interactive web applications for ML models with minimal code.

### Installation
```bash
pip install streamlit plotly
```

### Usage

#### Launch Dashboard
```bash
streamlit run streamlit_app.py
```

Opens browser at: http://localhost:8501

### Features

#### 📊 Dashboard Tab
- Real-time metrics (PV generation, load, prices)
- Action distribution visualization
- Time series plots
- Recent recommendations table

#### 🔮 Predictions Tab
- Generate tomorrow's 24-hour forecast
- Interactive forecast visualization
- Hourly action recommendations

#### 📈 Analytics Tab
- Generate data profiling reports
- Run hyperparameter optimization
- View optimization results

#### ℹ️ About Tab
- Project documentation
- Feature descriptions
- Technology stack

### Customization

#### Add Custom Metrics
```python
# In streamlit_app.py
st.metric(
    "Custom Metric",
    f"{value:.2f}",
    delta=f"{change:.1f}%"
)
```

#### Add New Visualizations
```python
import plotly.express as px

fig = px.line(df, x='time', y='pv_kW', title='PV Generation')
st.plotly_chart(fig, use_container_width=True)
```

### Deployment

#### Local Network
```bash
streamlit run streamlit_app.py --server.address 0.0.0.0
```

#### Cloud Deployment
- **Streamlit Cloud**: Free hosting for public repos
- **Heroku**: `Procfile` with `web: streamlit run streamlit_app.py`
- **Docker**: Containerize with Dockerfile

---

## 5️⃣ Time Series Forecasting

### Overview
**Darts** provides 40+ forecasting models with unified API.

See [ADVANCED_FORECASTING.md](ADVANCED_FORECASTING.md) for complete guide.

### Quick Start
```bash
pip install "darts[prophet,lightgbm]"
```

```python
from darts import TimeSeries
from darts.models import Prophet

# Convert to Darts format
series = TimeSeries.from_dataframe(df, value_cols='load_kW')

# Train model
model = Prophet()
model.fit(series)

# Forecast
forecast = model.predict(n=24)
```

---

## 6️⃣ Complete Workflow Example

### End-to-End ML Pipeline with All Tools

```python
"""Complete professional ML workflow."""

import yaml
from fetch_data import get_data
from features import prepare_dataset
from data_profiling import generate_profile_report
from experiment_tracking import WandbExperimentTracker
from hyperparameter_tuning import optimize_random_forest
from train import train_model
from sklearn.ensemble import RandomForestClassifier

# 1. Load configuration
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

# 2. Fetch and prepare data
print("Step 1: Fetching data...")
df_raw = get_data(cfg)
df = prepare_dataset(df_raw, cfg)

# 3. Data profiling
print("Step 2: Generating data profile...")
generate_profile_report(df, output_file="outputs/profile.html")

# 4. Initialize experiment tracking
print("Step 3: Setting up experiment tracking...")
tracker = WandbExperimentTracker(config=cfg)
tracker.start_run(
    run_name="optimized_model_v1",
    tags=["production", "optimized"]
)

# 5. Prepare features
feature_cols = [
    'pv_surplus', 'price_eur_per_kwh', 'hour', 'dow',
    'temp_C', 'cloudcover_pct', 'next_day_pv_flag'
]
feature_cols = [c for c in feature_cols if c in df.columns]

action_mapping = {a: i for i, a in enumerate(df['recommended_action'].unique())}
X = df[feature_cols].fillna(0)
y = df['recommended_action'].map(action_mapping)

# 6. Hyperparameter optimization
print("Step 4: Optimizing hyperparameters...")
opt_result = optimize_random_forest(X, y, n_trials=50)

# Log to wandb
tracker.log_metrics({
    "optimization_score": opt_result['score'],
    "n_trials": 50
})

# 7. Train final model with optimized parameters
print("Step 5: Training final model...")
model = RandomForestClassifier(**opt_result['params'])
model.fit(X, y)

# 8. Evaluate and log
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(model, X, y, cv=5)

tracker.log_metrics({
    "cv_mean_accuracy": cv_scores.mean(),
    "cv_std_accuracy": cv_scores.std(),
    "final_train_accuracy": model.score(X, y)
})

# 9. Save model
import joblib
joblib.dump(model, "model_optimized.joblib")
tracker.log_model("model_optimized.joblib", "optimized_model")

# 10. Finish
tracker.finish()

print("\n✓ Complete workflow finished!")
print("  - Data profile: outputs/profile.html")
print("  - Optimized model: model_optimized.joblib")
print("  - Experiment: https://wandb.ai")
```

### Run Complete Pipeline
```bash
# 1. Data profiling
python data_profiling.py

# 2. Hyperparameter optimization
python hyperparameter_tuning.py

# 3. Training with tracking
python experiment_tracking.py

# 4. Launch dashboard
streamlit run streamlit_app.py
```

---

## 🎯 Best Practices

### 1. Data Profiling
- ✅ Profile raw data before preprocessing
- ✅ Profile after feature engineering
- ✅ Compare train/test distributions
- ✅ Check for data drift over time

### 2. Experiment Tracking
- ✅ Track every experiment (even failed ones)
- ✅ Use meaningful run names and tags
- ✅ Log hyperparameters and metrics
- ✅ Version your datasets and models
- ✅ Add notes explaining experiment goals

### 3. Hyperparameter Optimization
- ✅ Start with few trials (10-20) for quick feedback
- ✅ Use cross-validation for robust estimates
- ✅ Visualize optimization progress
- ✅ Save study for later analysis
- ✅ Consider multi-objective optimization

### 4. Interactive Dashboards
- ✅ Keep UI simple and intuitive
- ✅ Add loading indicators for long operations
- ✅ Provide clear error messages
- ✅ Cache expensive computations
- ✅ Make dashboards responsive

---

## 📊 Comparison Matrix

| Tool | Purpose | When to Use | Output |
|------|---------|-------------|--------|
| **ydata-profiling** | Data exploration | Start of project, data quality checks | HTML reports |
| **wandb** | Experiment tracking | Every training run | Online dashboard |
| **Optuna** | Hyperparameter tuning | Model optimization | Best parameters |
| **Streamlit** | Interactive demos | Stakeholder presentations | Web app |
| **Darts** | Time series forecasting | Advanced predictions | Forecasts |

---

## 🚀 Quick Commands Reference

```bash
# Data profiling
python data_profiling.py

# Hyperparameter optimization
python hyperparameter_tuning.py

# Experiment tracking
python experiment_tracking.py

# Interactive dashboard
streamlit run streamlit_app.py

# Complete pipeline
python run_pipeline.py

# Tomorrow's predictions
python predict_tomorrow.py
```

---

## 📚 Additional Resources

- **ydata-profiling**: https://docs.profiling.ydata.ai/
- **Weights & Biases**: https://docs.wandb.ai/
- **Optuna**: https://optuna.readthedocs.io/
- **Streamlit**: https://docs.streamlit.io/
- **Darts**: https://unit8co.github.io/darts/

---

## 🎓 Learning Path

1. **Beginner**: Start with Streamlit dashboard
2. **Intermediate**: Add data profiling and experiment tracking
3. **Advanced**: Implement hyperparameter optimization
4. **Expert**: Integrate advanced forecasting with Darts/Chronos

---

**Version**: 2.0  
**Last Updated**: 2025-11-08