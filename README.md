# Prosumer Energy Advisor - Enhanced Version

A machine learning-based energy management system for prosumers (consumers with solar PV generation) that provides intelligent action recommendations to optimize energy costs and usage.

> **🚀 NEW**: Advanced forecasting capabilities with [Darts](https://unit8co.github.io/darts/), [Chronos](https://pypi.org/project/chronos-forecasting/), and [HTW Berlin Solar Calculator](https://solar.htw-berlin.de/rechner/) - see [ADVANCED_FORECASTING.md](ADVANCED_FORECASTING.md)

## 🎯 Overview

This system analyzes your solar PV generation, household load, and electricity prices to recommend optimal actions such as:
- **Use PV Direct**: Consume solar energy locally
- **Sell to Grid**: Export excess solar during high prices
- **Charge Battery**: Store energy for later use
- **Charge EV**: Charge electric vehicle during cheap periods
- **Idle**: No special action needed

## ✨ Key Features

### Core Capabilities
- **Smart Action Recommendations**: ML-based decision making for energy optimization
- **Real-time Price Integration**: ENTSO-E API for day-ahead electricity prices
- **Weather-aware Forecasting**: Open-Meteo integration for solar and weather data
- **Multiple Operation Modes**: Classification (Mode A) and Optimization (Mode B)
- **Comprehensive Visualization**: Energy flow analysis and action distribution plots

### 🚀 Professional ML Tools (NEW!)
- **Interactive Dashboard**: Streamlit web app for real-time monitoring
- **Data Profiling**: Automated EDA with ydata-profiling
- **Experiment Tracking**: MLOps with Weights & Biases
- **Hyperparameter Optimization**: Automated tuning with Optuna
- **Advanced Forecasting**: Darts, Chronos, HTW Solar integration

See [PROFESSIONAL_TOOLS.md](PROFESSIONAL_TOOLS.md) for complete guide.

### Advanced Forecasting (Optional)
- **Darts Integration**: 40+ professional forecasting models (ARIMA, Prophet, N-BEATS, TFT)
- **Chronos Foundation Model**: Zero-shot forecasting with Amazon's pretrained model
- **HTW Solar Calculator**: Accurate PV yield calculations for German locations
- **Enhanced Open-Meteo**: Advanced solar radiation and weather forecasts

See [ADVANCED_FORECASTING.md](ADVANCED_FORECASTING.md) for integration guide.

## ✨ Recent Enhancements

### Code Quality Improvements
- ✅ **Type Hints**: Added comprehensive type annotations to all functions
- ✅ **Documentation**: Added detailed docstrings with Args, Returns, and Raises sections
- ✅ **Error Handling**: Improved exception handling with specific error types
- ✅ **Input Validation**: Added validation for configuration parameters and data
- ✅ **Code Organization**: Better module-level documentation and structure

### Performance Optimizations
- ✅ **Vectorized Operations**: Replaced loops with pandas vectorized operations in `features.py`
- ✅ **Efficient Data Processing**: Optimized feature engineering and action labeling
- ✅ **Parallel Processing**: Added `n_jobs=-1` to RandomForest for multi-core training

### User Experience
- ✅ **Better Logging**: Clear progress messages and status updates
- ✅ **Informative Errors**: Descriptive error messages with actionable guidance
- ✅ **CLI Improvements**: Enhanced command-line interfaces with help text
- ✅ **Debug Code Removal**: Removed debug print statements from production code

### Code Maintainability
- ✅ **Constants**: Extracted magic numbers to named constants
- ✅ **Configuration Access**: Consistent config value extraction patterns
- ✅ **Modular Design**: Clear separation of concerns across modules

## 📁 Project Structure

```
Prosumer_advisor/
├── config.yaml              # Main configuration file
├── requirements.txt         # Python dependencies
├── model_meta.yaml         # Model metadata (generated)
├── model.joblib            # Trained model (generated)
│
├── entsoe_prices.py        # ENTSO-E API client for electricity prices
├── fetch_data.py           # Data fetching (PV, prices, weather, load)
├── features.py             # Feature engineering and oracle labeling
├── train.py                # Model training and evaluation
├── predict.py              # Prediction and recommendation generation
├── predict_tomorrow.py     # Tomorrow's predictions script
├── run_pipeline.py         # Complete training pipeline
├── visualize.py            # Visualization generation
├── opt_cost_oracle.py      # Cost-based optimization oracle
│
├── cache/                  # Cached API responses
└── outputs/                # Generated outputs
    ├── *.png              # Visualization plots
    └── *.csv              # Prediction results
```

## 🚀 Quick Start

### 1. Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Edit `config.yaml` to set your location and system parameters:

```yaml
location:
  lat: 52.52          # Your latitude
  lon: 13.405         # Your longitude
  tz: Europe/Berlin   # Your timezone
  start_date: 2024-06-01
  days: 100

pv_system:
  peak_power_kw: 5.0  # Your PV system size
  tilt_deg: 30
  azimuth_deg: 0

grid_price:
  zone: DE_LU
  token: ""           # Optional: ENTSO-E API token
```

### 3. Set ENTSO-E API Token (Optional)

For real electricity prices:

```bash
export ENTSOE_API_TOKEN="your-token-here"
```

Get your free token at: https://transparency.entsoe.eu/

### 4. Run Training Pipeline

```bash
python run_pipeline.py
```

This will:
1. Fetch historical data
2. Engineer features
3. Train the model
4. Evaluate performance
5. Generate visualizations

### 5. Generate Tomorrow's Predictions

```bash
python predict_tomorrow.py
```

Output saved to: `outputs/predictions_YYYY-MM-DD.csv`

## 📊 Usage Examples

### Predict for Multiple Days

```bash
python predict.py 7  # Predict next 7 days
```

### Fetch ENTSO-E Prices

```bash
python -m entsoe_prices --start 2024-06-01 --end 2024-06-02 --out prices.csv
```

### Custom Training

```python
import yaml
from fetch_data import get_data
from features import prepare_dataset
from train import train_model

# Load config
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

# Fetch and prepare data
df_raw = get_data(cfg)
df = prepare_dataset(df_raw, cfg)

# Train model
model, features, actions = train_model(df, cfg)
```

## 🔧 Configuration Options

### Model Settings

```yaml
model:
  mode: A                    # A=Classification, B=Optimization
  algorithm: RandomForest    # DecisionTree or RandomForest
  rf_n_estimators: 100      # Number of trees
  random_seed: 42
```

### Oracle Settings

```yaml
oracle:
  price_high_quantile: 0.6   # High price threshold (60th percentile)
  price_low_quantile: 0.3    # Low price threshold (30th percentile)
  ev_night_start: 0          # EV charging window start (midnight)
  ev_night_end: 6            # EV charging window end (6 AM)
  ev_enabled: true
  weather_precharge_enabled: true
  battery_capacity_kwh: 10
  ev_capacity_kwh: 50
```

## 📈 Features

### Input Features
- **PV Surplus**: PV generation - load
- **Price**: Electricity price (EUR/kWh)
- **Time Features**: Hour of day, day of week
- **Weather**: Temperature, cloud cover, irradiance
- **Lag Features**: Previous hours' load and PV
- **Rolling Averages**: 3-hour load, 6-hour PV
- **Forecast Flags**: Next-day low PV indicator

### Action Labels
Generated by rule-based oracle considering:
- PV surplus availability
- Price levels (high/low quantiles)
- Time of day (EV charging windows)
- Weather forecasts (pre-charging logic)

## 🎨 Visualizations

The pipeline generates three plots:

1. **PV/Load Overview**: Time series with action shading
2. **Action Distribution**: Bar chart of action frequencies
3. **Energy Flow**: Daily PV usage and grid import breakdown

## 🧪 Model Evaluation

The training pipeline provides:
- Classification report (precision, recall, F1-score)
- Confusion matrix
- Overall accuracy
- Per-action performance metrics

## 🔍 API Reference

### Main Functions

#### `get_data(cfg) -> pd.DataFrame`
Fetch and merge all data sources (PV, prices, weather, load).

#### `prepare_dataset(df_raw, cfg) -> pd.DataFrame`
Engineer features and label actions.

#### `train_model(df, cfg) -> Tuple[model, features, actions]`
Train classification model.

#### `recommend_actions_for_df(df, model, cfg, features, actions) -> pd.DataFrame`
Generate action recommendations with explanations.

## 🐛 Troubleshooting

### ENTSO-E API Errors

If you get "No data" errors:
- Check your API token is valid
- Verify the bidding zone code (e.g., `10Y1001A1001A82H` for DE-LU)
- Ensure dates are in correct format (YYYY-MM-DD)

### Missing Data

The system has fallbacks:
- PVGIS → Open-Meteo → Synthetic sine (for PV)
- ENTSO-E → Synthetic TOU prices (for prices)
- Open-Meteo → Empty DataFrame (for weather)

### Performance Issues

For large datasets:
- Reduce `days` in config
- Use `algorithm: DecisionTree` instead of RandomForest
- Disable weather features if not needed

## 📝 Development

### Code Style
- Type hints on all functions
- Docstrings with Google style
- Constants in UPPER_CASE
- Descriptive variable names

### Testing
```bash
# Run training pipeline
python run_pipeline.py

# Verify predictions
python predict_tomorrow.py

# Check visualizations in outputs/
```

## 🤝 Contributing

Enhancements made:
1. Added comprehensive type hints
2. Improved error handling and validation
3. Optimized performance with vectorization
4. Enhanced documentation
5. Better user feedback and logging
6. Removed debug code
7. Improved code organization

## 🎯 Professional ML Tools

### Interactive Dashboard
```bash
streamlit run streamlit_app.py
```

Features:
- Real-time energy monitoring
- Tomorrow's forecast generation
- Data profiling integration
- Hyperparameter optimization UI

### Data Profiling
```python
from data_profiling import generate_profile_report
generate_profile_report(df)
# Opens: outputs/data_profile.html
```

### Experiment Tracking
```python
from experiment_tracking import WandbExperimentTracker
tracker = WandbExperimentTracker()
tracker.start_run(run_name="experiment_1")
tracker.log_metrics({"accuracy": 0.85})
```

### Hyperparameter Optimization
```python
from hyperparameter_tuning import optimize_random_forest
result = optimize_random_forest(X_train, y_train, n_trials=50)
# Best parameters automatically found!
```

**Full documentation**: [PROFESSIONAL_TOOLS.md](PROFESSIONAL_TOOLS.md)

## 🔮 Advanced Forecasting

For production deployments or research projects, consider integrating advanced forecasting:

### Quick Start with Darts
```python
from forecasting_darts import DartsForecaster

# Train Prophet model
forecaster = DartsForecaster(model_type="prophet")
forecaster.train(df_train, target_col="load_kW")

# Forecast next 24 hours
forecast = forecaster.predict(n_hours=24)
```

### Quick Start with Chronos (Zero-Shot)
```python
from forecasting_chronos import ChronosForecaster

# No training needed!
forecaster = ChronosForecaster(model_size="small")
forecast = forecaster.forecast(df["load_kW"], prediction_length=24)
```

**Full documentation**: [ADVANCED_FORECASTING.md](ADVANCED_FORECASTING.md)

## 📄 License

This project is for educational and research purposes.

## 🙏 Acknowledgments

### Data & APIs
- **PVGIS** - Solar irradiance data
- **ENTSO-E** - Electricity price data
- **Open-Meteo** - Weather forecasts ([GitHub](https://github.com/open-meteo/open-meteo))
- **HTW Berlin** - Solar calculator ([Link](https://solar.htw-berlin.de/rechner/))

### ML & Forecasting
- **scikit-learn** - ML algorithms
- **Darts** - Time series forecasting ([Docs](https://unit8co.github.io/darts/))
- **Chronos** - Foundation model ([PyPI](https://pypi.org/project/chronos-forecasting/))

### Professional Tools
- **Streamlit** - Interactive dashboards ([Docs](https://docs.streamlit.io/))
- **ydata-profiling** - Automated EDA ([Docs](https://docs.profiling.ydata.ai/))
- **Weights & Biases** - Experiment tracking ([Docs](https://docs.wandb.ai/))
- **Optuna** - Hyperparameter optimization ([Docs](https://optuna.readthedocs.io/))

## 📚 Documentation

- **[README.md](README.md)** - This file (quick start)
- **[PROFESSIONAL_TOOLS.md](PROFESSIONAL_TOOLS.md)** - ML tools guide
- **[ADVANCED_FORECASTING.md](ADVANCED_FORECASTING.md)** - Forecasting integration
- **[CHANGELOG.md](CHANGELOG.md)** - Version history

## 🔗 Related Projects & Resources

- **Darts Examples**: https://github.com/unit8co/darts/tree/master/examples
- **Open-Meteo API Docs**: https://open-meteo.com/en/docs
- **ENTSO-E Transparency Platform**: https://transparency.entsoe.eu/
- **Chronos Paper**: https://arxiv.org/abs/2403.07815
- **Streamlit Gallery**: https://streamlit.io/gallery
- **Optuna Examples**: https://github.com/optuna/optuna-examples

---

**Version**: 2.0 (Enhanced with Professional ML Tools)
**Last Updated**: 2025-11-08# UrbanTechnology py PyCharmMiscProject at 21:36:06
