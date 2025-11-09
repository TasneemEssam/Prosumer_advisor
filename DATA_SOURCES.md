# 📊 Data Sources & Dataset Information

## Overview

**Important**: This project does **NOT** use a pre-existing static dataset. Instead, it **dynamically fetches and generates data** from multiple sources in real-time.

---

## 🔄 Data Generation Process

### How It Works

When you run the pipeline (`python run_pipeline.py`), the system:

1. **Reads configuration** from [`config.yaml`](config.yaml:1)
2. **Fetches data** from multiple APIs
3. **Generates synthetic data** where needed
4. **Merges everything** into a unified DataFrame
5. **Saves results** to `outputs/` directory

---

## 📡 Data Sources

### 1. **PV Generation Data** ☀️

**Primary Source**: [PVGIS API](https://re.jrc.ec.europa.eu/api/v5_3/seriescalc)
- **What**: Historical solar irradiance and PV power output
- **Coverage**: Europe, Africa, Asia, Americas
- **Resolution**: Hourly
- **Data**: Climate-based PV generation estimates
- **Free**: Yes, no API key required

**Fallback #1**: [Open-Meteo API](https://api.open-meteo.com/v1/forecast)
- **What**: Solar radiation forecasts
- **Coverage**: Global
- **Resolution**: Hourly
- **Data**: Shortwave radiation (W/m²)
- **Free**: Yes

**Fallback #2**: Synthetic Sine Wave
- **What**: Simple mathematical model
- **Formula**: `PV = peak_power * sin(hour_angle)` for daylight hours
- **Use**: Last resort if APIs fail

**Code**: [`fetch_data.py:47-199`](fetch_data.py:47)

---

### 2. **Electricity Prices** 💰

**Primary Source**: [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/)
- **What**: Day-ahead electricity market prices
- **Coverage**: European countries
- **Resolution**: Hourly
- **Data**: EUR/MWh (converted to EUR/kWh)
- **API Key**: Required (free registration)
- **Get Token**: https://transparency.entsoe.eu/

**How to Set Token**:
```bash
# Option 1: Environment variable
export ENTSOE_API_TOKEN="your-token-here"

# Option 2: In config.yaml
grid_price:
  token: "your-token-here"
```

**Fallback**: Synthetic Time-of-Use (TOU) Pricing
- **What**: Realistic price patterns
- **Pattern**:
  - Night (0-6h): €0.14/kWh
  - Morning (6-16h): €0.18/kWh
  - Peak (16-21h): €0.30/kWh
  - Evening (21-24h): €0.22/kWh
  - Weekend: 10% discount
- **Noise**: ±€0.005/kWh random variation

**Code**: [`fetch_data.py:203-316`](fetch_data.py:203)

---

### 3. **Weather Data** 🌤️

**Source**: [Open-Meteo API](https://open-meteo.com/)
- **What**: Weather forecasts and historical data
- **Coverage**: Global
- **Resolution**: Hourly
- **Data**:
  - Temperature (°C)
  - Cloud cover (%)
  - Solar radiation (W/m²)
  - Wind speed (m/s)
- **Free**: Yes, no API key required
- **APIs Used**:
  - Forecast: `https://api.open-meteo.com/v1/forecast`
  - Historical: `https://archive-api.open-meteo.com/v1/archive`

**Code**: [`fetch_data.py:318-379`](fetch_data.py:318)

---

### 4. **Household Load** 🏠

**Source**: **Synthetic Generation** (No external API)

**Method**: Realistic load profile generator
- **Pattern**:
  - Night (0-6h): 0.4 kW baseline
  - Morning peak (6-9h): 1.4 kW (cooking, showers)
  - Daytime (9-17h): 0.6 kW (appliances)
  - Evening peak (17-21h): 1.9 kW (cooking, TV, lighting)
  - Late evening (21-24h): 0.9 kW
- **Weekend**: +0.3 kW during 10-17h
- **Variation**: ±10% random noise
- **Seed**: Configurable for reproducibility

**Code**: [`fetch_data.py:443-486`](fetch_data.py:443)

---

## 📋 Configuration

### Location Settings

```yaml
location:
  lat: 52.52          # Berlin latitude
  lon: 13.405         # Berlin longitude
  tz: Europe/Berlin   # Timezone
  start_date: 2024-06-01
  days: 100           # Number of days to fetch
```

### PV System Settings

```yaml
pv_system:
  peak_power_kw: 5.0  # System size
  tilt_deg: 30        # Panel tilt
  azimuth_deg: 0      # 0=South, 90=West, -90=East
  loss_percent: 14    # System losses
```

### Price Settings

```yaml
grid_price:
  zone: DE_LU         # Germany/Luxembourg
  token: ""           # ENTSO-E API token
  fallback_price_eur_per_kwh: 0.20
```

---

## 📊 Generated Dataset Structure

### Output DataFrame

When you run `get_data(cfg)`, you get a DataFrame with:

| Column | Type | Source | Description |
|--------|------|--------|-------------|
| `index` | DatetimeIndex | - | Hourly timestamps (timezone-aware) |
| `pv_kW` | float | PVGIS/Open-Meteo | PV generation (kW) |
| `load_kW` | float | Synthetic | Household consumption (kW) |
| `price_eur_per_kwh` | float | ENTSO-E/Synthetic | Electricity price (€/kWh) |
| `temp_C` | float | Open-Meteo | Temperature (°C) |
| `cloudcover_pct` | float | Open-Meteo | Cloud cover (%) |
| `irradiance_wm2` | float | Open-Meteo | Solar radiation (W/m²) |

### Example Data

```python
import yaml
from fetch_data import get_data

# Load config
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

# Fetch data
df = get_data(cfg)

print(df.head())
```

**Output**:
```
                           pv_kW  load_kW  price_eur_per_kwh  temp_C  cloudcover_pct  irradiance_wm2
2024-06-01 00:00:00+02:00   0.00     0.42              0.140    15.2            45.0             0.0
2024-06-01 01:00:00+02:00   0.00     0.38              0.138    14.8            42.0             0.0
2024-06-01 02:00:00+02:00   0.00     0.41              0.142    14.5            40.0             0.0
2024-06-01 03:00:00+02:00   0.00     0.39              0.139    14.2            38.0             0.0
2024-06-01 04:00:00+02:00   0.00     0.43              0.141    14.0            35.0             0.0
```

---

## 🗂️ Cached Data

### Cache Directory

**Location**: `Prosumer_advisor/cache/`

**Purpose**: Store API responses to avoid repeated requests

**Files**:
- `pvgis_*.json` - PVGIS responses
- `entsoe_*.xml` - ENTSO-E price data
- `openmeteo_*.json` - Weather data

**Benefits**:
- Faster subsequent runs
- Reduced API load
- Offline development

---

## 💾 Saved Outputs

### Output Directory

**Location**: `Prosumer_advisor/outputs/`

**Generated Files**:

1. **`prices_used.csv`**
   - Electricity prices used in training
   - Format: timestamp, price_eur_per_kwh

2. **`predictions_YYYY-MM-DD.csv`**
   - Daily predictions
   - Columns: time, pv_kW, load_kW, price, action, reason

3. **Visualization PNGs**:
   - `pv_load_actions.png` - PV/Load overview
   - `action_frequency.png` - Action distribution
   - `grid_import.png` - Energy flow
   - `pv_usage.png` - PV utilization

4. **Model Files**:
   - `model.joblib` - Trained ML model
   - `model_meta.yaml` - Model metadata

---

## 🔄 Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Configuration (config.yaml)              │
│  - Location (lat, lon, timezone)                            │
│  - PV System (size, tilt, azimuth)                          │
│  - Date Range (start_date, days)                            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    fetch_data.get_data()                     │
└─────────────────────────────────────────────────────────────┘
         │                │                │                │
         ▼                ▼                ▼                ▼
    ┌────────┐      ┌────────┐      ┌────────┐      ┌────────┐
    │ PVGIS  │      │ENTSO-E │      │ Open-  │      │Synth.  │
    │  API   │      │  API   │      │ Meteo  │      │ Load   │
    └────────┘      └────────┘      └────────┘      └────────┘
         │                │                │                │
         ▼                ▼                ▼                ▼
    ┌────────┐      ┌────────┐      ┌────────┐      ┌────────┐
    │ pv_kW  │      │ price  │      │weather │      │load_kW │
    └────────┘      └────────┘      └────────┘      └────────┘
         │                │                │                │
         └────────────────┴────────────────┴────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  Merged DataFrame    │
              │  (hourly, tz-aware)  │
              └──────────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  Feature Engineering │
              │  (features.py)       │
              └──────────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │   Model Training     │
              │   (train.py)         │
              └──────────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  Predictions &       │
              │  Recommendations     │
              └──────────────────────┘
```

---

## 🚀 Quick Start Examples

### Example 1: Fetch 7 Days of Data

```python
import yaml
from fetch_data import get_data

cfg = {
    "location": {
        "lat": 52.52,
        "lon": 13.40,
        "tz": "Europe/Berlin",
        "start_date": "2024-06-01",
        "days": 7
    },
    "pv_system": {
        "peak_power_kw": 5.0,
        "tilt_deg": 30,
        "azimuth_deg": 0,
        "loss_percent": 14
    },
    "grid_price": {
        "zone": "DE_LU",
        "token": ""  # Will use fallback prices
    }
}

df = get_data(cfg)
print(f"Fetched {len(df)} hours of data")
print(df.describe())
```

### Example 2: Use Real ENTSO-E Prices

```python
import os
os.environ["ENTSOE_API_TOKEN"] = "your-token-here"

# Then run normally
df = get_data(cfg)
```

### Example 3: Different Location (Munich)

```python
cfg["location"]["lat"] = 48.14
cfg["location"]["lon"] = 11.58
cfg["location"]["tz"] = "Europe/Berlin"

df = get_data(cfg)
```

---

## 📊 Data Quality & Validation

### Automatic Validation

The system automatically:
- ✅ Handles missing data (forward/backward fill)
- ✅ Clips negative values (PV can't be negative)
- ✅ Validates timezone consistency
- ✅ Handles DST transitions
- ✅ Fills gaps in API responses

### Data Quality Checks

```python
# Check for missing values
print(df.isnull().sum())

# Check data range
print(df.describe())

# Verify PV generation is realistic
assert df['pv_kW'].max() <= cfg['pv_system']['peak_power_kw']
assert df['pv_kW'].min() >= 0

# Verify prices are reasonable
assert df['price_eur_per_kwh'].between(0.05, 0.50).all()
```

---

## 🔧 Troubleshooting

### Issue: "No data from PVGIS"

**Solution**:
1. Check internet connection
2. Verify lat/lon are valid
3. System will automatically use Open-Meteo fallback

### Issue: "ENTSO-E API failed"

**Solution**:
1. Check API token is set
2. Verify bidding zone code (e.g., "DE_LU")
3. System will use synthetic TOU prices as fallback

### Issue: "Weather data missing"

**Solution**:
- Not critical - system continues without weather
- Weather is optional for basic functionality
- Only affects temperature-based features

---

## 📚 API Documentation Links

- **PVGIS**: https://joint-research-centre.ec.europa.eu/pvgis-online-tool_en
- **ENTSO-E**: https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html
- **Open-Meteo**: https://open-meteo.com/en/docs

---

## 💡 Pro Tips

1. **Cache Everything**: First run is slow, subsequent runs are fast
2. **Use Real Prices**: Get ENTSO-E token for accurate economics
3. **Adjust Load**: Modify `generate_synthetic_load_series()` for your pattern
4. **Multiple Locations**: Easy to compare different cities
5. **Historical Analysis**: Set `start_date` to past dates

---

## 🎯 Summary

**Dataset Type**: **Dynamically Generated** (not static)

**Data Sources**:
- ✅ PVGIS (PV generation)
- ✅ ENTSO-E (electricity prices)
- ✅ Open-Meteo (weather)
- ✅ Synthetic (household load)

**Output**: Hourly time series with 7+ features

**Storage**: 
- Cached in `cache/`
- Results in `outputs/`
- No permanent dataset file

**Flexibility**: Change location, dates, system size anytime!

---

**Last Updated**: 2025-11-09  
**Version**: 2.0