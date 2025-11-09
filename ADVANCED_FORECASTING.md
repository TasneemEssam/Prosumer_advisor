# Advanced Forecasting Integration Guide

This guide explains how to integrate state-of-the-art forecasting tools into the Prosumer Energy Advisor project.

## 🎯 Overview

The project can be enhanced with advanced time series forecasting libraries:

1. **Darts** - Professional time series forecasting library
2. **Chronos** - Amazon's foundation model for time series
3. **HTW Berlin Solar Calculator** - Accurate PV yield calculations
4. **Open-Meteo** - High-quality weather forecasting API

## 📚 Resources

- **Darts**: https://unit8co.github.io/darts/
- **Chronos**: https://pypi.org/project/chronos-forecasting/
- **HTW Solar**: https://solar.htw-berlin.de/rechner/
- **Open-Meteo**: https://github.com/open-meteo/open-meteo

---

## 1️⃣ Darts Integration

### Overview
Darts is a Python library for time series forecasting with 40+ models including:
- Statistical models (ARIMA, ETS, Prophet)
- Deep learning (LSTM, Transformer, N-BEATS, TFT)
- Ensemble methods
- Built-in backtesting and evaluation

### Installation

```bash
pip install darts
# For deep learning models:
pip install "darts[torch]"
# For all features:
pip install "darts[all]"
```

### Integration Example

Create `Prosumer_advisor/forecasting_darts.py`:

```python
"""Darts-based forecasting for load and PV prediction."""

from typing import Dict, Any, Optional
import pandas as pd
from darts import TimeSeries
from darts.models import (
    ExponentialSmoothing,
    ARIMA,
    Prophet,
    NBEATSModel,
    TFTModel,
    LightGBMModel
)
from darts.metrics import mape, rmse


class DartsForecaster:
    """Advanced forecasting using Darts library.
    
    Supports multiple forecasting models for load and PV prediction.
    """
    
    def __init__(self, model_type: str = "prophet"):
        """Initialize forecaster.
        
        Args:
            model_type: One of 'prophet', 'arima', 'nbeats', 'tft', 'lightgbm'
        """
        self.model_type = model_type
        self.model = None
        
    def create_model(self):
        """Create forecasting model based on type."""
        if self.model_type == "prophet":
            return Prophet()
        elif self.model_type == "arima":
            return ARIMA()
        elif self.model_type == "nbeats":
            return NBEATSModel(
                input_chunk_length=24,
                output_chunk_length=24,
                n_epochs=100
            )
        elif self.model_type == "tft":
            return TFTModel(
                input_chunk_length=24,
                output_chunk_length=24,
                n_epochs=100
            )
        elif self.model_type == "lightgbm":
            return LightGBMModel(lags=24)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, df: pd.DataFrame, target_col: str = "load_kW"):
        """Train forecasting model.
        
        Args:
            df: DataFrame with DatetimeIndex and target column
            target_col: Column to forecast
        """
        # Convert to Darts TimeSeries
        series = TimeSeries.from_dataframe(
            df,
            value_cols=target_col,
            fill_missing_dates=True
        )
        
        # Create and train model
        self.model = self.create_model()
        self.model.fit(series)
        
        return self
    
    def predict(self, n_hours: int = 24) -> pd.DataFrame:
        """Generate forecast.
        
        Args:
            n_hours: Number of hours to forecast
            
        Returns:
            DataFrame with forecasted values
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Generate forecast
        forecast = self.model.predict(n=n_hours)
        
        # Convert back to DataFrame
        return forecast.pd_dataframe()
    
    def evaluate(self, df_test: pd.DataFrame, target_col: str = "load_kW"):
        """Evaluate model on test data.
        
        Args:
            df_test: Test DataFrame
            target_col: Target column name
            
        Returns:
            Dict with evaluation metrics
        """
        test_series = TimeSeries.from_dataframe(df_test, value_cols=target_col)
        forecast = self.model.predict(n=len(df_test))
        
        return {
            "mape": mape(test_series, forecast),
            "rmse": rmse(test_series, forecast)
        }


# Usage example
def forecast_load_with_darts(df_train: pd.DataFrame, n_hours: int = 24):
    """Forecast load using Darts Prophet model."""
    forecaster = DartsForecaster(model_type="prophet")
    forecaster.train(df_train, target_col="load_kW")
    forecast_df = forecaster.predict(n_hours=n_hours)
    return forecast_df
```

### Configuration Update

Add to `config.yaml`:

```yaml
forecasting:
  enabled: true
  model: prophet  # prophet, arima, nbeats, tft, lightgbm
  horizon_hours: 24
  retrain_frequency: daily
```

---

## 2️⃣ Chronos Integration

### Overview
Chronos is Amazon's pretrained foundation model for zero-shot time series forecasting.
- No training required
- Works on any time series
- Multiple model sizes (tiny, mini, small, base, large)

### Installation

```bash
pip install chronos-forecasting
```

### Integration Example

Create `Prosumer_advisor/forecasting_chronos.py`:

```python
"""Chronos foundation model for zero-shot forecasting."""

from typing import Optional
import pandas as pd
import torch
from chronos import ChronosPipeline


class ChronosForecaster:
    """Zero-shot forecasting using Amazon Chronos."""
    
    def __init__(self, model_size: str = "small"):
        """Initialize Chronos model.
        
        Args:
            model_size: One of 'tiny', 'mini', 'small', 'base', 'large'
        """
        model_name = f"amazon/chronos-t5-{model_size}"
        self.pipeline = ChronosPipeline.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    
    def forecast(
        self,
        historical_data: pd.Series,
        prediction_length: int = 24,
        num_samples: int = 20
    ) -> pd.DataFrame:
        """Generate probabilistic forecast.
        
        Args:
            historical_data: Historical time series
            prediction_length: Number of steps to forecast
            num_samples: Number of sample paths
            
        Returns:
            DataFrame with mean, median, and quantiles
        """
        # Convert to tensor
        context = torch.tensor(historical_data.values)
        
        # Generate forecast
        forecast = self.pipeline.predict(
            context,
            prediction_length=prediction_length,
            num_samples=num_samples
        )
        
        # Calculate statistics
        forecast_index = pd.date_range(
            start=historical_data.index[-1],
            periods=prediction_length + 1,
            freq='H'
        )[1:]
        
        low, median, high = forecast.quantile(
            torch.tensor([0.1, 0.5, 0.9]),
            dim=0
        )
        
        return pd.DataFrame({
            'forecast_median': median.numpy(),
            'forecast_mean': forecast.mean(dim=0).numpy(),
            'forecast_low': low.numpy(),
            'forecast_high': high.numpy()
        }, index=forecast_index)


# Usage example
def forecast_with_chronos(df: pd.DataFrame, target_col: str = "load_kW"):
    """Forecast using Chronos foundation model."""
    forecaster = ChronosForecaster(model_size="small")
    forecast_df = forecaster.forecast(
        historical_data=df[target_col],
        prediction_length=24
    )
    return forecast_df
```

---

## 3️⃣ HTW Berlin Solar Calculator Integration

### Overview
HTW Berlin provides accurate PV yield calculations for Germany with:
- Detailed irradiation data
- System-specific calculations
- Historical and forecast data

### Integration Example

Create `Prosumer_advisor/htw_solar.py`:

```python
"""HTW Berlin Solar Calculator integration."""

from typing import Dict, Any, Optional
import requests
import pandas as pd


class HTWSolarCalculator:
    """Interface to HTW Berlin Solar Calculator.
    
    Provides accurate PV yield calculations for German locations.
    """
    
    BASE_URL = "https://solar.htw-berlin.de/api"
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize calculator.
        
        Args:
            api_key: Optional API key for enhanced access
        """
        self.api_key = api_key
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})
    
    def calculate_pv_yield(
        self,
        lat: float,
        lon: float,
        peak_power_kw: float,
        tilt_deg: float,
        azimuth_deg: float,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Calculate PV yield for location and system.
        
        Args:
            lat: Latitude
            lon: Longitude
            peak_power_kw: System peak power in kW
            tilt_deg: Panel tilt angle
            azimuth_deg: Panel azimuth (0=South)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with hourly PV yield predictions
        """
        params = {
            "lat": lat,
            "lon": lon,
            "peak_power": peak_power_kw,
            "tilt": tilt_deg,
            "azimuth": azimuth_deg,
            "start": start_date,
            "end": end_date
        }
        
        # Note: This is a placeholder - actual API endpoint may differ
        # Check HTW Berlin documentation for exact API specification
        response = self.session.get(
            f"{self.BASE_URL}/pv_yield",
            params=params,
            timeout=30
        )
        response.raise_for_status()
        
        data = response.json()
        
        # Convert to DataFrame
        df = pd.DataFrame(data["hourly_data"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
        
        return df


# Integration with existing fetch_data.py
def fetch_pv_from_htw(cfg: Dict[str, Any], start_dt, end_dt) -> pd.Series:
    """Fetch PV data from HTW Berlin Solar Calculator."""
    calculator = HTWSolarCalculator()
    
    df = calculator.calculate_pv_yield(
        lat=cfg["location"]["lat"],
        lon=cfg["location"]["lon"],
        peak_power_kw=cfg["pv_system"]["peak_power_kw"],
        tilt_deg=cfg["pv_system"]["tilt_deg"],
        azimuth_deg=cfg["pv_system"]["azimuth_deg"],
        start_date=start_dt.strftime("%Y-%m-%d"),
        end_date=end_dt.strftime("%Y-%m-%d")
    )
    
    return df["pv_yield_kw"]
```

---

## 4️⃣ Enhanced Open-Meteo Integration

### Overview
Open-Meteo provides high-quality weather forecasts with:
- 16-day forecasts
- Historical data back to 1940
- Solar radiation forecasts
- Free API access

### Enhanced Integration

Update `fetch_data.py` with advanced Open-Meteo features:

```python
def fetch_weather_forecast_advanced(cfg: Dict[str, Any], start_dt, end_dt):
    """Enhanced Open-Meteo integration with solar forecasts."""
    lat = cfg["location"]["lat"]
    lon = cfg["location"]["lon"]
    tz_name = cfg["location"]["tz"]
    
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "temperature_2m",
            "cloudcover",
            "shortwave_radiation",
            "direct_radiation",
            "diffuse_radiation",
            "direct_normal_irradiance",
            "global_tilted_irradiance",  # For tilted panels
            "windspeed_10m",
            "precipitation"
        ],
        "timezone": tz_name,
        "start_date": start_dt.strftime("%Y-%m-%d"),
        "end_date": end_dt.strftime("%Y-%m-%d"),
        "tilt": cfg["pv_system"]["tilt_deg"],
        "azimuth": cfg["pv_system"]["azimuth_deg"]
    }
    
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    
    # Process enhanced weather data
    hourly = data["hourly"]
    df = pd.DataFrame({
        "temp_C": hourly["temperature_2m"],
        "cloudcover_pct": hourly["cloudcover"],
        "ghi_wm2": hourly["shortwave_radiation"],
        "dni_wm2": hourly["direct_normal_irradiance"],
        "dhi_wm2": hourly["diffuse_radiation"],
        "gti_wm2": hourly["global_tilted_irradiance"],  # Best for PV
        "windspeed_ms": hourly["windspeed_10m"],
        "precipitation_mm": hourly["precipitation"]
    })
    
    df.index = pd.to_datetime(hourly["time"])
    return df
```

---

## 🔧 Complete Integration Example

Create `Prosumer_advisor/advanced_pipeline.py`:

```python
"""Advanced forecasting pipeline with Darts and Chronos."""

from typing import Dict, Any
import pandas as pd
from forecasting_darts import DartsForecaster
from forecasting_chronos import ChronosForecaster


class AdvancedForecastingPipeline:
    """Complete forecasting pipeline with multiple models."""
    
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.load_forecaster = None
        self.pv_forecaster = None
        
    def train(self, df_train: pd.DataFrame):
        """Train all forecasting models."""
        # Train load forecaster with Darts
        self.load_forecaster = DartsForecaster(model_type="prophet")
        self.load_forecaster.train(df_train, target_col="load_kW")
        
        # Train PV forecaster with Darts
        self.pv_forecaster = DartsForecaster(model_type="prophet")
        self.pv_forecaster.train(df_train, target_col="pv_kW")
        
        print("✓ Advanced forecasting models trained")
        
    def forecast_tomorrow(self) -> pd.DataFrame:
        """Generate 24-hour forecast for tomorrow."""
        # Forecast load
        load_forecast = self.load_forecaster.predict(n_hours=24)
        
        # Forecast PV
        pv_forecast = self.pv_forecaster.predict(n_hours=24)
        
        # Combine forecasts
        df_forecast = pd.DataFrame({
            "load_kW": load_forecast.values.flatten(),
            "pv_kW": pv_forecast.values.flatten()
        }, index=load_forecast.index)
        
        return df_forecast
    
    def forecast_with_chronos(self, df_history: pd.DataFrame) -> pd.DataFrame:
        """Zero-shot forecast using Chronos."""
        chronos = ChronosForecaster(model_size="small")
        
        # Forecast load
        load_forecast = chronos.forecast(
            df_history["load_kW"],
            prediction_length=24
        )
        
        # Forecast PV
        pv_forecast = chronos.forecast(
            df_history["pv_kW"],
            prediction_length=24
        )
        
        return pd.DataFrame({
            "load_kW_forecast": load_forecast["forecast_median"],
            "pv_kW_forecast": pv_forecast["forecast_median"]
        })
```

---

## 📦 Updated Requirements

Add to `requirements.txt`:

```txt
# Advanced Forecasting
darts>=0.27.0
chronos-forecasting>=1.0.0
prophet>=1.1.5
lightgbm>=4.0.0

# Deep Learning (optional)
torch>=2.0.0
pytorch-lightning>=2.0.0

# Additional utilities
statsmodels>=0.14.0
pmdarima>=2.0.0
```

---

## 🚀 Usage Examples

### Example 1: Darts Prophet Forecasting

```python
from forecasting_darts import DartsForecaster

# Train model
forecaster = DartsForecaster(model_type="prophet")
forecaster.train(df_train, target_col="load_kW")

# Generate 24-hour forecast
forecast = forecaster.predict(n_hours=24)
print(forecast)
```

### Example 2: Chronos Zero-Shot Forecasting

```python
from forecasting_chronos import ChronosForecaster

# No training needed!
forecaster = ChronosForecaster(model_size="small")
forecast = forecaster.forecast(
    historical_data=df["load_kW"],
    prediction_length=24
)
print(forecast)
```

### Example 3: HTW Solar Calculator

```python
from htw_solar import HTWSolarCalculator

calculator = HTWSolarCalculator()
pv_yield = calculator.calculate_pv_yield(
    lat=52.52,
    lon=13.405,
    peak_power_kw=5.0,
    tilt_deg=30,
    azimuth_deg=0,
    start_date="2024-06-01",
    end_date="2024-06-02"
)
print(pv_yield)
```

---

## 📊 Model Comparison

| Model | Training Time | Accuracy | Use Case |
|-------|--------------|----------|----------|
| Prophet (Darts) | Fast | Good | Seasonal patterns |
| ARIMA (Darts) | Fast | Good | Short-term |
| N-BEATS (Darts) | Slow | Excellent | Complex patterns |
| TFT (Darts) | Slow | Excellent | Multi-variate |
| Chronos | None | Very Good | Zero-shot |
| LightGBM (Darts) | Fast | Very Good | Feature-rich |

---

## 🎯 Recommendations

1. **Start Simple**: Begin with Prophet or Chronos
2. **Evaluate**: Compare multiple models on your data
3. **Ensemble**: Combine predictions from multiple models
4. **Monitor**: Track forecast accuracy over time
5. **Retrain**: Update models regularly with new data

---

## 📚 Additional Resources

- **Darts Documentation**: https://unit8co.github.io/darts/
- **Darts Examples**: https://github.com/unit8co/darts/tree/master/examples
- **Chronos Paper**: https://arxiv.org/abs/2403.07815
- **Open-Meteo API**: https://open-meteo.com/en/docs
- **HTW Solar**: https://solar.htw-berlin.de/

---

**Next Steps**: Choose a forecasting approach and integrate it into your pipeline!