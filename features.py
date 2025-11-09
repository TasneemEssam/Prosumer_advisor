"""Feature engineering module for prosumer energy advisor.

This module provides functions to create derived features from raw energy data,
including time-based features, lag features, rolling averages, and forecast flags.
"""

from typing import Dict, Any
import pandas as pd
import numpy as np


def engineer_features(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Add derived feature columns to the DataFrame.
    
    Creates features including:
    - PV surplus (PV generation - load)
    - Time-based features (hour, day of week)
    - Lag features (previous hours' values)
    - Rolling averages
    - Next-day forecast flags
    
    Args:
        df: Input DataFrame with index as DatetimeIndex and columns 'pv_kW', 'load_kW'
        cfg: Configuration dictionary containing oracle and feature settings
        
    Returns:
        DataFrame with additional feature columns
        
    Raises:
        KeyError: If required columns are missing from input DataFrame
    """
    df = df.copy()
    
    # Validate required columns
    required_cols = ['pv_kW', 'load_kW']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}")
    
    # Basic features - PV surplus
    df['pv_surplus'] = df['pv_kW'] - df['load_kW']
    
    # Time-based features
    df['hour'] = df.index.hour
    df['dow'] = df.index.dayofweek  # Monday=0, Sunday=6
    
    # Lag features (previous hours' values)
    df['load_lag1'] = df['load_kW'].shift(1)
    df['load_lag2'] = df['load_kW'].shift(2)
    df['pv_lag1'] = df['pv_kW'].shift(1)
    
    # Rolling averages (using shift to avoid data leakage)
    df['roll_load_3h'] = df['load_kW'].shift(1).rolling(window=3, min_periods=1).mean()
    df['roll_pv_6h'] = df['pv_kW'].shift(1).rolling(window=6, min_periods=1).mean()
    
    # Weather-based features (if available)
    if 'cloudcover_pct' in df.columns:
        df['cloudcover_next_6h'] = (
            df['cloudcover_pct']
            .shift(-1)
            .rolling(window=6, min_periods=1)
            .mean()
        )
    
    # Next-day forecast flag (for pre-charging logic)
    if cfg.get('oracle', {}).get('weather_precharge_enabled', False):
        df = add_next_day_flag(df, cfg)
    
    return df

def add_next_day_flag(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Add binary flag for next-day low PV or high price conditions.
    
    Marks hours when the next day is forecast to have low PV generation or high prices,
    which can trigger pre-charging behavior in the evening hours.
    
    Args:
        df: DataFrame with DatetimeIndex and columns for irradiance/PV and price
        cfg: Configuration dictionary
        
    Returns:
        DataFrame with added 'next_day_pv_flag' column (0 or 1)
    """
    # Use irradiance if available, otherwise use PV generation
    if 'irradiance_wm2' in df.columns:
        daily_irradiance = df['irradiance_wm2'].resample('D').sum()
    else:
        daily_irradiance = df['pv_kW'].resample('D').sum()
    
    # Calculate daily average price
    if 'price_eur_per_kwh' in df.columns:
        daily_price_avg = df['price_eur_per_kwh'].resample('D').mean()
    else:
        daily_price_avg = pd.Series(dtype=float)
    
    # Vectorized approach for better performance
    day_floor = df.index.floor('D')
    next_day = day_floor + pd.Timedelta(days=1)
    
    # Map current and next day values
    irr_today = day_floor.map(daily_irradiance)
    irr_tomorrow = next_day.map(daily_irradiance)
    price_today = day_floor.map(daily_price_avg)
    price_tomorrow = next_day.map(daily_price_avg)
    
    # Thresholds for triggering pre-charge
    LOW_PV_THRESHOLD = 0.5  # Tomorrow's PV < 50% of today
    HIGH_PRICE_THRESHOLD = 1.2  # Tomorrow's price > 120% of today
    
    # Calculate flag: 1 if next day has low PV OR high price
    df['next_day_pv_flag'] = (
        ((irr_tomorrow < LOW_PV_THRESHOLD * irr_today) |
         (price_tomorrow > HIGH_PRICE_THRESHOLD * price_today))
        .fillna(False)
        .astype(int)
    )
    
    return df

def label_oracle_actions(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Apply rule-based oracle to label recommended actions.
    
    Uses a heuristic decision tree based on:
    - PV surplus availability
    - Electricity price levels (high/low quantiles)
    - Time of day (for EV charging windows)
    - Weather forecasts (if enabled)
    
    Args:
        df: DataFrame with features including pv_surplus, price_eur_per_kwh
        cfg: Configuration dictionary with oracle settings
        
    Returns:
        DataFrame with added 'recommended_action' column
    """
    df = df.copy()
    
    # Extract oracle configuration
    oracle_cfg = cfg.get('oracle', {})
    price_high_q = oracle_cfg.get('price_high_quantile', 0.6)
    price_low_q = oracle_cfg.get('price_low_quantile', 0.3)
    ev_start = oracle_cfg.get('ev_night_start', 0)
    ev_end = oracle_cfg.get('ev_night_end', 6)
    ev_enabled = oracle_cfg.get('ev_enabled', True)
    weather_precharge = oracle_cfg.get('weather_precharge_enabled', False)
    
    # Compute daily price thresholds (high and low quantiles)
    df['price_high_thresh'] = (
        df['price_eur_per_kwh']
        .groupby(df.index.date)
        .transform(lambda x: x.quantile(price_high_q) if len(x) > 0 else np.nan)
    )
    df['price_low_thresh'] = (
        df['price_eur_per_kwh']
        .groupby(df.index.date)
        .transform(lambda x: x.quantile(price_low_q) if len(x) > 0 else np.nan)
    )
    # Vectorized action determination for better performance
    actions = pd.Series(index=df.index, dtype=str)
    
    # Extract relevant columns
    pv_surplus = df['pv_surplus'].fillna(0)
    price = df['price_eur_per_kwh'].fillna(0)
    high_thr = df['price_high_thresh'].fillna(price)
    low_thr = df['price_low_thresh'].fillna(price)
    hour = df.index.hour
    
    # Detect flat price days (no variation)
    PRICE_EPSILON = 1e-6
    is_flat_day = (high_thr - low_thr).abs() < PRICE_EPSILON
    
    # Rule 1: PV surplus available
    has_surplus = pv_surplus > 0
    high_price = price > high_thr
    
    # When surplus + high price -> sell to grid
    actions[has_surplus & high_price & ~is_flat_day] = "sell_to_grid"
    # When surplus + not high price -> use PV directly
    actions[has_surplus & ~high_price] = "use_pv_direct"
    
    # Rule 2: No surplus (deficit or zero)
    no_surplus = ~has_surplus
    low_price = price < low_thr
    in_ev_window = (hour >= ev_start) & (hour < ev_end)
    
    # Flat day logic
    actions[no_surplus & is_flat_day & ev_enabled & in_ev_window] = "charge_ev"
    actions[no_surplus & is_flat_day & ~(ev_enabled & in_ev_window)] = "idle"
    
    # Low price logic (not flat day)
    actions[no_surplus & ~is_flat_day & low_price & ev_enabled & in_ev_window] = "charge_ev"
    actions[no_surplus & ~is_flat_day & low_price & ~(ev_enabled & in_ev_window)] = "charge_battery"
    
    # Default to idle for remaining cases
    actions[no_surplus & ~is_flat_day & ~low_price] = "idle"
    
    # Weather-aware pre-charging override
    if weather_precharge and 'next_day_pv_flag' in df.columns:
        EVENING_HOUR = 18
        evening_precharge = (
            (actions == "idle") &
            (df['next_day_pv_flag'] == 1) &
            (hour >= EVENING_HOUR)
        )
        actions[evening_precharge] = "charge_battery"
    
    df['recommended_action'] = actions
    return df

def prepare_dataset(df_raw: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Prepare complete dataset with features and labels.
    
    Performs full data preparation pipeline:
    1. Feature engineering (time features, lags, rolling averages)
    2. Oracle action labeling
    3. Data cleaning (remove rows with missing essential values)
    
    Args:
        df_raw: Raw DataFrame with basic columns (pv_kW, load_kW, price_eur_per_kwh)
        cfg: Configuration dictionary
        
    Returns:
        Clean DataFrame ready for model training/prediction
        
    Raises:
        KeyError: If required columns are missing
    """
    # Engineer features
    df = engineer_features(df_raw, cfg)
    
    # Label oracle actions
    df = label_oracle_actions(df, cfg)
    
    # Drop rows with NaN in essential columns
    essential_cols = ['pv_kW', 'load_kW', 'price_eur_per_kwh', 'pv_surplus', 'recommended_action']
    existing_cols = [col for col in essential_cols if col in df.columns]
    df = df.dropna(subset=existing_cols)
    
    return df
