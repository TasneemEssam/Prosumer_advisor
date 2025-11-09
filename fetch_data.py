"""Data fetching module for prosumer energy advisor.

This module handles fetching of PV generation, electricity prices, weather data,
and synthetic load generation from various APIs and sources.
"""

import os
import datetime
from typing import Dict, Any, Optional
from dateutil import tz

import requests
import pandas as pd
import numpy as np

# Configure requests session with timeout and user agent
requests_session = requests.Session()
requests_session.headers.update({"User-Agent": "ProsumerAdvisor/1.0"})

# Constants
DEFAULT_REQUEST_TIMEOUT = 30  # seconds
PVGIS_BASE_URL = "https://re.jrc.ec.europa.eu/api/v5_3/seriescalc"
ENTSOE_BASE_URL = "https://web-api.tp.entsoe.eu/api"


def get_cfg(cfg: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Safely extract nested configuration values.
    
    Args:
        cfg: Configuration dictionary
        *keys: Sequence of nested keys to traverse
        default: Default value if key path doesn't exist
        
    Returns:
        Value at the key path, or default if not found
        
    Example:
        >>> get_cfg(cfg, "location", "lat", default=52.52)
    """
    cur = cfg
    for k in keys:
        if cur is None or not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return cur if cur is not None else default

def fetch_pv_series(
    cfg: Dict[str, Any],
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp
) -> pd.Series:
    """Fetch PV power generation time series.
    
    Attempts multiple data sources in order:
    1. PVGIS API (preferred, uses historical climate data)
    2. Open-Meteo irradiance (fallback)
    3. Synthetic sine wave (last resort)
    
    Args:
        cfg: Configuration dictionary with PV system parameters
        start_dt: Start datetime (timezone-aware)
        end_dt: End datetime (timezone-aware)
        
    Returns:
        Series of PV power in kW, indexed by datetime
        
    Raises:
        ValueError: If configuration parameters are invalid
    """
    import pandas as pd
    import numpy as np
    from dateutil import tz

    lat = get_cfg(cfg, "location", "lat")
    lon = get_cfg(cfg, "location", "lon")
    peak_kw = float(get_cfg(cfg, "pv_system", "peak_power_kw", default=5.0))
    tilt = float(get_cfg(cfg, "pv_system", "tilt_deg", default=30))
    azimuth = float(get_cfg(cfg, "pv_system", "azimuth_deg", default=0))
    loss = float(get_cfg(cfg, "pv_system", "loss_percent", default=14))
    tz_name = get_cfg(cfg, "location", "tz", default="Europe/Berlin")

    # Use a stable climate year only for SHAPE; we map month/day/hour onto target window.
    pvgis_year = int(get_cfg(cfg, "pv_system", "pvgis_year", default=2019))
    url_base = "https://re.jrc.ec.europa.eu/api/v5_3/seriescalc"
    params = {
        "lat": lat,
        "lon": lon,
        "startyear": pvgis_year,
        "endyear": pvgis_year,
        "peakpower": peak_kw,
        "loss": loss,
        "angle": tilt,
        "aspect": azimuth,
        "outputformat": "json",
        "browser": 0
    }

    try:
        resp = requests_session.get(url_base, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        hourly = data["outputs"]["hourly"]
        df = pd.DataFrame(hourly)

        # Parse PVGIS time robustly
        try:
            times = pd.to_datetime(df["time"], format="%Y%m%d:%H%M", errors="raise")
        except Exception:
            times = pd.to_datetime(df["time"], errors="coerce")
        if times.isna().all():
            raise ValueError("PVGIS 'time' field could not be parsed.")

        # PVGIS timestamps are UTC; convert to site tz
        times = times.dt.tz_localize("UTC").dt.tz_convert(tz_name)
        df = df.set_index(times)

        # Prefer PVGIS-provided power if available
        power_candidates = ["P", "P_PV", "Pmax", "Pdc", "P_ac", "P_AC"]
        p_col = next((c for c in power_candidates if c in df.columns), None)

        if p_col is not None:
            df["pv_kW"] = pd.to_numeric(df[p_col], errors="coerce") / 1000.0
            if df["pv_kW"].notna().sum() == 0:
                p_col = None  # treat as missing

        if p_col is None:
            # Derive PV power from irradiance if 'P' is missing
            if "G(i)" not in df.columns:
                raise KeyError("PVGIS response lacks both 'P' and 'G(i)'.")
            Gi = pd.to_numeric(df["G(i)"], errors="coerce").clip(lower=0.0) / 1000.0  # kW/m²
            pr = (1.0 - loss / 100.0) * 0.90
            if "T2m" in df.columns:
                T = pd.to_numeric(df["T2m"], errors="coerce").fillna(25.0)
                temp_derate = 1.0 - 0.004 * (T - 25.0).clip(lower=0.0)
            else:
                temp_derate = 1.0
            df["pv_kW"] = (Gi * peak_kw * pr * temp_derate).clip(lower=0.0)

        # Build the requested index and map by month-day-hour (ignore year), DST-safe
        target_idx = pd.date_range(start_dt, end_dt, freq="H", tz=tz_name, inclusive="left")

        # Collapse duplicates caused by DST by averaging within (month, day, hour)
        src_groups = (
            df.groupby([df.index.month.rename("m"),
                        df.index.day.rename("d"),
                        df.index.hour.rename("h")])["pv_kW"]
            .mean()
        )

        # Reindex to the target’s (month, day, hour) keys
        keys_tgt = pd.MultiIndex.from_arrays(
            [target_idx.month, target_idx.day, target_idx.hour],
            names=["m", "d", "h"]
        )
        pv_vals = src_groups.reindex(keys_tgt).values

        # Fill gaps (e.g., Feb 29 missing in non-leap PVGIS year), clip negatives
        pv_series = (
            pd.Series(pv_vals, index=target_idx, name="pv_kW")
            .ffill().bfill()
            .fillna(0.0)
            .clip(lower=0.0)
        )
        mode = f"power {p_col}" if p_col else "G(i)->power"
        print(f"[fetch_pv_series] PVGIS {pvgis_year}: using {mode} and mapped to target period.")
        return pv_series


    except Exception as e:
        print(f"[fetch_pv_series] PVGIS fetch/mapping failed, will use Open-Meteo irradiance. Error: {e}")

    # Fallback #1: Open-Meteo irradiance
    try:
        weather = fetch_weather_series(cfg, start_dt, end_dt)
        if weather.empty or "irradiance_wm2" not in weather:
            raise RuntimeError("Open-Meteo weather missing or no irradiance.")
        pr = (1.0 - loss / 100.0) * 0.90
        irr = weather["irradiance_wm2"].clip(lower=0.0) / 1000.0  # kW/m²
        temp = weather.get("temp_C", pd.Series(25.0, index=irr.index))
        derate = 1.0 - 0.004 * (temp.fillna(25.0) - 25.0).clip(lower=0.0)
        pv_kW = (irr * peak_kw * pr * derate).clip(lower=0.0)
        pv_kW.name = "pv_kW"
        print("[fetch_pv_series] Using Open-Meteo irradiance-based PV fallback.")
        return pv_kW
    except Exception as e:
        print(f"[fetch_pv_series] Open-Meteo fallback failed, using simple sine fallback. Error: {e}")

    # Fallback #2: simple diurnal sine
    idx = pd.date_range(start_dt, end_dt, freq="H", tz=tz_name, inclusive="left")
    pv = []
    for ts in idx:
        h = ts.hour + ts.minute / 60.0
        if 6 <= h < 18:
            angle = (h - 6) / 12 * np.pi
            pv_val = max(0, peak_kw * abs(np.sin(angle)))
        else:
            pv_val = 0.0
        pv.append(pv_val)
    return pd.Series(pv, index=idx, name="pv_kW")



def fetch_price_series(
    cfg: Dict[str, Any],
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp
) -> pd.Series:
    """Fetch day-ahead electricity price time series.
    
    Attempts to fetch real prices from ENTSO-E API. Falls back to synthetic
    time-of-use pricing if API is unavailable or token is missing.
    
    Args:
        cfg: Configuration dictionary with grid_price settings
        start_dt: Start datetime (timezone-aware)
        end_dt: End datetime (timezone-aware)
        
    Returns:
        Series of electricity prices in EUR/kWh, indexed by datetime
    """
    zone_label = get_cfg(cfg, "grid_price", "zone", default="DE_LU")
    token = get_cfg(cfg, "grid_price", "token") or os.environ.get("ENTSOE_API_TOKEN")
    tz_name = get_cfg(cfg, "location", "tz", default="Europe/Berlin")

    # Minimal mapping: if the label is DE_LU, use the correct BZN EIC; otherwise assume it's already an EIC
    eic = "10Y1001A1001A82H" if str(zone_label).upper() == "DE_LU" else zone_label  # CHANGED

    # Build time interval in UTC for ENTSO-E query. PeriodEnd is exclusive.
    def fetch_interval(s_dt, e_dt):
        url = "https://web-api.tp.entsoe.eu/api"  # CHANGED (matches your working PS test)
        s_utc = s_dt.astimezone(datetime.timezone.utc)
        e_utc = e_dt.astimezone(datetime.timezone.utc)
        period_start = s_utc.strftime("%Y%m%d%H00")
        period_end = e_utc.strftime("%Y%m%d%H00")
        params = {
            "documentType": "A44",
            "processType": "A01",     # CHANGED
            "in_Domain": eic,         # CHANGED
            "out_Domain": eic,        # CHANGED
            "periodStart": period_start,
            "periodEnd": period_end,
            "securityToken": token
        }
        try:
            resp = requests_session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            xml_text = resp.text
            # Parse XML minimally: take first <timeInterval><start> and all <price.amount>
            prices = []
            times = []
            start_idx = xml_text.find("<timeInterval>")
            if start_idx != -1:
                si = xml_text.find("<start>", start_idx) + len("<start>")
                se = xml_text.find("</start>", si)
                interval_start_str = xml_text[si:se]  # e.g. "2024-06-01T00:00:00Z"
                interval_start = datetime.datetime.fromisoformat(interval_start_str.replace("Z", "+00:00"))
                # move to local tz for the index we return
                interval_start = interval_start.astimezone(tz.gettz(tz_name))

                pos = 0
                ptag = "<price.amount>"
                etag = "</price.amount>"
                while True:
                    pi = xml_text.find(ptag, pos)
                    if pi == -1:
                        break
                    pj = xml_text.find(etag, pi)
                    price_val = float(xml_text[pi+len(ptag):pj])  # EUR/MWh
                    prices.append(price_val / 1000.0)             # → EUR/kWh
                    pos = pj

                for i in range(len(prices)):
                    times.append(interval_start + datetime.timedelta(hours=i))
            else:
                raise ValueError("No timeInterval found in XML response.")
            idx = pd.DatetimeIndex(times)
            # Fix tz: times are already tz-aware after astimezone; just ensure conversion, no tz_localize on aware
            if idx.tz is None:
                idx = idx.tz_localize(tz_name)
            else:
                idx = idx.tz_convert(tz_name)
            return pd.Series(prices, index=idx, name="price_eur_per_kwh")
        except Exception as e:
            print(f"[fetch_price_series] ENTSO-E API failed for {s_dt.date()}: {e}")
            return None

    # Fetch in chunks (month by month if span is large)
    price_series_list = []
    current = start_dt
    while current < end_dt:
        if (end_dt - current).days > 31:
            chunk_end = current + datetime.timedelta(days=31)
        else:
            chunk_end = end_dt
        series = None
        if token:
            series = fetch_interval(current, chunk_end)
        if series is None:
            # Fallback flat / simple TOU
            rng = np.random.default_rng(0)
            idx = pd.date_range(current, chunk_end, freq='H', tz=tz_name, inclusive='left')
            hr = idx.hour
            dow = idx.dayofweek
            base = 0.20
            tou = np.where((hr>=0)&(hr<6), 0.14,
                  np.where((hr>=6)&(hr<16), 0.18,
                  np.where((hr>=16)&(hr<21), 0.30, 0.22)))
            tou = np.where(dow>=5, tou*0.9, tou)
            noise = rng.normal(0, 0.005, size=len(idx))
            series = pd.Series((tou+noise).clip(0.05), index=idx, name="price_eur_per_kwh")
        price_series_list.append(series)
        current = chunk_end

    price_series = pd.concat(price_series_list)
    price_series = price_series[~price_series.index.duplicated(keep='first')]
    return price_series[(price_series.index >= start_dt) & (price_series.index < end_dt)]

def fetch_weather_series(
    cfg: Dict[str, Any],
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp
) -> pd.DataFrame:
    """Fetch weather data from Open-Meteo API.
    
    Retrieves temperature, cloud cover, and solar irradiance data.
    Automatically selects forecast or historical API based on date range.
    
    Args:
        cfg: Configuration dictionary with location settings
        start_dt: Start datetime (timezone-aware)
        end_dt: End datetime (timezone-aware)
        
    Returns:
        DataFrame with columns: temp_C, cloudcover_pct, irradiance_wm2
    """
    lat = get_cfg(cfg, "location", "lat")
    lon = get_cfg(cfg, "location", "lon")
    tz_name = get_cfg(cfg, "location", "tz", default="UTC")
    # Determine if we need forecast or historical API:
    today = datetime.datetime.now(tz.gettz(tz_name)).date()
    end_date = end_dt.date()
    start_date = start_dt.date()
    use_forecast = start_date >= today
    # Build base URL accordingly
    if use_forecast:
        url = "https://api.open-meteo.com/v1/forecast"
    else:
        url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,cloudcover,shortwave_radiation",
        "timezone": tz_name
    }
    params["start_date"] = str(start_date)
    params["end_date"] = str(end_date)
    try:
        resp = requests_session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        hourly = data.get('hourly')
        if not hourly:
            raise ValueError("No 'hourly' in Open-Meteo response")
        times = [pd.to_datetime(t) for t in hourly['time']]
        times = pd.DatetimeIndex(times)
        if times.tz is None:
            times = times.tz_localize(tz_name)
        df = pd.DataFrame({
            'temp_C': hourly['temperature_2m'],
            'cloudcover_pct': hourly['cloudcover'],
            'irradiance_wm2': hourly.get('shortwave_radiation', hourly.get('shortwave_radiation_sum'))
        }, index=times)
        df = df[(df.index >= start_dt) & (df.index < end_dt)]
        print(f"[fetch_weather_series] Retrieved weather data from Open-Meteo, entries: {len(df)}")
        return df
    except Exception as e:
        print(f"[fetch_weather_series] Open-Meteo fetch failed, proceeding without weather. Error: {e}")
        idx = pd.date_range(start_dt, end_dt, freq='H', tz=tz_name, inclusive='left')
        return pd.DataFrame(index=idx, columns=['temp_C','cloudcover_pct','irradiance_wm2'])

def get_data(cfg: Dict[str, Any]) -> pd.DataFrame:
    """Fetch and merge all required data sources.
    
    Orchestrates fetching of:
    - PV generation (from PVGIS or Open-Meteo)
    - Electricity prices (from ENTSO-E or synthetic)
    - Weather data (from Open-Meteo)
    - Synthetic load profile
    
    Args:
        cfg: Configuration dictionary with all settings
        
    Returns:
        DataFrame with hourly data including:
            - pv_kW: PV generation
            - load_kW: Household load
            - price_eur_per_kwh: Electricity price
            - temp_C, cloudcover_pct, irradiance_wm2: Weather data (if available)
            
    Raises:
        ValueError: If configuration is invalid
    """
    tz_name = get_cfg(cfg, "location", "tz", default="Europe/Berlin")
    start_date = get_cfg(cfg, "location", "start_date")
    days = get_cfg(cfg, "location", "days", default=1)
    if start_date:
        start_dt = pd.to_datetime(start_date).tz_localize(tz_name)
    else:
        now = pd.Timestamp.now(tz=tz_name)
        start_dt = (now - pd.Timedelta(days=days)).replace(minute=0, second=0, microsecond=0)
    end_dt = start_dt + datetime.timedelta(days=days)
    print(f"[get_data] Fetching data from {start_dt} to {end_dt} ...")

    pv = fetch_pv_series(cfg, start_dt, end_dt)
    price = fetch_price_series(cfg, start_dt, end_dt)

    idx = price.index if price is not None else pv.index
    if idx is None or len(idx) == 0:
        idx = pd.date_range(start_dt, end_dt, freq='H', tz=tz_name, inclusive='left')

    load = generate_synthetic_load_series(cfg, idx)
    weather = fetch_weather_series(cfg, start_dt, end_dt)

    df = pd.DataFrame(index=idx)
    if pv is not None:
        df = df.join(pv, how='left')
    if load is not None:
        df = df.join(load, how='left')
    if price is not None:
        df = df.join(price, how='left')
    if not weather.empty:
        df = df.join(weather, how='left')

    if 'pv_kW' in df:
        df['pv_kW'] = df['pv_kW'].fillna(0.0)
    if 'load_kW' in df:
        df['load_kW'] = df['load_kW'].ffill().bfill().fillna(0.0)
    if 'price_eur_per_kwh' in df:
        df['price_eur_per_kwh'] = df['price_eur_per_kwh'].ffill().bfill()
    
    return df

def generate_synthetic_load_series(
    cfg: Dict[str, Any],
    index: pd.DatetimeIndex
) -> pd.Series:
    """Generate synthetic household load profile.
    
    Creates a realistic load pattern with:
    - Morning peak (6-9 AM)
    - Daytime baseline (9 AM-5 PM)
    - Evening peak (5-9 PM)
    - Night baseline (9 PM-6 AM)
    - Weekend adjustments
    - Random variation
    
    Args:
        cfg: Configuration dictionary (uses random_seed if present)
        index: DatetimeIndex for the output series
        
    Returns:
        Series of load values in kW
    """
    import math
    rng = np.random.default_rng(get_cfg(cfg, "model", "random_seed", default=0))
    values = []
    for ts in index:
        hour = ts.hour
        dow = ts.weekday()  # 0 = Monday
        base = 0.4
        if hour < 6:
            usage = base
        elif hour < 9:
            usage = base + 1.0
        elif hour < 17:
            usage = base + 0.2
        elif hour < 21:
            usage = base + 1.5
        else:
            usage = base + 0.5
        if dow >= 5 and 10 <= hour < 17:
            usage += 0.3
        usage *= rng.normal(1.0, 0.1)
        usage = max(0.1, usage)
        values.append(usage)
    return pd.Series(values, index=index, name='load_KW').rename('load_kW')
