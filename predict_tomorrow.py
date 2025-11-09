"""Tomorrow's prediction script for prosumer energy advisor.

This script generates hourly action recommendations for the next day,
saves them to CSV, and provides a preview of the results.
"""

import os
from datetime import datetime, timedelta
from typing import Dict, Any

import yaml
import pytz
import pandas as pd

from fetch_data import get_data
from features import prepare_dataset
from predict import recommend_actions_for_df, load_model_and_config


def main() -> None:
    """Generate and save predictions for tomorrow.
    
    Workflow:
    1. Load configuration
    2. Determine tomorrow's date in local timezone
    3. Fetch data (PV, price, weather, load) for tomorrow
    4. Engineer features
    5. Load trained model
    6. Generate action recommendations
    7. Save results to CSV
    8. Display preview
    
    Raises:
        FileNotFoundError: If config.yaml or model files are missing
        ValueError: If configuration or model metadata is invalid
    """
    # Load configuration
    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        raise SystemExit("ERROR: config.yaml not found")
    
    # Determine tomorrow's date in local timezone
    tz_name = cfg.get("location", {}).get("tz", "Europe/Berlin")
    try:
        tzinfo = pytz.timezone(tz_name)
    except pytz.UnknownTimeZoneError:
        raise SystemExit(f"ERROR: Unknown timezone: {tz_name}")
    
    now_local = datetime.now(tzinfo)
    tomorrow_local = (now_local + timedelta(days=1)).date()
    
    print(f"[predict_tomorrow] Generating predictions for {tomorrow_local}")
    print(f"[predict_tomorrow] Timezone: {tz_name}")
    
    # Create prediction configuration for tomorrow
    cfg_pred = dict(cfg)  # Shallow copy
    cfg_pred.setdefault("location", {})
    cfg_pred["location"]["start_date"] = tomorrow_local.isoformat()
    cfg_pred["location"]["days"] = 1
    
    # Fetch raw data
    print("[predict_tomorrow] Fetching data...")
    try:
        df_raw = get_data(cfg_pred)
    except Exception as e:
        raise SystemExit(f"ERROR: Failed to fetch data: {e}")
    
    if df_raw.empty:
        raise SystemExit("ERROR: No data fetched for tomorrow")
    
    print(f"[predict_tomorrow] Fetched {len(df_raw)} hourly data points")
    
    # Build features
    print("[predict_tomorrow] Engineering features...")
    try:
        df_feat = prepare_dataset(df_raw, cfg_pred)
    except Exception as e:
        raise SystemExit(f"ERROR: Feature engineering failed: {e}")
    
    # Load trained model and metadata
    print("[predict_tomorrow] Loading model...")
    try:
        model, feature_cols, action_mapping = load_model_and_config()
    except FileNotFoundError as e:
        raise SystemExit(f"ERROR: {e}")
    except ValueError as e:
        raise SystemExit(f"ERROR: Invalid model metadata: {e}")
    
    # Generate predictions
    print("[predict_tomorrow] Generating recommendations...")
    try:
        res_df = recommend_actions_for_df(
            df_feat, model, cfg_pred, feature_cols, action_mapping
        )
    except Exception as e:
        raise SystemExit(f"ERROR: Prediction failed: {e}")
    
    # Merge predictions back to raw dataframe
    df_out = df_raw.copy()
    if "time" in res_df.columns:
        df_out["recommended_action"] = res_df.set_index("time")["recommended_action"]
    else:
        df_out["recommended_action"] = res_df["recommended_action"].values
    
    # Save to CSV
    out_dir = cfg.get("visualization", {}).get("plot_dir", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"predictions_{tomorrow_local.isoformat()}.csv")
    
    try:
        df_out.to_csv(out_csv)
        print(f"\n✓ Successfully saved predictions to: {out_csv}")
    except Exception as e:
        raise SystemExit(f"ERROR: Failed to save CSV: {e}")
    
    # Display preview
    print(f"\n{'='*80}")
    print(f"Preview of Tomorrow's Predictions ({tomorrow_local})")
    print(f"{'='*80}\n")
    
    # Select columns for preview
    preview_cols = []
    for col in ["pv_kW", "load_kW"]:
        if col in df_out.columns:
            preview_cols.append(col)
    
    price_cols = [c for c in df_out.columns if "price" in c.lower()]
    if price_cols:
        preview_cols.append(price_cols[0])
    
    if "recommended_action" in df_out.columns:
        preview_cols.append("recommended_action")
    
    if preview_cols:
        print("First 6 hours:")
        print(df_out[preview_cols].head(6).to_string())
        print(f"\n{'='*80}")
        
        # Action distribution
        if "recommended_action" in df_out.columns:
            action_counts = df_out["recommended_action"].value_counts()
            print("\nAction Distribution:")
            for action, count in action_counts.items():
                pct = 100 * count / len(df_out)
                print(f"  {action:20s}: {count:2d} hours ({pct:5.1f}%)")
            print(f"{'='*80}\n")
    else:
        print("No preview columns available")
    
    print(f"Total: {len(df_out)} hourly predictions saved")


if __name__ == "__main__":
    main()
