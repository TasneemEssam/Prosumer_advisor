"""Prediction module for prosumer energy advisor.

This module handles loading trained models and generating action recommendations
for new data, including human-readable explanations.
"""

from typing import Dict, Any, Tuple, List, Optional
import os
import pandas as pd
import numpy as np
import joblib
import yaml

from features import engineer_features, label_oracle_actions
from opt_cost_oracle import CostOracle


def load_model_and_config(
    model_path: str = "model.joblib",
    meta_yaml: str = "model_meta.yaml"
) -> Tuple[Optional[object], List[str], Dict[str, int]]:
    """Load trained model and associated metadata.
    
    Args:
        model_path: Path to saved model file
        meta_yaml: Path to metadata YAML file
        
    Returns:
        Tuple of (model, feature_cols, action_mapping)
        
    Raises:
        FileNotFoundError: If required files don't exist
        ValueError: If metadata is incomplete
    """
    # Load model (may not exist for Mode B)
    model = None
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            print(f"[load_model_and_config] Loaded model from {model_path}")
        except Exception as e:
            print(f"[load_model_and_config] Warning: Could not load model: {e}")
    else:
        print(f"[load_model_and_config] No model file found at {model_path} (Mode B?)")

    # Load metadata
    if not os.path.exists(meta_yaml):
        raise FileNotFoundError(
            f"{meta_yaml} not found. Please run training first to generate metadata."
        )

    with open(meta_yaml, "r", encoding="utf-8") as f:
        meta = yaml.safe_load(f) or {}

    feature_cols = meta.get("feature_cols")
    action_mapping = meta.get("action_mapping")
    
    if feature_cols is None or action_mapping is None:
        raise ValueError(
            f"{meta_yaml} is missing required fields: 'feature_cols' and/or 'action_mapping'"
        )

    print(f"[load_model_and_config] Loaded metadata: {len(feature_cols)} features, "
          f"{len(action_mapping)} actions")
    
    return model, feature_cols, action_mapping


def recommend_actions_for_df(
    df_input: pd.DataFrame,
    model: Optional[object],
    cfg: Dict[str, Any],
    feature_cols: List[str],
    action_mapping: Dict[str, int]
) -> pd.DataFrame:
    """Generate action recommendations with explanations.
    
    Args:
        df_input: Input DataFrame with raw data
        model: Trained model (None for Mode B/oracle)
        cfg: Configuration dictionary
        feature_cols: List of feature column names
        action_mapping: Dict mapping action names to integers
        
    Returns:
        DataFrame with columns: time, recommended_action, reason
    """
    # Engineer features
    df_feat = engineer_features(df_input, cfg)
    
    # Determine prediction mode
    mode = cfg.get('model', {}).get('mode', 'A')
    
    # Generate predictions
    if model is not None and mode == 'A':
        # Mode A: Use trained ML model
        X = df_feat[feature_cols].fillna(0)
        y_pred = model.predict(X)
        inv_map = {v: k for k, v in action_mapping.items()}
        actions = [inv_map[i] for i in y_pred]
        print(f"[recommend_actions_for_df] Using trained {type(model).__name__} model")
    else:
        # Mode B or no model: Use oracle directly
        df_labeled = label_oracle_actions(df_feat, cfg)
        actions = df_labeled['recommended_action'].tolist()
        print("[recommend_actions_for_df] Using rule-based oracle")
    
    # Generate explanations for each action
    results = []
    for ts, action in zip(df_input.index, actions):
        reason = generate_reason(action, df_feat.loc[ts], cfg)
        results.append({
            "time": ts,
            "recommended_action": action,
            "reason": reason
        })
    
    res_df = pd.DataFrame(results)
    return res_df

def generate_reason(action: str, row: pd.Series, cfg: Dict[str, Any]) -> str:
    """Generate human-readable explanation for an action.
    
    Args:
        action: Recommended action name
        row: Feature row (Series) for the specific hour
        cfg: Configuration dictionary
        
    Returns:
        Human-readable explanation string
    """
    pv_surplus = row.get('pv_surplus', 0)
    price = row.get('price_eur_per_kwh', None)
    high_thr = row.get('price_high_thresh', None)
    low_thr = row.get('price_low_thresh', None)
    next_flag = row.get('next_day_pv_flag', 0)
    # Construct reason based on logic
    if action == "sell_to_grid":
        return "High PV surplus + high price ⇒ selling excess solar to grid for profit."
    elif action == "use_pv_direct":
        if pv_surplus > 0:
            return "PV surplus available + price not high ⇒ use solar energy locally."
        else:
            return "Using available solar to cover load."
    elif action == "charge_battery":
        if price is not None and low_thr is not None and price <= low_thr:
            if next_flag == 1:
                return "Cheap power now and low solar expected ⇒ charging battery for tomorrow."
            else:
                return "Low electricity price now ⇒ charge battery for later use."
        else:
            return "Storing energy in battery for later (policy override for future need)."
    elif action == "charge_ev":
        return "Cheap nighttime electricity ⇒ charge EV while rates are low."
    elif action == "idle":
        if pv_surplus < 0 and price is not None and high_thr is not None and price > high_thr:
            return "High price and no surplus ⇒ avoid extra usage (just meet demand)."
        else:
            return "No surplus and price moderate ⇒ no extra action (idle)."
    else:
        return "Action based on policy conditions."

def main() -> None:
    """CLI entry point for generating predictions.
    
    Usage:
        python predict.py [days]
    
    Args:
        days: Number of days to predict (default: 1)
    """
    import sys
    from fetch_data import get_data
    
    # Parse command line arguments
    days = 1
    if len(sys.argv) > 1:
        try:
            days = int(sys.argv[1])
        except ValueError:
            print(f"Warning: Invalid days argument '{sys.argv[1]}', using default: 1")
    
    # Load configuration
    try:
        with open("config.yaml", 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        raise SystemExit("ERROR: config.yaml not found")
    
    # Override days if specified
    if days:
        cfg.setdefault('location', {})['days'] = days
    
    print(f"[predict] Fetching data for {days} day(s)...")
    
    # Fetch data
    try:
        df_pred = get_data(cfg)
    except Exception as e:
        raise SystemExit(f"ERROR: Failed to fetch data: {e}")
    
    # Load model and metadata
    try:
        model, feature_cols, action_mapping = load_model_and_config()
    except FileNotFoundError as e:
        raise SystemExit(f"ERROR: {e}")
    
    # Generate recommendations
    print(f"[predict] Generating recommendations for {len(df_pred)} hours...")
    res_df = recommend_actions_for_df(df_pred, model, cfg, feature_cols, action_mapping)
    
    # Display results
    print(f"\n{'='*80}")
    print(f"Recommendations for {days} day(s)")
    print(f"{'='*80}\n")
    
    for _, row in res_df.iterrows():
        t_str = pd.to_datetime(row['time']).strftime("%Y-%m-%d %H:%M")
        print(f"{t_str}: {row['recommended_action']:15s} -- {row['reason']}")
    
    print(f"\n{'='*80}")
    print(f"Total: {len(res_df)} hourly recommendations")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
