"""Complete training and evaluation pipeline for prosumer energy advisor.

This script orchestrates the full workflow:
1. Data fetching
2. Feature engineering
3. Model training
4. Model evaluation
5. Visualization generation
"""

import os
from typing import Dict, Any

import yaml
import pandas as pd

from fetch_data import get_data
from features import prepare_dataset
from train import train_model, evaluate_model
from predict import recommend_actions_for_df
from visualize import plot_overview, plot_action_distribution, plot_energy_flow


def main() -> None:
    """Execute complete training and evaluation pipeline.
    
    Raises:
        FileNotFoundError: If config.yaml is missing
        SystemExit: If any pipeline step fails
    """
    print("="*80)
    print("Prosumer Energy Advisor - Training Pipeline")
    print("="*80 + "\n")
    
    # Load configuration
    try:
        with open("config.yaml", 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        raise SystemExit("ERROR: config.yaml not found")
    
    # Fetch data
    print("[1/7] Fetching data...")
    try:
        df_raw = get_data(cfg)
        print(f"      ✓ Fetched {len(df_raw)} hourly data points")
    except Exception as e:
        raise SystemExit(f"ERROR: Data fetching failed: {e}")
    
    # Prepare dataset (features + labels)
    print("\n[2/7] Preparing dataset (feature engineering + labeling)...")
    try:
        df = prepare_dataset(df_raw, cfg)
        print(f"      ✓ Prepared {len(df)} samples with {len(df.columns)} features")
    except Exception as e:
        raise SystemExit(f"ERROR: Dataset preparation failed: {e}")
    
    # Save price data for reference
    out_dir = cfg.get('visualization', {}).get('plot_dir', 'outputs')
    os.makedirs(out_dir, exist_ok=True)
    
    if "price_eur_per_kwh" in df_raw.columns:
        price_csv = os.path.join(out_dir, "prices_used.csv")
        df_raw["price_eur_per_kwh"].to_csv(price_csv)
        print(f"      ✓ Saved price data to {price_csv}")
    
    # Split data into train and test (time-based split)
    TRAIN_RATIO = 0.8
    split_idx = int(len(df) * TRAIN_RATIO)
    df_train = df.iloc[:split_idx]
    df_test = df.iloc[split_idx:]
    
    print(f"\n[3/7] Splitting data (train/test = {TRAIN_RATIO:.0%}/{1-TRAIN_RATIO:.0%})...")
    print(f"      Train: {df_train.index[0]} to {df_train.index[-1]} ({len(df_train)} samples)")
    print(f"      Test:  {df_test.index[0]} to {df_test.index[-1]} ({len(df_test)} samples)")
    
    # Train model
    print(f"\n[4/7] Training model...")
    try:
        model, feature_cols, action_mapping = train_model(df_train, cfg)
        print(f"      ✓ Model trained successfully")
    except Exception as e:
        raise SystemExit(f"ERROR: Model training failed: {e}")
    
    # Evaluate model
    print(f"\n[5/7] Evaluating model on test set...")
    try:
        evaluate_model(model, df_test, feature_cols, action_mapping)
    except Exception as e:
        print(f"      Warning: Evaluation failed: {e}")
    
    # Generate predictions for visualization
    print(f"\n[6/7] Generating predictions for visualization...")
    try:
        res_df = recommend_actions_for_df(df_test, model, cfg, feature_cols, action_mapping)
        df_test = df_test.copy()
        df_test['recommended_action'] = res_df.set_index('time')['recommended_action']
        print(f"      ✓ Generated {len(res_df)} predictions")
    except Exception as e:
        print(f"      Warning: Prediction generation failed: {e}")
        df_test['recommended_action'] = 'idle'  # Fallback
    
    # Generate visualizations
    print(f"\n[7/7] Creating visualizations...")
    viz_count = 0
    
    try:
        plot_overview(df_test, cfg)
        viz_count += 1
        print(f"      ✓ Created PV/Load overview plot")
    except Exception as e:
        print(f"      Warning: Overview plot failed: {e}")
    
    try:
        plot_action_distribution(df_test, cfg)
        viz_count += 1
        print(f"      ✓ Created action distribution plot")
    except Exception as e:
        print(f"      Warning: Action distribution plot failed: {e}")
    
    try:
        plot_energy_flow(df_test, cfg)
        viz_count += 1
        print(f"      ✓ Created energy flow plots")
    except Exception as e:
        print(f"      Warning: Energy flow plot failed: {e}")
    
    # Summary
    print("\n" + "="*80)
    print("Pipeline Complete!")
    print("="*80)
    print(f"✓ Model saved to: model.joblib")
    print(f"✓ Metadata saved to: model_meta.yaml")
    print(f"✓ Visualizations saved to: {out_dir}/ ({viz_count} plots)")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
