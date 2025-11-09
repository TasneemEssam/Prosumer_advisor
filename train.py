"""Model training module for prosumer energy advisor.

This module handles training of machine learning models for action classification,
including feature selection, model training, and evaluation.
"""

from typing import Dict, Any, Tuple, Optional, List
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import yaml


def train_model(
    df: pd.DataFrame,
    cfg: Dict[str, Any]
) -> Tuple[Optional[object], List[str], Dict[str, int]]:
    """Train machine learning model for action classification.
    
    Supports two modes:
    - Mode A: Classification model (DecisionTree or RandomForest)
    - Mode B: Direct optimization (no ML model, uses oracle at prediction time)
    
    Args:
        df: Training DataFrame with features and 'recommended_action' target
        cfg: Configuration dictionary with model settings
        
    Returns:
        Tuple of (trained_model, feature_columns, action_mapping)
        - trained_model: Fitted sklearn model (None for Mode B)
        - feature_columns: List of feature column names used
        - action_mapping: Dict mapping action names to integer labels
        
    Raises:
        ValueError: If mode is unknown or required columns are missing
    """
    model_cfg = cfg.get('model', {})
    mode = model_cfg.get('mode', 'A')
    target_col = 'recommended_action'
    
    # Validate target column exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")
    
    # Define candidate feature columns
    candidate_features = [
        'pv_surplus', 'price_eur_per_kwh', 'hour', 'dow',
        'temp_C', 'cloudcover_pct', 'next_day_pv_flag',
        'load_lag1', 'load_lag2', 'pv_lag1',
        'roll_load_3h', 'roll_pv_6h'
    ]
    
    # Select only features that exist and have non-null values
    feature_cols = [
        col for col in candidate_features
        if col in df.columns and df[col].notna().any()
    ]
    
    if not feature_cols:
        raise ValueError("No valid feature columns found in DataFrame")
    
    # Prepare training data
    df_train = df.copy()
    
    # Create action mapping (consistent ordering)
    unique_actions = sorted(df_train[target_col].unique())
    action_mapping = {action: idx for idx, action in enumerate(unique_actions)}
    
    # Encode target variable
    y = df_train[target_col].map(action_mapping)
    X = df_train[feature_cols]
    
    # Handle missing values (forward fill, backward fill, then zero)
    X = X.ffill().bfill().fillna(0)
    
    # Train model based on mode
    model = None
    
    if mode == 'A':
        # Mode A: Classification model
        algo = model_cfg.get('algorithm', 'RandomForest')
        random_seed = model_cfg.get('random_seed', 42)
        
        if algo == 'DecisionTree':
            max_depth = model_cfg.get('tree_max_depth', None)
            model = DecisionTreeClassifier(
                max_depth=max_depth,
                random_state=random_seed,
                min_samples_split=5,  # Prevent overfitting
                min_samples_leaf=2
            )
        elif algo == 'RandomForest':
            n_estimators = model_cfg.get('rf_n_estimators', 100)
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=random_seed,
                max_depth=10,  # Prevent overfitting
                min_samples_split=5,
                min_samples_leaf=2,
                n_jobs=-1  # Use all CPU cores
            )
        else:
            raise ValueError(f"Unknown algorithm: {algo}")
        
        print(f"[train_model] Training {algo} with {len(feature_cols)} features on {len(X)} samples...")
        model.fit(X, y)
        print(f"[train_model] Training complete. Model score: {model.score(X, y):.3f}")
        
    elif mode == 'B':
        # Mode B: Direct optimization (no ML model)
        print("[train_model] Mode B: Using direct optimization oracle (no ML model)")
        model = None
    else:
        raise ValueError(f"Unknown mode: {mode}. Expected 'A' or 'B'")
    
    # Save model to disk
    if model is not None:
        joblib.dump(model, "model.joblib")
        print("[train_model] Saved model to model.joblib")
    
    # Save metadata
    meta = {
        "feature_cols": feature_cols,
        "action_mapping": action_mapping,
        "mode": mode,
        "algorithm": model_cfg.get('algorithm', 'RandomForest') if mode == 'A' else None
    }
    
    with open("model_meta.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f)
    print("[train_model] Saved metadata to model_meta.yaml")
    
    return model, feature_cols, action_mapping

def evaluate_model(
    model: Optional[object],
    df_test: pd.DataFrame,
    feature_cols: List[str],
    action_mapping: Dict[str, int]
) -> Optional[pd.DataFrame]:
    """Evaluate trained model on test data.
    
    Computes and displays classification metrics including:
    - Precision, recall, F1-score per action
    - Overall accuracy
    - Confusion matrix
    
    Args:
        model: Trained sklearn model (None for Mode B)
        df_test: Test DataFrame with features and 'recommended_action'
        feature_cols: List of feature column names
        action_mapping: Dict mapping action names to integer labels
        
    Returns:
        Confusion matrix as DataFrame, or None if no model/data
    """
    if model is None:
        print("[evaluate_model] No model to evaluate (Mode B uses oracle directly)")
        return None
    
    if df_test is None or df_test.empty:
        print("[evaluate_model] No test data provided")
        return None
    
    # Prepare test data
    inv_map = {v: k for k, v in action_mapping.items()}
    X_test = df_test[feature_cols].fillna(0)
    y_true = df_test['recommended_action'].map(action_mapping)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = (y_pred == y_true).mean()
    print(f"\n{'='*60}")
    print(f"Model Evaluation Results")
    print(f"{'='*60}")
    print(f"Test samples: {len(y_true)}")
    print(f"Overall accuracy: {accuracy:.3f}")
    print(f"{'='*60}\n")
    
    # Classification report
    print("Classification Report:")
    print("-" * 60)
    target_names = [inv_map[i] for i in sorted(inv_map.keys())]
    print(classification_report(
        y_true, y_pred,
        target_names=target_names,
        digits=3,
        zero_division=0
    ))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    print("-" * 60)
    cm = confusion_matrix(y_true, y_pred)
    labels = [inv_map[i] for i in sorted(inv_map.keys())]
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print(cm_df)
    print(f"{'='*60}\n")
    
    return cm_df
