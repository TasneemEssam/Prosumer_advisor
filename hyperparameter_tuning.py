"""Hyperparameter optimization using Optuna.

This module provides automated hyperparameter tuning for the forecasting models.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score


def optimize_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 50,
    cv_folds: int = 3
) -> Dict[str, Any]:
    """Optimize RandomForest hyperparameters using Optuna.
    
    Requires: pip install optuna
    
    Args:
        X_train: Training features
        y_train: Training labels
        n_trials: Number of optimization trials
        cv_folds: Number of cross-validation folds
        
    Returns:
        Dict with best parameters and score
        
    Example:
        >>> best_params = optimize_random_forest(X_train, y_train)
        >>> model = RandomForestClassifier(**best_params['params'])
    """
    try:
        import optuna
    except ImportError:
        raise ImportError(
            "Optuna not installed. Install with:\n"
            "pip install optuna"
        )
    
    def objective(trial):
        """Optuna objective function."""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'random_state': 42,
            'n_jobs': -1
        }
        
        model = RandomForestClassifier(**params)
        score = cross_val_score(
            model, X_train, y_train,
            cv=cv_folds,
            scoring='accuracy',
            n_jobs=-1
        ).mean()
        
        return score
    
    print(f"[optuna] Starting hyperparameter optimization ({n_trials} trials)...")
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        study_name='random_forest_optimization'
    )
    
    # Optimize
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    print(f"\n[optuna] ✓ Optimization complete!")
    print(f"[optuna] Best score: {study.best_value:.4f}")
    print(f"[optuna] Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    return {
        'params': study.best_params,
        'score': study.best_value,
        'study': study
    }


def optimize_decision_tree(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 30,
    cv_folds: int = 3
) -> Dict[str, Any]:
    """Optimize DecisionTree hyperparameters.
    
    Args:
        X_train: Training features
        y_train: Training labels
        n_trials: Number of optimization trials
        cv_folds: Number of cross-validation folds
        
    Returns:
        Dict with best parameters and score
    """
    try:
        import optuna
    except ImportError:
        raise ImportError("Optuna not installed")
    
    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 2, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
            'random_state': 42
        }
        
        model = DecisionTreeClassifier(**params)
        score = cross_val_score(
            model, X_train, y_train,
            cv=cv_folds,
            scoring='accuracy'
        ).mean()
        
        return score
    
    print(f"[optuna] Starting DecisionTree optimization ({n_trials} trials)...")
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\n[optuna] ✓ Best score: {study.best_value:.4f}")
    print(f"[optuna] Best parameters: {study.best_params}")
    
    return {
        'params': study.best_params,
        'score': study.best_value,
        'study': study
    }


def visualize_optimization(study, output_dir: str = "outputs") -> None:
    """Create optimization visualization plots.
    
    Args:
        study: Optuna study object
        output_dir: Directory to save plots
    """
    try:
        import optuna
        from pathlib import Path
    except ImportError:
        raise ImportError("Optuna not installed")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Optimization history
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_html(f"{output_dir}/optimization_history.html")
    
    # Parameter importances
    fig = optuna.visualization.plot_param_importances(study)
    fig.write_html(f"{output_dir}/param_importances.html")
    
    # Parallel coordinate plot
    fig = optuna.visualization.plot_parallel_coordinate(study)
    fig.write_html(f"{output_dir}/parallel_coordinate.html")
    
    print(f"[optuna] ✓ Visualizations saved to {output_dir}/")


if __name__ == "__main__":
    import yaml
    from fetch_data import get_data
    from features import prepare_dataset
    
    # Load config and data
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    
    print("Fetching and preparing data...")
    df_raw = get_data(cfg)
    df = prepare_dataset(df_raw, cfg)
    
    # Prepare features
    feature_cols = [
        'pv_surplus', 'price_eur_per_kwh', 'hour', 'dow',
        'temp_C', 'cloudcover_pct', 'next_day_pv_flag',
        'load_lag1', 'load_lag2', 'pv_lag1',
        'roll_load_3h', 'roll_pv_6h'
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    # Create action mapping
    action_mapping = {a: i for i, a in enumerate(df['recommended_action'].unique())}
    
    X = df[feature_cols].fillna(0)
    y = df['recommended_action'].map(action_mapping)
    
    # Optimize RandomForest
    result = optimize_random_forest(X, y, n_trials=50)
    
    # Visualize results
    visualize_optimization(result['study'])
    
    print("\n✓ Hyperparameter optimization complete!")
    print(f"  Best accuracy: {result['score']:.4f}")
    print(f"  Visualizations: outputs/optimization_*.html")