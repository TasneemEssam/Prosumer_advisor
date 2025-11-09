"""Experiment tracking using Weights & Biases (wandb).

This module provides MLOps capabilities for tracking experiments, models, and metrics.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


class WandbExperimentTracker:
    """Track ML experiments with Weights & Biases.
    
    Requires: pip install wandb
    
    Features:
    - Automatic metric logging
    - Hyperparameter tracking
    - Model versioning
    - Visualization dashboards
    """
    
    def __init__(
        self,
        project_name: str = "prosumer-energy-advisor",
        entity: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize experiment tracker.
        
        Args:
            project_name: W&B project name
            entity: W&B entity (username or team)
            config: Configuration dictionary to log
        """
        try:
            import wandb
            self.wandb = wandb
        except ImportError:
            raise ImportError(
                "wandb not installed. Install with:\n"
                "pip install wandb\n"
                "Then login with: wandb login"
            )
        
        self.project_name = project_name
        self.entity = entity
        self.run = None
        self.config = config or {}
    
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[list] = None,
        notes: Optional[str] = None
    ):
        """Start a new experiment run.
        
        Args:
            run_name: Name for this run
            tags: List of tags for organization
            notes: Description of the experiment
        """
        self.run = self.wandb.init(
            project=self.project_name,
            entity=self.entity,
            name=run_name,
            config=self.config,
            tags=tags or [],
            notes=notes
        )
        
        print(f"[wandb] ✓ Started run: {self.run.name}")
        print(f"[wandb] Dashboard: {self.run.url}")
        
        return self.run
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to W&B.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number (e.g., epoch)
        """
        if self.run is None:
            raise RuntimeError("No active run. Call start_run() first.")
        
        self.wandb.log(metrics, step=step)
    
    def log_model(self, model_path: str, model_name: str = "model"):
        """Log trained model as artifact.
        
        Args:
            model_path: Path to model file
            model_name: Name for the model artifact
        """
        if self.run is None:
            raise RuntimeError("No active run")
        
        artifact = self.wandb.Artifact(model_name, type='model')
        artifact.add_file(model_path)
        self.run.log_artifact(artifact)
        
        print(f"[wandb] ✓ Logged model: {model_name}")
    
    def log_dataframe(self, df: pd.DataFrame, name: str = "data"):
        """Log DataFrame as W&B Table.
        
        Args:
            df: DataFrame to log
            name: Name for the table
        """
        if self.run is None:
            raise RuntimeError("No active run")
        
        table = self.wandb.Table(dataframe=df)
        self.wandb.log({name: table})
    
    def log_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: list
    ):
        """Log confusion matrix visualization.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
        """
        if self.run is None:
            raise RuntimeError("No active run")
        
        self.wandb.log({
            "confusion_matrix": self.wandb.plot.confusion_matrix(
                probs=None,
                y_true=y_true,
                preds=y_pred,
                class_names=class_names
            )
        })
    
    def finish(self):
        """Finish the current run."""
        if self.run:
            self.run.finish()
            print("[wandb] ✓ Run finished")


def train_with_tracking(df_train: pd.DataFrame, cfg: Dict[str, Any]):
    """Example: Train model with W&B tracking.
    
    Args:
        df_train: Training data
        cfg: Configuration dictionary
    """
    from train import train_model
    from sklearn.model_selection import train_test_split
    
    # Initialize tracker
    tracker = WandbExperimentTracker(config=cfg)
    tracker.start_run(
        run_name=f"experiment_{cfg.get('model', {}).get('algorithm', 'RF')}",
        tags=["training", cfg.get('model', {}).get('mode', 'A')],
        notes="Training with enhanced features"
    )
    
    # Split data for validation
    train_idx = int(len(df_train) * 0.8)
    df_t = df_train.iloc[:train_idx]
    df_v = df_train.iloc[train_idx:]
    
    # Train model
    model, feature_cols, action_mapping = train_model(df_t, cfg)
    
    # Evaluate on validation set
    from train import evaluate_model
    evaluate_model(model, df_v, feature_cols, action_mapping)
    
    # Log metrics
    if model is not None:
        from sklearn.metrics import accuracy_score, f1_score
        
        X_val = df_v[feature_cols].fillna(0)
        y_val = df_v['recommended_action'].map(action_mapping)
        y_pred = model.predict(X_val)
        
        metrics = {
            "accuracy": accuracy_score(y_val, y_pred),
            "f1_macro": f1_score(y_val, y_pred, average='macro'),
            "train_samples": len(df_t),
            "val_samples": len(df_v)
        }
        
        tracker.log_metrics(metrics)
        
        # Log confusion matrix
        inv_map = {v: k for k, v in action_mapping.items()}
        class_names = [inv_map[i] for i in sorted(inv_map.keys())]
        tracker.log_confusion_matrix(y_val, y_pred, class_names)
        
        # Log model
        tracker.log_model("model.joblib", "prosumer_model")
    
    tracker.finish()
    
    print("\n✓ Training with W&B tracking complete!")
    print("  View results at: https://wandb.ai")


if __name__ == "__main__":
    import yaml
    from fetch_data import get_data
    from features import prepare_dataset
    
    # Load config
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    
    # Fetch and prepare data
    print("Fetching data...")
    df_raw = get_data(cfg)
    df = prepare_dataset(df_raw, cfg)
    
    # Train with tracking
    train_with_tracking(df, cfg)