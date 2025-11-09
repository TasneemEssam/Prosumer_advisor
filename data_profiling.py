"""Data profiling and exploratory data analysis using ydata-profiling.

This module provides automated EDA reports for understanding the dataset.
"""

from typing import Dict, Any, Optional
import pandas as pd
from pathlib import Path


def generate_profile_report(
    df: pd.DataFrame,
    title: str = "Prosumer Energy Data Profile",
    output_file: str = "outputs/data_profile.html",
    minimal: bool = False
) -> None:
    """Generate comprehensive data profiling report.
    
    Requires: pip install ydata-profiling
    
    Args:
        df: DataFrame to profile
        title: Report title
        output_file: Output HTML file path
        minimal: If True, generate minimal report (faster)
        
    Example:
        >>> from fetch_data import get_data
        >>> df = get_data(cfg)
        >>> generate_profile_report(df)
    """
    try:
        from ydata_profiling import ProfileReport
    except ImportError:
        raise ImportError(
            "ydata-profiling not installed. Install with:\n"
            "pip install ydata-profiling"
        )
    
    # Create output directory
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Configure report
    config = {
        "title": title,
        "minimal": minimal,
        "explorative": not minimal,
        "correlations": {
            "pearson": {"calculate": True},
            "spearman": {"calculate": True},
            "kendall": {"calculate": False}  # Slow for large datasets
        },
        "missing_diagrams": {
            "heatmap": True,
            "dendrogram": True
        }
    }
    
    print(f"[data_profiling] Generating {'minimal' if minimal else 'full'} profile report...")
    
    # Generate report
    profile = ProfileReport(df, **config)
    
    # Save to file
    profile.to_file(output_file)
    
    print(f"[data_profiling] ✓ Report saved to: {output_file}")
    print(f"[data_profiling] Open in browser to view detailed analysis")


def quick_profile(df: pd.DataFrame) -> None:
    """Generate quick minimal profile (faster for large datasets).
    
    Args:
        df: DataFrame to profile
    """
    generate_profile_report(df, minimal=True, output_file="outputs/quick_profile.html")


def compare_datasets(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    labels: tuple = ("Training", "Test"),
    output_file: str = "outputs/comparison_report.html"
) -> None:
    """Compare two datasets (e.g., train vs test).
    
    Args:
        df1: First dataset
        df2: Second dataset
        labels: Labels for the datasets
        output_file: Output HTML file path
    """
    try:
        from ydata_profiling import ProfileReport
    except ImportError:
        raise ImportError("ydata-profiling not installed")
    
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"[data_profiling] Comparing {labels[0]} vs {labels[1]}...")
    
    # Generate comparison report
    profile1 = ProfileReport(df1, title=labels[0], minimal=True)
    profile2 = ProfileReport(df2, title=labels[1], minimal=True)
    
    comparison = profile1.compare(profile2)
    comparison.to_file(output_file)
    
    print(f"[data_profiling] ✓ Comparison report saved to: {output_file}")


if __name__ == "__main__":
    import yaml
    from fetch_data import get_data
    from features import prepare_dataset
    
    # Load config
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    
    # Fetch data
    print("Fetching data...")
    df_raw = get_data(cfg)
    
    # Generate profile
    generate_profile_report(
        df_raw,
        title="Prosumer Energy Raw Data Profile",
        output_file="outputs/raw_data_profile.html"
    )
    
    # Prepare dataset and profile features
    df_features = prepare_dataset(df_raw, cfg)
    generate_profile_report(
        df_features,
        title="Prosumer Energy Features Profile",
        output_file="outputs/features_profile.html"
    )
    
    print("\n✓ Data profiling complete!")
    print("  - outputs/raw_data_profile.html")
    print("  - outputs/features_profile.html")