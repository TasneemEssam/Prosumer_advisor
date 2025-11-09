"""Interactive Streamlit dashboard for Prosumer Energy Advisor.

Run with: streamlit run streamlit_app.py

Requires: pip install streamlit plotly
"""

import streamlit as st
import pandas as pd
import numpy as np
import yaml
from datetime import datetime, timedelta
from pathlib import Path


# Page configuration
st.set_page_config(
    page_title="Prosumer Energy Advisor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_config():
    """Load configuration file."""
    try:
        with open("config.yaml") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        st.error("config.yaml not found!")
        return {}


def main():
    """Main Streamlit application."""
    
    # Header
    st.title("⚡ Prosumer Energy Advisor")
    st.markdown("**AI-powered energy management for solar prosumers**")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Load config
        cfg = load_config()
        
        # Location settings
        st.subheader("📍 Location")
        lat = st.number_input(
            "Latitude",
            value=cfg.get("location", {}).get("lat", 52.52),
            format="%.2f"
        )
        lon = st.number_input(
            "Longitude",
            value=cfg.get("location", {}).get("lon", 13.405),
            format="%.2f"
        )
        
        # PV system
        st.subheader("☀️ PV System")
        peak_power = st.slider(
            "Peak Power (kW)",
            min_value=1.0,
            max_value=20.0,
            value=float(cfg.get("pv_system", {}).get("peak_power_kw", 5.0)),
            step=0.5
        )
        
        # Date range
        st.subheader("📅 Analysis Period")
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=7)
        )
        days = st.slider("Number of Days", 1, 30, 7)
        
        # Action button
        run_analysis = st.button("🚀 Run Analysis", type="primary")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dashboard",
        "🔮 Predictions",
        "📈 Analytics",
        "ℹ️ About"
    ])
    
    with tab1:
        show_dashboard(cfg, run_analysis)
    
    with tab2:
        show_predictions(cfg)
    
    with tab3:
        show_analytics(cfg)
    
    with tab4:
        show_about()


def show_dashboard(cfg, run_analysis):
    """Show main dashboard."""
    st.header("Energy Management Dashboard")
    
    if run_analysis:
        with st.spinner("Fetching data and generating recommendations..."):
            try:
                # Import here to avoid errors if modules not available
                from fetch_data import get_data
                from features import prepare_dataset
                from predict import load_model_and_config, recommend_actions_for_df
                
                # Fetch data
                df_raw = get_data(cfg)
                df = prepare_dataset(df_raw, cfg)
                
                # Load model and predict
                model, feature_cols, action_mapping = load_model_and_config()
                results = recommend_actions_for_df(df, model, cfg, feature_cols, action_mapping)
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Total PV Generation",
                        f"{df['pv_kW'].sum():.1f} kWh",
                        delta=f"{df['pv_kW'].mean():.2f} kW avg"
                    )
                
                with col2:
                    st.metric(
                        "Total Load",
                        f"{df['load_kW'].sum():.1f} kWh",
                        delta=f"{df['load_kW'].mean():.2f} kW avg"
                    )
                
                with col3:
                    avg_price = df['price_eur_per_kwh'].mean()
                    st.metric(
                        "Avg Price",
                        f"€{avg_price:.3f}/kWh",
                        delta=f"€{avg_price*100:.1f} ct/kWh"
                    )
                
                with col4:
                    self_consumption = min(df['pv_kW'].sum(), df['load_kW'].sum())
                    st.metric(
                        "Self-Consumption",
                        f"{self_consumption:.1f} kWh",
                        delta=f"{100*self_consumption/df['pv_kW'].sum():.1f}%"
                    )
                
                # Action distribution
                st.subheader("Recommended Actions Distribution")
                action_counts = results['recommended_action'].value_counts()
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Create plotly chart
                    try:
                        import plotly.express as px
                        fig = px.pie(
                            values=action_counts.values,
                            names=action_counts.index,
                            title="Action Distribution"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except ImportError:
                        st.bar_chart(action_counts)
                
                with col2:
                    st.dataframe(
                        action_counts.to_frame("Hours"),
                        use_container_width=True
                    )
                
                # Time series plot
                st.subheader("PV Generation vs Load")
                
                chart_data = pd.DataFrame({
                    'PV Generation': df['pv_kW'],
                    'Load': df['load_kW']
                }, index=df.index)
                
                st.line_chart(chart_data)
                
                # Recent recommendations
                st.subheader("Recent Recommendations")
                st.dataframe(
                    results.tail(24)[['time', 'recommended_action', 'reason']],
                    use_container_width=True,
                    hide_index=True
                )
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Make sure you have run the training pipeline first: `python run_pipeline.py`")
    else:
        st.info("👈 Configure settings in the sidebar and click 'Run Analysis'")


def show_predictions(cfg):
    """Show prediction interface."""
    st.header("Tomorrow's Predictions")
    
    if st.button("Generate Tomorrow's Forecast"):
        with st.spinner("Generating 24-hour forecast..."):
            try:
                from predict_tomorrow import main as predict_tomorrow_main
                
                # Run prediction
                predict_tomorrow_main()
                
                # Load results
                tomorrow = (datetime.now() + timedelta(days=1)).date()
                output_file = f"outputs/predictions_{tomorrow.isoformat()}.csv"
                
                if Path(output_file).exists():
                    df_pred = pd.read_csv(output_file, index_col=0, parse_dates=True)
                    
                    st.success(f"✓ Forecast generated for {tomorrow}")
                    
                    # Display forecast
                    st.subheader("Hourly Forecast")
                    
                    # Plot
                    if 'pv_kW' in df_pred.columns and 'load_kW' in df_pred.columns:
                        chart_data = pd.DataFrame({
                            'PV Generation': df_pred['pv_kW'],
                            'Load': df_pred['load_kW']
                        })
                        st.line_chart(chart_data)
                    
                    # Actions
                    if 'recommended_action' in df_pred.columns:
                        st.subheader("Recommended Actions")
                        action_counts = df_pred['recommended_action'].value_counts()
                        st.bar_chart(action_counts)
                        
                        # Detailed table
                        st.dataframe(
                            df_pred[['pv_kW', 'load_kW', 'recommended_action']].head(24),
                            use_container_width=True
                        )
                else:
                    st.warning("Prediction file not found")
                    
            except Exception as e:
                st.error(f"Error generating forecast: {str(e)}")


def show_analytics(cfg):
    """Show analytics and insights."""
    st.header("Advanced Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Data Profiling")
        if st.button("Generate Data Profile Report"):
            with st.spinner("Generating comprehensive data profile..."):
                try:
                    from data_profiling import generate_profile_report
                    from fetch_data import get_data
                    
                    df = get_data(cfg)
                    generate_profile_report(df, minimal=True)
                    
                    st.success("✓ Profile report generated!")
                    st.info("Open `outputs/data_profile.html` in your browser")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    with col2:
        st.subheader("🎯 Hyperparameter Tuning")
        if st.button("Optimize Model Parameters"):
            with st.spinner("Running Optuna optimization..."):
                try:
                    st.info("This may take several minutes...")
                    from hyperparameter_tuning import optimize_random_forest
                    from fetch_data import get_data
                    from features import prepare_dataset
                    
                    df_raw = get_data(cfg)
                    df = prepare_dataset(df_raw, cfg)
                    
                    # Prepare data
                    feature_cols = [
                        'pv_surplus', 'price_eur_per_kwh', 'hour', 'dow'
                    ]
                    feature_cols = [c for c in feature_cols if c in df.columns]
                    
                    action_mapping = {a: i for i, a in enumerate(df['recommended_action'].unique())}
                    X = df[feature_cols].fillna(0)
                    y = df['recommended_action'].map(action_mapping)
                    
                    result = optimize_random_forest(X, y, n_trials=20)
                    
                    st.success(f"✓ Best accuracy: {result['score']:.4f}")
                    st.json(result['params'])
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")


def show_about():
    """Show about information."""
    st.header("About Prosumer Energy Advisor")
    
    st.markdown("""
    ### 🎯 Overview
    
    The Prosumer Energy Advisor is an AI-powered system that helps solar prosumers 
    (consumers with PV generation) optimize their energy usage and costs.
    
    ### ✨ Features
    
    - **Smart Recommendations**: ML-based action suggestions
    - **Real-time Pricing**: Integration with ENTSO-E electricity prices
    - **Weather Forecasting**: Solar and weather data from Open-Meteo
    - **Multiple Modes**: Classification and optimization approaches
    - **Interactive Dashboard**: This Streamlit interface!
    
    ### 🚀 Actions
    
    The system recommends five types of actions:
    
    1. **Use PV Direct**: Consume solar energy locally
    2. **Sell to Grid**: Export excess during high prices
    3. **Charge Battery**: Store energy for later use
    4. **Charge EV**: Charge electric vehicle during cheap periods
    5. **Idle**: No special action needed
    
    ### 📚 Documentation
    
    - [README.md](README.md) - Quick start guide
    - [ADVANCED_FORECASTING.md](ADVANCED_FORECASTING.md) - Advanced features
    - [CHANGELOG.md](CHANGELOG.md) - Version history
    
    ### 🛠️ Technology Stack
    
    - **ML**: scikit-learn, Darts, Chronos
    - **Data**: pandas, numpy
    - **APIs**: ENTSO-E, PVGIS, Open-Meteo
    - **Visualization**: matplotlib, plotly, streamlit
    - **MLOps**: wandb, optuna, ydata-profiling
    
    ### 📄 License
    
    Educational and research purposes.
    
    ---
    
    **Version**: 2.0 (Enhanced)  
    **Last Updated**: 2025-11-08
    """)


if __name__ == "__main__":
    main()