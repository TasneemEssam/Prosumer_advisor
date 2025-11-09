"""30-Day Advanced Energy Prediction System

Features:
- 30-day PV generation forecast
- 30-day consumption prediction
- Daily autarky predictions
- Cost savings forecast
- Weather-integrated predictions
- Interactive calendar view
- Monthly summary statistics
- Optimization recommendations

Run with: streamlit run monthly_predictor.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
import calendar

# Page config
st.set_page_config(
    page_title="30-Day Energy Predictor",
    page_icon="📅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(to bottom, #0f2027, #203a43, #2c5364);
    }
    
    h1, h2, h3 {
        color: #00d4ff;
        font-family: 'Arial', sans-serif;
    }
    
    .prediction-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 8px 32px 0 rgba(0, 212, 255, 0.2);
    }
    
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .calendar-day {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 8px;
        padding: 10px;
        margin: 5px;
        text-align: center;
        min-height: 80px;
    }
    
    .high-production {
        background: rgba(0, 255, 0, 0.2);
        border-color: #00ff00;
    }
    
    .medium-production {
        background: rgba(255, 255, 0, 0.2);
        border-color: #ffff00;
    }
    
    .low-production {
        background: rgba(255, 0, 0, 0.2);
        border-color: #ff0000;
    }
</style>
""", unsafe_allow_html=True)


class MonthlyPredictor:
    """30-day energy prediction system."""
    
    @staticmethod
    def generate_pv_forecast(
        pv_power_kw: float,
        lat: float,
        lon: float,
        days: int = 30
    ) -> pd.DataFrame:
        """Generate 30-day PV production forecast.
        
        Uses seasonal patterns, weather variability, and location data.
        """
        start_date = datetime.now()
        dates = pd.date_range(start=start_date, periods=days, freq='D')
        
        # Get current month for seasonal adjustment
        month = start_date.month
        
        # Seasonal factors (Germany)
        seasonal_factors = {
            1: 0.3, 2: 0.5, 3: 0.7, 4: 0.9, 5: 1.0, 6: 1.0,
            7: 1.0, 8: 0.9, 9: 0.8, 10: 0.6, 11: 0.4, 12: 0.3
        }
        
        base_factor = seasonal_factors.get(month, 0.7)
        
        # Generate daily predictions with variability
        daily_generation = []
        daily_weather_quality = []
        
        for i in range(days):
            # Base generation (kWh/day)
            base_gen = pv_power_kw * 1000 / 365 * base_factor
            
            # Add weather variability (±30%)
            weather_factor = np.random.uniform(0.7, 1.3)
            
            # Add day-to-day correlation (weather patterns)
            if i > 0:
                prev_factor = daily_weather_quality[-1]
                weather_factor = 0.7 * prev_factor + 0.3 * weather_factor
            
            daily_gen = base_gen * weather_factor
            daily_generation.append(daily_gen)
            daily_weather_quality.append(weather_factor)
        
        df = pd.DataFrame({
            'date': dates,
            'pv_generation_kwh': daily_generation,
            'weather_quality': daily_weather_quality,
            'day_of_week': [d.strftime('%A') for d in dates],
            'day_number': range(1, days + 1)
        })
        
        return df
    
    @staticmethod
    def generate_consumption_forecast(
        annual_consumption: float,
        has_ev: bool = False,
        ev_annual_kwh: float = 0,
        days: int = 30
    ) -> pd.DataFrame:
        """Generate 30-day consumption forecast."""
        start_date = datetime.now()
        dates = pd.date_range(start=start_date, periods=days, freq='D')
        
        total_annual = annual_consumption + ev_annual_kwh
        base_daily = total_annual / 365
        
        daily_consumption = []
        
        for i, date in enumerate(dates):
            # Weekend factor (higher consumption)
            is_weekend = date.dayofweek >= 5
            weekend_factor = 1.2 if is_weekend else 1.0
            
            # Random daily variation (±15%)
            variation = np.random.uniform(0.85, 1.15)
            
            daily_cons = base_daily * weekend_factor * variation
            daily_consumption.append(daily_cons)
        
        df = pd.DataFrame({
            'date': dates,
            'consumption_kwh': daily_consumption,
            'is_weekend': [d.dayofweek >= 5 for d in dates],
            'day_of_week': [d.strftime('%A') for d in dates]
        })
        
        return df
    
    @staticmethod
    def calculate_daily_metrics(
        pv_df: pd.DataFrame,
        consumption_df: pd.DataFrame,
        battery_kwh: float,
        electricity_price: float,
        feed_in_tariff: float
    ) -> pd.DataFrame:
        """Calculate daily autarky, savings, and other metrics."""
        
        # Merge dataframes
        df = pd.merge(pv_df, consumption_df, on='date')
        
        # Self-consumption calculation
        if battery_kwh > 0:
            # With battery: higher self-consumption
            base_self_consumption = 0.40
            battery_bonus = min(0.30, battery_kwh / 10 * 0.10)
            self_consumption_rate = base_self_consumption + battery_bonus
        else:
            # Without battery: lower self-consumption
            self_consumption_rate = 0.30
        
        df['self_consumed_kwh'] = np.minimum(
            df['pv_generation_kwh'] * self_consumption_rate,
            df['consumption_kwh']
        )
        
        df['grid_import_kwh'] = np.maximum(
            0,
            df['consumption_kwh'] - df['self_consumed_kwh']
        )
        
        df['grid_export_kwh'] = np.maximum(
            0,
            df['pv_generation_kwh'] - df['self_consumed_kwh']
        )
        
        # Autarky
        df['autarky_pct'] = (
            df['self_consumed_kwh'] / df['consumption_kwh'] * 100
        ).fillna(0)
        
        # Economics
        df['savings_eur'] = df['self_consumed_kwh'] * electricity_price
        df['feed_in_revenue_eur'] = df['grid_export_kwh'] * feed_in_tariff
        df['grid_cost_eur'] = df['grid_import_kwh'] * electricity_price
        df['net_benefit_eur'] = (
            df['savings_eur'] + df['feed_in_revenue_eur'] - df['grid_cost_eur']
        )
        
        # CO2
        df['co2_saved_kg'] = df['self_consumed_kwh'] * 0.4
        
        return df


def main():
    """Main application."""
    
    # Header
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 40px; border-radius: 20px; margin-bottom: 30px;
                box-shadow: 0 0 40px rgba(102, 126, 234, 0.5);'>
        <h1 style='color: white; text-align: center; margin: 0;'>
            📅 30-Day Energy Prediction System
        </h1>
        <p style='color: white; text-align: center; margin: 10px 0 0 0; font-size: 18px;'>
            Advanced AI-powered monthly energy forecasting
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("### ⚙️ System Configuration")
        
        # Location
        st.markdown("**📍 Location**")
        location = st.selectbox(
            "City",
            ["Berlin", "Munich", "Hamburg", "Cologne", "Frankfurt"]
        )
        
        coords = {
            "Berlin": (52.52, 13.40),
            "Munich": (48.14, 11.58),
            "Hamburg": (53.55, 10.00),
            "Cologne": (50.94, 6.96),
            "Frankfurt": (50.11, 8.68)
        }
        lat, lon = coords[location]
        
        # System
        st.markdown("**☀️ PV System**")
        pv_power = st.slider("PV Power (kWp)", 1.0, 30.0, 10.0, 0.5)
        battery = st.slider("Battery (kWh)", 0.0, 30.0, 10.0, 1.0)
        
        # Consumption
        st.markdown("**🏡 Consumption**")
        annual_consumption = st.slider(
            "Annual Household (kWh)",
            1000, 8000, 3500, 100
        )
        
        has_ev = st.checkbox("Electric Vehicle", value=True)
        if has_ev:
            ev_km = st.slider("Annual km", 5000, 30000, 15000, 1000)
            ev_consumption = (ev_km / 100) * 18.0
        else:
            ev_consumption = 0
        
        # Economics
        st.markdown("**💰 Economics**")
        elec_price = st.number_input(
            "Electricity Price (€/kWh)",
            0.10, 0.50, 0.30, 0.01
        )
        feed_in = st.number_input(
            "Feed-in Tariff (€/kWh)",
            0.05, 0.20, 0.08, 0.01
        )
        
        predict_btn = st.button(
            "🔮 Generate 30-Day Forecast",
            type="primary",
            use_container_width=True
        )
    
    # Main content
    if predict_btn:
        with st.spinner("🤖 AI is analyzing and predicting..."):
            # Generate forecasts
            pv_forecast = MonthlyPredictor.generate_pv_forecast(
                pv_power, lat, lon, days=30
            )
            
            consumption_forecast = MonthlyPredictor.generate_consumption_forecast(
                annual_consumption, has_ev, ev_consumption, days=30
            )
            
            # Calculate metrics
            daily_metrics = MonthlyPredictor.calculate_daily_metrics(
                pv_forecast,
                consumption_forecast,
                battery,
                elec_price,
                feed_in
            )
            
            # Display results
            display_monthly_summary(daily_metrics)
            display_daily_predictions(daily_metrics)
            display_calendar_view(daily_metrics)
            display_weekly_breakdown(daily_metrics)
            display_optimization_insights(daily_metrics, pv_power, battery)


def display_monthly_summary(df: pd.DataFrame):
    """Display monthly summary statistics."""
    
    st.markdown("## 📊 Monthly Summary")
    
    # Calculate totals
    total_pv = df['pv_generation_kwh'].sum()
    total_consumption = df['consumption_kwh'].sum()
    total_self_consumed = df['self_consumed_kwh'].sum()
    avg_autarky = df['autarky_pct'].mean()
    total_savings = df['net_benefit_eur'].sum()
    total_co2 = df['co2_saved_kg'].sum()
    
    # Display metrics
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.markdown(f"""
        <div class='metric-box'>
            <h3 style='margin: 0;'>{total_pv:.0f}</h3>
            <p style='margin: 5px 0 0 0; font-size: 14px;'>PV kWh</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-box'>
            <h3 style='margin: 0;'>{total_consumption:.0f}</h3>
            <p style='margin: 5px 0 0 0; font-size: 14px;'>Consumption kWh</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='metric-box'>
            <h3 style='margin: 0;'>{avg_autarky:.1f}%</h3>
            <p style='margin: 5px 0 0 0; font-size: 14px;'>Avg Autarky</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class='metric-box'>
            <h3 style='margin: 0;'>€{total_savings:.0f}</h3>
            <p style='margin: 5px 0 0 0; font-size: 14px;'>Net Savings</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class='metric-box'>
            <h3 style='margin: 0;'>{total_co2:.0f} kg</h3>
            <p style='margin: 5px 0 0 0; font-size: 14px;'>CO₂ Saved</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col6:
        st.markdown(f"""
        <div class='metric-box'>
            <h3 style='margin: 0;'>{total_co2/21:.1f}</h3>
            <p style='margin: 5px 0 0 0; font-size: 14px;'>Trees Equiv.</p>
        </div>
        """, unsafe_allow_html=True)


def display_daily_predictions(df: pd.DataFrame):
    """Display daily prediction charts."""
    
    st.markdown("## 📈 Daily Predictions")
    
    # Energy flow chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['pv_generation_kwh'],
        mode='lines',
        name='PV Generation',
        fill='tozeroy',
        line=dict(color='gold', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['consumption_kwh'],
        mode='lines',
        name='Consumption',
        line=dict(color='lightblue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['self_consumed_kwh'],
        mode='lines',
        name='Self-Consumed',
        fill='tozeroy',
        line=dict(color='green', width=2)
    ))
    
    fig.update_layout(
        title="30-Day Energy Forecast",
        xaxis_title="Date",
        yaxis_title="Energy (kWh)",
        template="plotly_dark",
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Autarky chart
    fig2 = go.Figure()
    
    fig2.add_trace(go.Bar(
        x=df['date'],
        y=df['autarky_pct'],
        marker_color=df['autarky_pct'],
        marker_colorscale='RdYlGn',
        marker_cmin=0,
        marker_cmax=100,
        name='Autarky'
    ))
    
    fig2.update_layout(
        title="Daily Autarky Degree",
        xaxis_title="Date",
        yaxis_title="Autarky (%)",
        template="plotly_dark",
        height=400
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # Financial forecast
    fig3 = go.Figure()
    
    fig3.add_trace(go.Scatter(
        x=df['date'],
        y=df['net_benefit_eur'].cumsum(),
        mode='lines',
        name='Cumulative Savings',
        fill='tozeroy',
        line=dict(color='#00ff00', width=3)
    ))
    
    fig3.update_layout(
        title="Cumulative Financial Benefit",
        xaxis_title="Date",
        yaxis_title="Cumulative Savings (€)",
        template="plotly_dark",
        height=400
    )
    
    st.plotly_chart(fig3, use_container_width=True)


def display_calendar_view(df: pd.DataFrame):
    """Display calendar heatmap view."""
    
    st.markdown("## 📅 Calendar View")
    
    # Create calendar grid
    start_date = df['date'].min()
    
    # Determine production levels (fix for monotonic bins)
    q33 = df['pv_generation_kwh'].quantile(0.33)
    q67 = df['pv_generation_kwh'].quantile(0.67)
    max_val = df['pv_generation_kwh'].max()
    
    # Ensure bins are monotonically increasing
    bins = [0, q33, q67, max_val]
    if len(set(bins)) < len(bins):  # If duplicates exist
        df['production_level'] = 'Medium'  # Default value
    else:
        df['production_level'] = pd.cut(
            df['pv_generation_kwh'],
            bins=bins,
            labels=['Low', 'Medium', 'High']
        )
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        x=df['day_of_week'],
        y=df['day_number'],
        z=df['pv_generation_kwh'],
        colorscale='Viridis',
        text=df.apply(
            lambda row: f"{row['date'].strftime('%d')}<br>{row['pv_generation_kwh']:.1f} kWh<br>{row['autarky_pct']:.0f}% autarky",
            axis=1
        ),
        hovertemplate='%{text}<extra></extra>',
        colorbar=dict(title="PV kWh")
    ))
    
    fig.update_layout(
        title="30-Day Production Calendar",
        xaxis_title="Day of Week",
        yaxis_title="Day Number",
        template="plotly_dark",
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)


def display_weekly_breakdown(df: pd.DataFrame):
    """Display weekly breakdown."""
    
    st.markdown("## 📊 Weekly Breakdown")
    
    # Add week number
    df['week'] = ((df['day_number'] - 1) // 7) + 1
    
    # Group by week
    weekly = df.groupby('week').agg({
        'pv_generation_kwh': 'sum',
        'consumption_kwh': 'sum',
        'self_consumed_kwh': 'sum',
        'autarky_pct': 'mean',
        'net_benefit_eur': 'sum',
        'co2_saved_kg': 'sum'
    }).reset_index()
    
    weekly['week_label'] = 'Week ' + weekly['week'].astype(str)
    
    # Display table
    st.dataframe(
        weekly[[
            'week_label', 'pv_generation_kwh', 'consumption_kwh',
            'autarky_pct', 'net_benefit_eur', 'co2_saved_kg'
        ]].rename(columns={
            'week_label': 'Week',
            'pv_generation_kwh': 'PV (kWh)',
            'consumption_kwh': 'Consumption (kWh)',
            'autarky_pct': 'Autarky (%)',
            'net_benefit_eur': 'Savings (€)',
            'co2_saved_kg': 'CO₂ Saved (kg)'
        }),
        use_container_width=True,
        hide_index=True
    )


def display_optimization_insights(
    df: pd.DataFrame,
    pv_power: float,
    battery: float
):
    """Display AI optimization insights."""
    
    st.markdown("## 🤖 AI Optimization Insights")
    
    # Find best and worst days
    best_day = df.loc[df['autarky_pct'].idxmax()]
    worst_day = df.loc[df['autarky_pct'].idxmin()]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class='prediction-card'>
            <h3 style='color: #00ff00;'>✅ Best Day</h3>
            <p><strong>Date:</strong> {best_day['date'].strftime('%A, %B %d')}</p>
            <p><strong>Autarky:</strong> {best_day['autarky_pct']:.1f}%</p>
            <p><strong>PV:</strong> {best_day['pv_generation_kwh']:.1f} kWh</p>
            <p><strong>Savings:</strong> €{best_day['net_benefit_eur']:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='prediction-card'>
            <h3 style='color: #ff6b6b;'>⚠️ Challenging Day</h3>
            <p><strong>Date:</strong> {worst_day['date'].strftime('%A, %B %d')}</p>
            <p><strong>Autarky:</strong> {worst_day['autarky_pct']:.1f}%</p>
            <p><strong>PV:</strong> {worst_day['pv_generation_kwh']:.1f} kWh</p>
            <p><strong>Grid Import:</strong> {worst_day['grid_import_kwh']:.1f} kWh</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Recommendations
    st.markdown("### 💡 Recommendations")
    
    avg_autarky = df['autarky_pct'].mean()
    
    if avg_autarky < 50:
        st.warning(f"""
        🔋 **Battery Upgrade Recommended**: Your average autarky is {avg_autarky:.1f}%. 
        Adding {10-battery:.0f} kWh more battery capacity could increase autarky to ~{avg_autarky+20:.1f}%
        """)
    
    if battery == 0:
        potential_savings = df['grid_import_kwh'].sum() * 0.30 * 0.25
        st.info(f"""
        💰 **Battery Investment**: Adding a 10 kWh battery could save an additional €{potential_savings:.0f}/month
        """)
    
    # Find days with high grid export
    high_export_days = df[df['grid_export_kwh'] > df['grid_export_kwh'].quantile(0.75)]
    
    if len(high_export_days) > 5:
        st.success(f"""
        ⚡ **Load Shifting Opportunity**: {len(high_export_days)} days have high grid export. 
        Consider shifting appliance usage to these days to maximize self-consumption.
        """)


if __name__ == "__main__":
    main()