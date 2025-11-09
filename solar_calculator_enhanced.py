"""Enhanced HTW Berlin-style Solar Calculator with Weather Forecasting & Smart Recommendations

Features:
- Beautiful UI with custom CSS
- Real-time weather forecasting (Open-Meteo API)
- Smart charging recommendations
- Solar production predictions
- Advanced visualizations
- Export capabilities

Run with: streamlit run solar_calculator_enhanced.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, Optional
import requests

# Page configuration
st.set_page_config(
    page_title="Solar Calculator Pro",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(to bottom, #f8f9fa 0%, #e9ecef 100%);
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .success-box {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .warning-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .info-box {
        background: #d1ecf1;
        border-left: 4px solid #17a2b8;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    h1 {
        color: #2c3e50;
        font-weight: 700;
    }
    h2 {
        color: #34495e;
        font-weight: 600;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 10px 24px;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)


class WeatherForecaster:
    """Fetch weather and solar radiation forecasts from Open-Meteo API."""
    
    BASE_URL = "https://api.open-meteo.com/v1/forecast"
    
    @staticmethod
    def fetch_forecast(lat: float, lon: float, days: int = 7) -> pd.DataFrame:
        """Fetch weather forecast including solar radiation.
        
        Args:
            lat: Latitude
            lon: Longitude
            days: Number of days to forecast
            
        Returns:
            DataFrame with hourly weather data
        """
        try:
            params = {
                "latitude": lat,
                "longitude": lon,
                "hourly": [
                    "temperature_2m",
                    "cloudcover",
                    "shortwave_radiation",
                    "direct_radiation",
                    "diffuse_radiation",
                    "windspeed_10m"
                ],
                "timezone": "auto",
                "forecast_days": days
            }
            
            response = requests.get(
                WeatherForecaster.BASE_URL,
                params=params,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            hourly = data["hourly"]
            df = pd.DataFrame({
                "timestamp": pd.to_datetime(hourly["time"]),
                "temp_C": hourly["temperature_2m"],
                "cloudcover_pct": hourly["cloudcover"],
                "ghi_wm2": hourly["shortwave_radiation"],
                "dni_wm2": hourly["direct_radiation"],
                "dhi_wm2": hourly["diffuse_radiation"],
                "windspeed_ms": hourly["windspeed_10m"]
            })
            
            df = df.set_index("timestamp")
            return df
            
        except Exception as e:
            st.warning(f"⚠️ Could not fetch weather data: {e}")
            return pd.DataFrame()


class SolarCalculator:
    """Enhanced solar calculations with weather integration."""
    
    @staticmethod
    def calculate_pv_from_irradiance(
        ghi_wm2: float,
        pv_power_kw: float,
        efficiency: float = 0.18,
        performance_ratio: float = 0.85
    ) -> float:
        """Calculate PV output from irradiance.
        
        Args:
            ghi_wm2: Global horizontal irradiance (W/m²)
            pv_power_kw: PV system peak power (kW)
            efficiency: Panel efficiency (default 18%)
            performance_ratio: System performance ratio (default 85%)
            
        Returns:
            PV output in kW
        """
        # Standard test conditions: 1000 W/m²
        stc_irradiance = 1000
        pv_output = (ghi_wm2 / stc_irradiance) * pv_power_kw * performance_ratio
        return max(0, pv_output)
    
    @staticmethod
    def predict_daily_generation(
        weather_df: pd.DataFrame,
        pv_power_kw: float
    ) -> pd.DataFrame:
        """Predict hourly PV generation from weather forecast.
        
        Args:
            weather_df: Weather forecast DataFrame
            pv_power_kw: PV system size
            
        Returns:
            DataFrame with predicted PV generation
        """
        if weather_df.empty:
            return pd.DataFrame()
        
        df = weather_df.copy()
        df["pv_kw"] = df["ghi_wm2"].apply(
            lambda x: SolarCalculator.calculate_pv_from_irradiance(x, pv_power_kw)
        )
        
        # Add cloud cover impact
        df["pv_kw"] = df["pv_kw"] * (1 - df["cloudcover_pct"] / 200)
        
        return df


class SmartCharging:
    """Smart charging recommendations based on solar forecast."""
    
    @staticmethod
    def find_optimal_charging_windows(
        pv_forecast: pd.DataFrame,
        ev_battery_kwh: float,
        charging_power_kw: float = 11.0
    ) -> Dict[str, Any]:
        """Find optimal EV charging windows.
        
        Args:
            pv_forecast: PV generation forecast
            ev_battery_kwh: EV battery capacity
            charging_power_kw: Charger power
            
        Returns:
            Dict with charging recommendations
        """
        if pv_forecast.empty:
            return {}
        
        # Calculate hours needed to charge
        hours_needed = ev_battery_kwh / charging_power_kw
        
        # Find windows with highest solar production
        df = pv_forecast.copy()
        df["rolling_sum"] = df["pv_kw"].rolling(
            window=int(hours_needed),
            min_periods=1
        ).sum()
        
        # Find best window
        best_idx = df["rolling_sum"].idxmax()
        best_window_start = best_idx
        best_window_end = best_idx + timedelta(hours=hours_needed)
        
        # Calculate solar vs grid charging
        best_window_df = df.loc[best_window_start:best_window_end]
        solar_energy = best_window_df["pv_kw"].sum()
        grid_energy = max(0, ev_battery_kwh - solar_energy)
        
        return {
            "optimal_start": best_window_start,
            "optimal_end": best_window_end,
            "solar_energy_kwh": solar_energy,
            "grid_energy_kwh": grid_energy,
            "solar_percentage": (solar_energy / ev_battery_kwh * 100) if ev_battery_kwh > 0 else 0,
            "estimated_cost_savings": solar_energy * 0.30  # Assuming 0.30 €/kWh
        }


def main():
    """Main application."""
    
    # Header with gradient
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 30px; border-radius: 10px; margin-bottom: 30px;'>
        <h1 style='color: white; text-align: center; margin: 0;'>
            ☀️ Solar Calculator Pro
        </h1>
        <p style='color: white; text-align: center; margin: 10px 0 0 0; font-size: 18px;'>
            Advanced solar energy calculator with weather forecasting & smart recommendations
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Calculator selection
    calculator = st.sidebar.selectbox(
        "🎯 Select Calculator",
        [
            "🏠 Complete System (Solarisator)",
            "📊 Independence Calculator",
            "🚗 Smart EV Charging",
            "🔋 Battery Inspector",
            "🔌 Plug-in Solar",
            "🌤️ Weather Forecast"
        ]
    )
    
    if "Complete System" in calculator:
        complete_system_calculator()
    elif "Independence" in calculator:
        independence_calculator()
    elif "Smart EV" in calculator:
        smart_ev_charging()
    elif "Battery" in calculator:
        battery_inspector()
    elif "Plug-in" in calculator:
        plugin_solar()
    elif "Weather" in calculator:
        weather_forecast_viewer()


def complete_system_calculator():
    """Enhanced complete system calculator with weather integration."""
    
    st.header("🏠 Complete System Calculator")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("⚙️ Configuration")
        
        # Location
        with st.expander("📍 Location", expanded=True):
            location = st.selectbox(
                "City",
                ["Berlin", "Munich", "Hamburg", "Cologne", "Frankfurt", "Custom"]
            )
            
            if location == "Custom":
                lat = st.number_input("Latitude", value=52.52, format="%.2f")
                lon = st.number_input("Longitude", value=13.40, format="%.2f")
            else:
                coords = {
                    "Berlin": (52.52, 13.40),
                    "Munich": (48.14, 11.58),
                    "Hamburg": (53.55, 10.00),
                    "Cologne": (50.94, 6.96),
                    "Frankfurt": (50.11, 8.68)
                }
                lat, lon = coords[location]
        
        # PV System
        with st.expander("☀️ PV System", expanded=True):
            pv_power = st.slider("PV Power (kWp)", 1.0, 30.0, 10.0, 0.5)
            battery_capacity = st.slider("Battery (kWh)", 0.0, 30.0, 10.0, 1.0)
        
        # Consumption
        with st.expander("🏡 Consumption", expanded=True):
            household = st.slider("Household (kWh/year)", 1000, 8000, 3500, 100)
            ev_enabled = st.checkbox("Electric Vehicle", value=True)
            if ev_enabled:
                annual_km = st.slider("Annual km", 5000, 30000, 15000, 1000)
                ev_consumption = (annual_km / 100) * 18.0
            else:
                ev_consumption = 0
        
        # Economics
        with st.expander("💰 Economics", expanded=False):
            elec_price = st.number_input("Electricity (€/kWh)", 0.10, 0.50, 0.30, 0.01)
            feed_in = st.number_input("Feed-in (€/kWh)", 0.05, 0.20, 0.08, 0.01)
        
        use_weather = st.checkbox("🌤️ Use Weather Forecast", value=True)
        calculate_btn = st.button("🚀 Calculate", type="primary", use_container_width=True)
    
    with col2:
        if calculate_btn:
            with st.spinner("🔄 Calculating your energy system..."):
                # Fetch weather if enabled
                weather_df = pd.DataFrame()
                if use_weather:
                    weather_df = WeatherForecaster.fetch_forecast(lat, lon, days=7)
                
                # Calculate results
                results = calculate_complete_system(
                    pv_power, battery_capacity, household, ev_consumption,
                    elec_price, feed_in, weather_df
                )
                
                display_complete_results(results, weather_df, pv_power)


def calculate_complete_system(
    pv_power, battery_capacity, household, ev_consumption,
    elec_price, feed_in, weather_df
):
    """Calculate complete system performance."""
    
    # Annual PV generation
    if not weather_df.empty:
        # Use weather forecast for next 7 days, extrapolate to year
        daily_gen = weather_df["ghi_wm2"].mean() * 24 * pv_power * 0.18 * 0.85 / 1000
        pv_generation = daily_gen * 365
    else:
        pv_generation = pv_power * 1000  # Simplified
    
    total_consumption = household + ev_consumption
    
    # Self-consumption with battery
    if battery_capacity > 0:
        self_consumption_rate = min(0.70, 0.30 + (battery_capacity / pv_power) * 0.10)
    else:
        self_consumption_rate = 0.30
    
    self_consumed = min(pv_generation * self_consumption_rate, total_consumption)
    grid_import = max(0, total_consumption - self_consumed)
    grid_export = max(0, pv_generation - self_consumed)
    
    autarky = (self_consumed / total_consumption * 100) if total_consumption > 0 else 0
    
    # Economics
    savings = self_consumed * elec_price
    feed_in_revenue = grid_export * feed_in
    grid_costs = grid_import * elec_price
    annual_benefit = savings + feed_in_revenue - grid_costs
    
    # CO2
    co2_savings = self_consumed * 0.4
    
    return {
        "pv_generation": pv_generation,
        "total_consumption": total_consumption,
        "household": household,
        "ev": ev_consumption,
        "self_consumed": self_consumed,
        "grid_import": grid_import,
        "grid_export": grid_export,
        "autarky": autarky,
        "annual_benefit": annual_benefit,
        "co2_savings": co2_savings
    }


def display_complete_results(results, weather_df, pv_power):
    """Display enhanced results with weather integration."""
    
    # Key metrics with custom styling
    st.markdown("### 📊 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <h3 style='color: #28a745; margin: 0;'>{results['autarky']:.1f}%</h3>
            <p style='color: #6c757d; margin: 5px 0 0 0;'>Autarky Degree</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <h3 style='color: #17a2b8; margin: 0;'>€{results['annual_benefit']:.0f}</h3>
            <p style='color: #6c757d; margin: 5px 0 0 0;'>Annual Benefit</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <h3 style='color: #ffc107; margin: 0;'>{results['pv_generation']:.0f} kWh</h3>
            <p style='color: #6c757d; margin: 5px 0 0 0;'>PV Generation</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class='metric-card'>
            <h3 style='color: #28a745; margin: 0;'>{results['co2_savings']:.0f} kg</h3>
            <p style='color: #6c757d; margin: 5px 0 0 0;'>CO₂ Saved</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Weather-based PV forecast
    if not weather_df.empty:
        st.markdown("### 🌤️ 7-Day Solar Production Forecast")
        
        pv_forecast = SolarCalculator.predict_daily_generation(weather_df, pv_power)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=pv_forecast.index,
            y=pv_forecast["pv_kw"],
            mode='lines',
            name='PV Output',
            fill='tozeroy',
            line=dict(color='gold', width=3)
        ))
        
        fig.update_layout(
            title="Predicted PV Output (Next 7 Days)",
            xaxis_title="Time",
            yaxis_title="Power (kW)",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Daily summary
        daily_summary = pv_forecast.resample('D')['pv_kw'].sum()
        
        st.markdown("### 📅 Daily Generation Forecast")
        
        fig2 = px.bar(
            x=daily_summary.index,
            y=daily_summary.values,
            labels={'x': 'Date', 'y': 'Energy (kWh)'},
            title="Daily Energy Production"
        )
        fig2.update_traces(marker_color='gold')
        
        st.plotly_chart(fig2, use_container_width=True)
    
    # Energy flow Sankey
    st.markdown("### ⚡ Energy Flow Diagram")
    
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            label=["PV", "Self-Use", "Grid Export", "Household", "EV", "Grid Import"],
            color=["gold", "green", "orange", "blue", "purple", "gray"]
        ),
        link=dict(
            source=[0, 0, 1, 1, 5, 5],
            target=[1, 2, 3, 4, 3, 4],
            value=[
                results['self_consumed'],
                results['grid_export'],
                results['household'],
                results['ev'],
                max(0, results['household'] - results['self_consumed'] * 0.6),
                max(0, results['ev'] - results['self_consumed'] * 0.4)
            ]
        )
    ))
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


def smart_ev_charging():
    """Smart EV charging with solar forecast."""
    
    st.header("🚗 Smart EV Charging Optimizer")
    st.markdown("Find the best time to charge your EV with solar power")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("⚙️ Configuration")
        
        # Location
        lat = st.number_input("Latitude", value=52.52, format="%.2f")
        lon = st.number_input("Longitude", value=13.40, format="%.2f")
        
        # PV System
        pv_power = st.slider("PV Power (kWp)", 1.0, 20.0, 8.0, 0.5)
        
        # EV
        ev_battery = st.slider("EV Battery (kWh)", 20, 100, 60, 5)
        charging_power = st.slider("Charger Power (kW)", 3.7, 22.0, 11.0, 0.1)
        current_soc = st.slider("Current SOC (%)", 0, 100, 20, 5)
        
        calculate_btn = st.button("🔍 Find Optimal Charging Time", type="primary")
    
    with col2:
        if calculate_btn:
            with st.spinner("🔄 Analyzing solar forecast..."):
                # Fetch weather
                weather_df = WeatherForecaster.fetch_forecast(lat, lon, days=3)
                
                if not weather_df.empty:
                    # Predict PV
                    pv_forecast = SolarCalculator.predict_daily_generation(weather_df, pv_power)
                    
                    # Calculate energy needed
                    energy_needed = ev_battery * (100 - current_soc) / 100
                    
                    # Find optimal window
                    recommendations = SmartCharging.find_optimal_charging_windows(
                        pv_forecast, energy_needed, charging_power
                    )
                    
                    # Display recommendations
                    st.markdown("### 🎯 Charging Recommendations")
                    
                    if recommendations:
                        st.markdown(f"""
                        <div class='success-box'>
                            <h4>✅ Optimal Charging Window</h4>
                            <p><strong>Start:</strong> {recommendations['optimal_start'].strftime('%Y-%m-%d %H:%M')}</p>
                            <p><strong>End:</strong> {recommendations['optimal_end'].strftime('%Y-%m-%d %H:%M')}</p>
                            <p><strong>Solar Energy:</strong> {recommendations['solar_energy_kwh']:.1f} kWh ({recommendations['solar_percentage']:.1f}%)</p>
                            <p><strong>Grid Energy:</strong> {recommendations['grid_energy_kwh']:.1f} kWh</p>
                            <p><strong>Estimated Savings:</strong> €{recommendations['estimated_cost_savings']:.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Visualization
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=pv_forecast.index,
                        y=pv_forecast["pv_kw"],
                        mode='lines',
                        name='PV Output',
                        fill='tozeroy',
                        line=dict(color='gold', width=2)
                    ))
                    
                    # Highlight optimal window
                    if recommendations:
                        fig.add_vrect(
                            x0=recommendations['optimal_start'],
                            x1=recommendations['optimal_end'],
                            fillcolor="green",
                            opacity=0.2,
                            line_width=0,
                            annotation_text="Optimal Window",
                            annotation_position="top left"
                        )
                    
                    fig.update_layout(
                        title="Solar Production Forecast with Optimal Charging Window",
                        xaxis_title="Time",
                        yaxis_title="PV Power (kW)",
                        height=500,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Tips
                    st.markdown("""
                    <div class='info-box'>
                        <h4>💡 Smart Charging Tips</h4>
                        <ul>
                            <li>Charge during peak sun hours (10 AM - 3 PM)</li>
                            <li>Use smart charger with solar integration</li>
                            <li>Consider battery storage for overnight charging</li>
                            <li>Monitor weather forecasts for cloudy days</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)


def weather_forecast_viewer():
    """Weather forecast visualization."""
    
    st.header("🌤️ Weather & Solar Forecast")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        lat = st.number_input("Latitude", value=52.52, format="%.2f")
        lon = st.number_input("Longitude", value=13.40, format="%.2f")
        days = st.slider("Forecast Days", 1, 7, 3)
        
        fetch_btn = st.button("📡 Fetch Forecast", type="primary")
    
    with col2:
        if fetch_btn:
            with st.spinner("🔄 Fetching weather data..."):
                weather_df = WeatherForecaster.fetch_forecast(lat, lon, days)
                
                if not weather_df.empty:
                    # Temperature
                    fig1 = go.Figure()
                    fig1.add_trace(go.Scatter(
                        x=weather_df.index,
                        y=weather_df["temp_C"],
                        mode='lines',
                        name='Temperature',
                        line=dict(color='red', width=2)
                    ))
                    fig1.update_layout(
                        title="Temperature Forecast",
                        xaxis_title="Time",
                        yaxis_title="Temperature (°C)",
                        height=300
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                    
                    # Solar radiation
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(
                        x=weather_df.index,
                        y=weather_df["ghi_wm2"],
                        mode='lines',
                        name='GHI',
                        fill='tozeroy',
                        line=dict(color='gold', width=2)
                    ))
                    fig2.update_layout(
                        title="Solar Irradiance Forecast",
                        xaxis_title="Time",
                        yaxis_title="Irradiance (W/m²)",
                        height=300
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    # Cloud cover
                    fig3 = go.Figure()
                    fig3.add_trace(go.Scatter(
                        x=weather_df.index,
                        y=weather_df["cloudcover_pct"],
                        mode='lines',
                        name='Cloud Cover',
                        fill='tozeroy',
                        line=dict(color='gray', width=2)
                    ))
                    fig3.update_layout(
                        title="Cloud Cover Forecast",
                        xaxis_title="Time",
                        yaxis_title="Cloud Cover (%)",
                        height=300
                    )
                    st.plotly_chart(fig3, use_container_width=True)


def independence_calculator():
    """Independence calculator (from original)."""
    st.header("📊 Independence Calculator")
    st.info("Calculate your energy independence based on PV and battery size")
    # Implementation from original solar_calculator_app.py


def battery_inspector():
    """Battery inspector (from original)."""
    st.header("🔋 Battery Storage Inspector")
    st.info("Find the right battery size for your system")
    # Implementation from original solar_calculator_app.py


def plugin_solar():
    """Plug-in solar calculator (from original)."""
    st.header("🔌 Plug-in Solar Simulator")
    st.info("Calculate savings with balcony solar systems")
    # Implementation from original solar_calculator_app.py


if __name__ == "__main__":
    main()