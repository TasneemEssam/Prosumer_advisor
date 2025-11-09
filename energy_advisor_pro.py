"""Energy Advisor Pro - All-in-One Comprehensive Energy Management System

Combines ALL features from 4 apps into one unified interface:
- BHT Berlin calculators (5 calculators)
- Weather forecasting & smart charging
- AI-powered insights & gamification
- 30-day predictions & planning

Developed for Berliner Hochschule für Technik (BHT)
Run with: streamlit run energy_advisor_pro.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
import requests
from dataclasses import dataclass

# Page config
st.set_page_config(
    page_title="Energy Advisor Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Professional & Modern
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(to bottom right, #0f2027, #203a43, #2c5364);
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3 {
        color: #00d4ff;
        font-weight: 700;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3); }
        50% { box-shadow: 0 12px 40px rgba(102, 126, 234, 0.5); }
    }
    
    .info-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        color: #000;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        font-weight: 600;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: #000;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        font-weight: 600;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 700;
        border-radius: 10px;
        padding: 12px 30px;
        border: none;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    .energy-score {
        font-size: 72px;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# HELPER CLASSES
# ============================================================================

class WeatherForecaster:
    """Fetch weather forecasts from Open-Meteo API."""
    
    BASE_URL = "https://api.open-meteo.com/v1/forecast"
    
    @staticmethod
    def fetch_forecast(lat: float, lon: float, days: int = 7) -> pd.DataFrame:
        """Fetch weather forecast."""
        try:
            params = {
                "latitude": lat,
                "longitude": lon,
                "hourly": ["temperature_2m", "cloudcover", "shortwave_radiation",
                          "direct_radiation", "diffuse_radiation", "windspeed_10m"],
                "timezone": "auto",
                "forecast_days": days
            }
            
            response = requests.get(WeatherForecaster.BASE_URL, params=params, timeout=10)
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
            
            return df.set_index("timestamp")
        except Exception as e:
            st.warning(f"⚠️ Could not fetch weather data: {e}")
            return pd.DataFrame()


class SolarCalculator:
    """Solar calculations with weather integration."""
    
    @staticmethod
    def calculate_pv_from_irradiance(ghi_wm2: float, pv_power_kw: float) -> float:
        """Calculate PV output from irradiance."""
        stc_irradiance = 1000
        performance_ratio = 0.85
        return max(0, (ghi_wm2 / stc_irradiance) * pv_power_kw * performance_ratio)
    
    @staticmethod
    def predict_daily_generation(weather_df: pd.DataFrame, pv_power_kw: float) -> pd.DataFrame:
        """Predict PV generation from weather."""
        if weather_df.empty:
            return pd.DataFrame()
        
        df = weather_df.copy()
        df["pv_kw"] = df["ghi_wm2"].apply(
            lambda x: SolarCalculator.calculate_pv_from_irradiance(x, pv_power_kw)
        )
        df["pv_kw"] = df["pv_kw"] * (1 - df["cloudcover_pct"] / 200)
        return df


@dataclass
class Achievement:
    """User achievement."""
    name: str
    icon: str
    description: str
    unlocked: bool
    progress: float


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application."""
    
    # Header
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 40px; border-radius: 20px; margin-bottom: 30px;
                box-shadow: 0 0 40px rgba(102, 126, 234, 0.5);'>
        <h1 style='color: white; text-align: center; margin: 0; font-size: 48px;'>
            ⚡ Energy Advisor Pro
        </h1>
        <p style='color: white; text-align: center; margin: 10px 0 0 0; font-size: 20px;'>
            All-in-One Comprehensive Energy Management System
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("### 🎯 Navigation")
        
        page = st.radio(
            "Select Feature",
            [
                "🏠 BHT Calculators",
                "🌤️ Weather & Smart Charging",
                "🤖 AI Energy Advisor",
                "📅 30-Day Predictions",
                "📊 Quick Analysis"
            ]
        )
    
    # Route to appropriate page
    if "BHT" in page:
        bht_calculators()
    elif "Weather" in page:
        weather_and_charging()
    elif "AI" in page:
        ai_energy_advisor()
    elif "30-Day" in page:
        monthly_predictions()
    elif "Quick" in page:
        quick_analysis()


# ============================================================================
# PAGE 1: HTW CALCULATORS
# ============================================================================

def bht_calculators():
    """BHT Berlin-style calculators."""
    
    st.header("🏠 BHT Berlin Solar Calculators")
    
    calculator = st.selectbox(
        "Select Calculator",
        [
            "Solarisator (Complete System)",
            "Independence Calculator",
            "Solar Mobility Tool",
            "Battery Inspector",
            "Plug-in Solar Simulator"
        ]
    )
    
    if "Solarisator" in calculator:
        solarisator()
    elif "Independence" in calculator:
        independence_calculator()
    elif "Mobility" in calculator:
        solar_mobility()
    elif "Battery" in calculator:
        battery_inspector()
    elif "Plug-in" in calculator:
        plugin_solar()


def solarisator():
    """Complete system calculator."""
    st.subheader("🏠 Solarisator - Complete System")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        pv_power = st.slider("PV Power (kWp)", 1.0, 30.0, 10.0, 0.5)
        battery = st.slider("Battery (kWh)", 0.0, 30.0, 10.0, 1.0)
        household = st.slider("Household (kWh/year)", 1000, 8000, 3500, 100)
        
        ev_enabled = st.checkbox("Electric Vehicle", value=True)
        if ev_enabled:
            annual_km = st.slider("Annual km", 5000, 30000, 15000, 1000)
            ev_consumption = (annual_km / 100) * 18.0
        else:
            ev_consumption = 0
        
        elec_price = st.number_input("Electricity (€/kWh)", 0.10, 0.50, 0.30, 0.01)
        feed_in = st.number_input("Feed-in (€/kWh)", 0.05, 0.20, 0.08, 0.01)
        
        calc_btn = st.button("🚀 Calculate", type="primary")
    
    with col2:
        if calc_btn:
            # Calculate
            pv_gen = pv_power * 1000
            total_consumption = household + ev_consumption
            
            if battery > 0:
                self_consumption_rate = min(0.70, 0.30 + (battery / pv_power) * 0.10)
            else:
                self_consumption_rate = 0.30
            
            self_consumed = min(pv_gen * self_consumption_rate, total_consumption)
            grid_import = max(0, total_consumption - self_consumed)
            grid_export = max(0, pv_gen - self_consumed)
            autarky = (self_consumed / total_consumption * 100) if total_consumption > 0 else 0
            
            savings = self_consumed * elec_price
            feed_in_revenue = grid_export * feed_in
            grid_costs = grid_import * elec_price
            annual_benefit = savings + feed_in_revenue - grid_costs
            co2_savings = self_consumed * 0.4
            
            # Display
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class='metric-card'>
                    <h3 style='margin: 0;'>{autarky:.1f}%</h3>
                    <p style='margin: 5px 0 0 0;'>Autarky</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class='metric-card'>
                    <h3 style='margin: 0;'>€{annual_benefit:.0f}</h3>
                    <p style='margin: 5px 0 0 0;'>Annual Benefit</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class='metric-card'>
                    <h3 style='margin: 0;'>{pv_gen:.0f} kWh</h3>
                    <p style='margin: 5px 0 0 0;'>PV Generation</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class='metric-card'>
                    <h3 style='margin: 0;'>{co2_savings:.0f} kg</h3>
                    <p style='margin: 5px 0 0 0;'>CO₂ Saved</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Sankey diagram
            st.subheader("⚡ Energy Flow")
            
            fig = go.Figure(go.Sankey(
                node=dict(
                    label=["PV", "Self-Use", "Grid Export", "Household", "EV", "Grid Import"],
                    color=["gold", "green", "orange", "blue", "purple", "gray"]
                ),
                link=dict(
                    source=[0, 0, 1, 1, 5, 5],
                    target=[1, 2, 3, 4, 3, 4],
                    value=[self_consumed, grid_export, household, ev_consumption,
                           max(0, household - self_consumed * 0.6),
                           max(0, ev_consumption - self_consumed * 0.4)]
                )
            ))
            
            fig.update_layout(height=400, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)


def independence_calculator():
    """Independence calculator."""
    st.subheader("📊 Independence Calculator")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        consumption = st.slider("Annual Consumption (kWh)", 1000, 10000, 4000, 100)
        pv_power = st.slider("PV Power (kWp)", 1.0, 20.0, 5.0, 0.5)
        battery = st.slider("Battery (kWh)", 0.0, 20.0, 5.0, 0.5)
        calc_btn = st.button("Calculate", type="primary")
    
    with col2:
        if calc_btn:
            # Create heatmap
            pv_range = np.linspace(1, 20, 20)
            battery_range = np.linspace(0, 20, 21)
            autarky_matrix = np.zeros((len(battery_range), len(pv_range)))
            
            for i, batt in enumerate(battery_range):
                for j, pv in enumerate(pv_range):
                    pv_gen = pv * 1000
                    rate = 0.30 + (min(0.40, (batt / pv) * 0.10) if batt > 0 else 0)
                    self_consumed = min(pv_gen * rate, consumption)
                    autarky_matrix[i, j] = (self_consumed / consumption) * 100
            
            fig = go.Figure(data=go.Heatmap(
                z=autarky_matrix,
                x=pv_range,
                y=battery_range,
                colorscale='RdYlGn',
                colorbar=dict(title="Autarky %")
            ))
            
            fig.update_layout(
                title="Autarky Degree Matrix",
                xaxis_title="PV Size (kWp)",
                yaxis_title="Battery (kWh)",
                height=500,
                template="plotly_dark"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            current_autarky = autarky_matrix[int(battery), int((pv_power - 1) * 20 / 19)]
            st.success(f"🎯 Your Autarky: **{current_autarky:.1f}%**")


def solar_mobility():
    """Solar mobility tool."""
    st.subheader("🚗 Solar Mobility Tool")
    
    col1, col2 = st.columns(2)
    
    with col1:
        annual_km = st.slider("Annual Driving (km)", 5000, 50000, 15000, 1000)
        consumption = st.slider("Consumption (kWh/100km)", 10.0, 30.0, 18.0, 0.5)
        pv_power = st.slider("PV Power (kWp)", 1.0, 20.0, 8.0, 0.5)
    
    with col2:
        ev_demand = (annual_km / 100) * consumption
        pv_generation = pv_power * 1000
        solar_ev = min(ev_demand, pv_generation * 0.40)
        grid_ev = ev_demand - solar_ev
        solar_pct = (solar_ev / ev_demand * 100) if ev_demand > 0 else 0
        
        st.metric("EV Demand", f"{ev_demand:.0f} kWh")
        st.metric("Solar Charging", f"{solar_ev:.0f} kWh ({solar_pct:.1f}%)")
        st.metric("Grid Charging", f"{grid_ev:.0f} kWh")
        
        savings = solar_ev * 0.30
        st.success(f"💰 Annual Savings: €{savings:.0f}")
        st.info(f"🌍 CO₂ Avoided: {solar_ev * 0.4:.0f} kg/year")


def battery_inspector():
    """Battery inspector."""
    st.subheader("🔋 Battery Storage Inspector")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pv_power = st.slider("PV Power (kWp)", 1.0, 20.0, 6.0, 0.5)
        consumption = st.slider("Consumption (kWh)", 2000, 8000, 4000, 100)
        battery = st.slider("Battery (kWh)", 0.0, 30.0, 10.0, 1.0)
    
    with col2:
        pv_gen = pv_power * 1000
        autarky_no_battery = min(30, (pv_gen / consumption) * 30)
        battery_bonus = min(40, (battery / pv_power) * 10) if battery > 0 else 0
        autarky_with_battery = min(80, autarky_no_battery + battery_bonus)
        improvement = autarky_with_battery - autarky_no_battery
        
        df = pd.DataFrame({
            "Scenario": ["Without Battery", "With Battery", "Improvement"],
            "Autarky (%)": [autarky_no_battery, autarky_with_battery, improvement]
        })
        
        fig = px.bar(df, x="Scenario", y="Autarky (%)", title="Autarky Comparison",
                     color="Scenario", color_discrete_sequence=['gray', 'green', 'blue'])
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        
        optimal = pv_power * 1.0
        if battery < optimal * 0.7:
            st.warning(f"⚠️ Consider {optimal:.1f} kWh for optimal performance")
        else:
            st.success("✅ Good sizing!")


def plugin_solar():
    """Plug-in solar simulator."""
    st.subheader("🔌 Plug-in Solar Simulator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_modules = st.slider("Modules", 1, 4, 2)
        module_power = st.slider("Module Power (Wp)", 300, 500, 400, 10)
        location = st.selectbox("Location", ["Balcony (South)", "Balcony (East/West)", "Roof", "Wall"])
        elec_price = st.number_input("Electricity (€/kWh)", 0.20, 0.50, 0.30, 0.01)
        system_cost = st.number_input("System Cost (€)", 300, 2000, 800, 50)
    
    with col2:
        factors = {"Balcony (South)": 0.85, "Balcony (East/West)": 0.70, "Roof": 1.00, "Wall": 0.60}
        total_power = (num_modules * module_power) / 1000
        annual_gen = total_power * 1000 * factors[location]
        self_consumed = annual_gen * 0.70
        annual_savings = self_consumed * elec_price
        payback = system_cost / annual_savings if annual_savings > 0 else 0
        
        st.metric("System Power", f"{total_power:.2f} kWp")
        st.metric("Annual Generation", f"{annual_gen:.0f} kWh")
        st.metric("Annual Savings", f"€{annual_savings:.0f}")
        st.metric("Payback Period", f"{payback:.1f} years")
        
        years = np.arange(0, 26)
        cumulative = years * annual_savings
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=years, y=cumulative, mode='lines', name='Savings',
                                fill='tozeroy', line=dict(color='green', width=3)))
        fig.add_hline(y=system_cost, line_dash="dash", line_color="red", annotation_text="Cost")
        fig.update_layout(title="25-Year Projection", template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# PAGE 2: WEATHER & SMART CHARGING
# ============================================================================

def weather_and_charging():
    """Weather forecasting and smart charging."""
    
    st.header("🌤️ Weather Forecasting & Smart Charging")
    
    tab1, tab2 = st.tabs(["📡 Weather Forecast", "🚗 Smart EV Charging"])
    
    with tab1:
        weather_forecast()
    
    with tab2:
        smart_ev_charging()


def weather_forecast():
    """Weather forecast viewer."""
    st.subheader("📡 Weather & Solar Forecast")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        location = st.selectbox("City", ["Berlin", "Munich", "Hamburg", "Cologne", "Frankfurt"])
        coords = {
            "Berlin": (52.52, 13.40), "Munich": (48.14, 11.58),
            "Hamburg": (53.55, 10.00), "Cologne": (50.94, 6.96), "Frankfurt": (50.11, 8.68)
        }
        lat, lon = coords[location]
        days = st.slider("Forecast Days", 1, 7, 3)
        fetch_btn = st.button("📡 Fetch Forecast", type="primary")
    
    with col2:
        if fetch_btn:
            with st.spinner("Fetching weather data..."):
                weather_df = WeatherForecaster.fetch_forecast(lat, lon, days)
                
                if not weather_df.empty:
                    # Temperature
                    fig1 = go.Figure()
                    fig1.add_trace(go.Scatter(x=weather_df.index, y=weather_df["temp_C"],
                                             mode='lines', name='Temperature', line=dict(color='red', width=2)))
                    fig1.update_layout(title="Temperature", template="plotly_dark", height=300)
                    st.plotly_chart(fig1, use_container_width=True)
                    
                    # Solar radiation
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=weather_df.index, y=weather_df["ghi_wm2"],
                                             mode='lines', fill='tozeroy', line=dict(color='gold', width=2)))
                    fig2.update_layout(title="Solar Irradiance", template="plotly_dark", height=300)
                    st.plotly_chart(fig2, use_container_width=True)


def smart_ev_charging():
    """Smart EV charging optimizer."""
    st.subheader("🚗 Smart EV Charging Optimizer")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        location = st.selectbox("Location", ["Berlin", "Munich", "Hamburg"])
        coords = {"Berlin": (52.52, 13.40), "Munich": (48.14, 11.58), "Hamburg": (53.55, 10.00)}
        lat, lon = coords[location]
        
        pv_power = st.slider("PV Power (kWp)", 1.0, 20.0, 8.0, 0.5)
        ev_battery = st.slider("EV Battery (kWh)", 20, 100, 60, 5)
        current_soc = st.slider("Current SOC (%)", 0, 100, 20, 5)
        charging_power = st.slider("Charger Power (kW)", 3.7, 22.0, 11.0, 0.1)
        
        optimize_btn = st.button("🔍 Find Optimal Time", type="primary")
    
    with col2:
        if optimize_btn:
            with st.spinner("Analyzing solar forecast..."):
                weather_df = WeatherForecaster.fetch_forecast(lat, lon, days=3)
                
                if not weather_df.empty:
                    pv_forecast = SolarCalculator.predict_daily_generation(weather_df, pv_power)
                    energy_needed = ev_battery * (100 - current_soc) / 100
                    hours_needed = energy_needed / charging_power
                    
                    # Find best window
                    pv_forecast["rolling_sum"] = pv_forecast["pv_kw"].rolling(
                        window=int(hours_needed), min_periods=1
                    ).sum()
                    
                    best_idx = pv_forecast["rolling_sum"].idxmax()
                    best_start = best_idx
                    best_end = best_idx + timedelta(hours=hours_needed)
                    
                    best_window = pv_forecast.loc[best_start:best_end]
                    solar_energy = best_window["pv_kw"].sum()
                    grid_energy = max(0, energy_needed - solar_energy)
                    solar_pct = (solar_energy / energy_needed * 100) if energy_needed > 0 else 0
                    savings = solar_energy * 0.30
                    
                    st.markdown(f"""
                    <div class='success-box'>
                        <h4>✅ Optimal Charging Window</h4>
                        <p><strong>Start:</strong> {best_start.strftime('%Y-%m-%d %H:%M')}</p>
                        <p><strong>End:</strong> {best_end.strftime('%Y-%m-%d %H:%M')}</p>
                        <p><strong>Solar Energy:</strong> {solar_energy:.1f} kWh ({solar_pct:.1f}%)</p>
                        <p><strong>Grid Energy:</strong> {grid_energy:.1f} kWh</p>
                        <p><strong>Savings:</strong> €{savings:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Visualization
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=pv_forecast.index, y=pv_forecast["pv_kw"],
                                           mode='lines', fill='tozeroy', line=dict(color='gold', width=2)))
                    fig.add_vrect(x0=best_start, x1=best_end, fillcolor="green", opacity=0.2,
                                 annotation_text="Optimal Window")
                    fig.update_layout(title="Solar Forecast with Optimal Window",
                                     template="plotly_dark", height=400)
                    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# PAGE 3: AI ENERGY ADVISOR
# ============================================================================

def ai_energy_advisor():
    """AI-powered energy advisor."""
    
    st.header("🤖 AI Energy Advisor")
    
    feature = st.selectbox(
        "Select AI Feature",
        ["Energy Health Dashboard", "Pattern Recognition", "Carbon Tracker",
         "Community Intelligence", "Achievements"]
    )
    
    if "Health" in feature:
        energy_health_dashboard()
    elif "Pattern" in feature:
        pattern_recognition()
    elif "Carbon" in feature:
        carbon_tracker()
    elif "Community" in feature:
        community_intelligence()
    elif "Achievements" in feature:
        achievements_system()


def energy_health_dashboard():
    """Energy health dashboard."""
    st.subheader("🏥 Energy Health Dashboard")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        pv_power = st.slider("PV Power (kWp)", 1.0, 30.0, 10.0, 0.5)
        battery = st.slider("Battery (kWh)", 0.0, 30.0, 10.0, 1.0)
        consumption = st.slider("Consumption (kWh/year)", 2000, 10000, 4000, 100)
        analyze_btn = st.button("🤖 AI Analysis", type="primary")
    
    with col2:
        if analyze_btn:
            pv_gen = pv_power * 1000
            rate = 0.30 + (battery / pv_power) * 0.10 if battery > 0 else 0.30
            self_consumed = min(pv_gen * rate, consumption)
            autarky = (self_consumed / consumption * 100) if consumption > 0 else 0
            co2 = self_consumed * 0.4
            cost = self_consumed * 0.30
            
            # Health score
            score = int(min(100, autarky * 0.35 + rate * 100 * 0.25 +
                           min(co2 / 1000 * 100, 100) * 0.20 +
                           min(cost / 1000 * 100, 100) * 0.20))
            
            if score >= 90:
                rating, color = "🏆 EXCELLENT", "#00ff00"
            elif score >= 75:
                rating, color = "⭐ GREAT", "#7fff00"
            elif score >= 60:
                rating, color = "👍 GOOD", "#ffff00"
            else:
                rating, color = "⚠️ FAIR", "#ffa500"
            
            st.markdown(f"""
            <div class='info-card'>
                <h2 style='text-align: center;'>Energy Health Score</h2>
                <div class='energy-score'>{score}</div>
                <h3 style='text-align: center; color: {color};'>{rating}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class='metric-card'>
                    <h3 style='margin: 0;'>{autarky:.1f}%</h3>
                    <p style='margin: 5px 0 0 0;'>Autarky</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class='metric-card'>
                    <h3 style='margin: 0;'>{co2:.0f} kg</h3>
                    <p style='margin: 5px 0 0 0;'>CO₂ Saved</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class='metric-card'>
                    <h3 style='margin: 0;'>€{cost:.0f}</h3>
                    <p style='margin: 5px 0 0 0;'>Savings</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class='metric-card'>
                    <h3 style='margin: 0;'>{rate*100:.1f}%</h3>
                    <p style='margin: 5px 0 0 0;'>Self-Use</p>
                </div>
                """, unsafe_allow_html=True)
            
            if score < 60:
                st.markdown("""
                <div class='warning-box'>
                    💡 <strong>AI Insight:</strong> Consider adding battery storage to increase autarky by 20-30%.
                </div>
                """, unsafe_allow_html=True)


def pattern_recognition():
    """Pattern recognition."""
    st.subheader("🧠 AI Pattern Recognition")
    st.info("AI analyzes your consumption patterns and provides recommendations")
    
    # Sample patterns
    patterns = [
        ("Evening Peak", 0.92, "High usage at 19:00", "Shift loads to midday"),
        ("Weekend Surge", 0.87, "20% higher on weekends", "Optimize weekend charging"),
        ("Seasonal Variation", 0.95, "Winter heating detected", "Consider heat pump optimization")
    ]
    
    for name, confidence, desc, rec in patterns:
        color = "#00ff00" if confidence > 0.9 else "#ffff00"
        st.markdown(f"""
        <div class='info-card'>
            <h3 style='color: #00d4ff;'>{name}</h3>
            <p><strong>Confidence:</strong> <span style='color: {color};'>{confidence*100:.0f}%</span></p>
            <p><strong>Description:</strong> {desc}</p>
            <div class='success-box'>
                💡 <strong>Recommendation:</strong> {rec}
            </div>
        </div>
        """, unsafe_allow_html=True)


def carbon_tracker():
    """Carbon footprint tracker."""
    st.subheader("🌍 Carbon Footprint Tracker")
    
    solar_energy = st.slider("Annual Solar Energy (kWh)", 1000, 15000, 8000, 100)
    
    co2_avoided = solar_energy * 0.4
    trees = co2_avoided / 21
    km_avoided = co2_avoided / 0.12
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <h3 style='margin: 0;'>{co2_avoided:.0f} kg</h3>
            <p style='margin: 5px 0 0 0;'>CO₂ Avoided</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <h3 style='margin: 0;'>{trees:.0f}</h3>
            <p style='margin: 5px 0 0 0;'>Trees Equivalent</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <h3 style='margin: 0;'>{km_avoided:.0f} km</h3>
            <p style='margin: 5px 0 0 0;'>Car Driving Avoided</p>
        </div>
        """, unsafe_allow_html=True)


def community_intelligence():
    """Community intelligence."""
    st.subheader("👥 Community Intelligence")
    
    user_autarky = st.slider("Your Autarky (%)", 0.0, 100.0, 65.0, 1.0)
    
    community_avg = 45.0
    percentile = 50 + (user_autarky - community_avg)
    rank = 'Top 25%' if percentile >= 75 else 'Above Average' if percentile >= 50 else 'Below Average'
    
    st.markdown(f"""
    <div class='info-card'>
        <h3 style='color: #00d4ff;'>Your Ranking</h3>
        <h2 style='color: #00ff00; font-size: 48px;'>{rank}</h2>
        <p>Among 1,247 similar households</p>
        <p>You're in the <strong>{percentile:.0f}th percentile</strong></p>
    </div>
    """, unsafe_allow_html=True)


def achievements_system():
    """Achievements system."""
    st.subheader("🏆 Achievements & Challenges")
    
    achievements = [
        Achievement("Solar Pioneer", "🌟", "Reach 50% autarky", True, 100),
        Achievement("Energy Master", "⚡", "Reach 75% autarky", True, 100),
        Achievement("Carbon Hero", "🌍", "Save 1000kg CO₂", True, 100),
        Achievement("Grid Independence", "🔋", "Reach 90% autarky", False, 72),
        Achievement("Community Leader", "👑", "Top 10%", False, 45),
    ]
    
    cols = st.columns(5)
    
    for i, ach in enumerate(achievements):
        with cols[i]:
            opacity = 1.0 if ach.unlocked else 0.3
            st.markdown(f"""
            <div style='text-align: center; opacity: {opacity};'>
                <div style='font-size: 60px;'>{ach.icon}</div>
                <p style='color: white; font-weight: 600;'>{ach.name}</p>
                <p style='color: #aaa; font-size: 12px;'>{ach.description}</p>
                <p style='color: #00d4ff;'>{ach.progress:.0f}%</p>
            </div>
            """, unsafe_allow_html=True)


# ============================================================================
# PAGE 4: 30-DAY PREDICTIONS
# ============================================================================

def monthly_predictions():
    """30-day predictions."""
    
    st.header("📅 30-Day Energy Predictions")
    
    with st.sidebar:
        st.markdown("### Configuration")
        location = st.selectbox("City", ["Berlin", "Munich", "Hamburg"])
        pv_power = st.slider("PV (kWp)", 1.0, 30.0, 10.0, 0.5)
        battery = st.slider("Battery (kWh)", 0.0, 30.0, 10.0, 1.0)
        consumption = st.slider("Consumption (kWh/year)", 1000, 8000, 3500, 100)
        predict_btn = st.button("🔮 Generate Forecast", type="primary")
    
    if predict_btn:
        with st.spinner("Generating 30-day forecast..."):
            # Generate data
            dates = pd.date_range(start=datetime.now(), periods=30, freq='D')
            month = datetime.now().month
            seasonal_factor = {1: 0.3, 2: 0.5, 3: 0.7, 4: 0.9, 5: 1.0, 6: 1.0,
                             7: 1.0, 8: 0.9, 9: 0.8, 10: 0.6, 11: 0.4, 12: 0.3}[month]
            
            pv_gen = []
            for i in range(30):
                base = pv_power * 1000 / 365 * seasonal_factor
                weather = np.random.uniform(0.7, 1.3)
                if i > 0:
                    weather = 0.7 * pv_gen[-1] / base + 0.3 * weather
                pv_gen.append(base * weather)
            
            cons = [consumption / 365 * (1.2 if d.dayofweek >= 5 else 1.0) *
                   np.random.uniform(0.85, 1.15) for d in dates]
            
            df = pd.DataFrame({
                'date': dates,
                'pv_kwh': pv_gen,
                'consumption_kwh': cons
            })
            
            rate = 0.30 + (battery / pv_power) * 0.10 if battery > 0 else 0.30
            df['self_consumed'] = np.minimum(df['pv_kwh'] * rate, df['consumption_kwh'])
            df['autarky'] = (df['self_consumed'] / df['consumption_kwh'] * 100).fillna(0)
            df['savings'] = df['self_consumed'] * 0.30
            
            # Summary
            st.markdown("## 📊 Monthly Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class='metric-card'>
                    <h3 style='margin: 0;'>{df['pv_kwh'].sum():.0f}</h3>
                    <p style='margin: 5px 0 0 0;'>PV kWh</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class='metric-card'>
                    <h3 style='margin: 0;'>{df['autarky'].mean():.1f}%</h3>
                    <p style='margin: 5px 0 0 0;'>Avg Autarky</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class='metric-card'>
                    <h3 style='margin: 0;'>€{df['savings'].sum():.0f}</h3>
                    <p style='margin: 5px 0 0 0;'>Savings</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class='metric-card'>
                    <h3 style='margin: 0;'>{df['self_consumed'].sum() * 0.4:.0f} kg</h3>
                    <p style='margin: 5px 0 0 0;'>CO₂ Saved</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Charts
            st.markdown("## 📈 Daily Predictions")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['date'], y=df['pv_kwh'], mode='lines',
                                    name='PV', fill='tozeroy', line=dict(color='gold', width=2)))
            fig.add_trace(go.Scatter(x=df['date'], y=df['consumption_kwh'], mode='lines',
                                    name='Consumption', line=dict(color='lightblue', width=2)))
            fig.update_layout(title="30-Day Energy Forecast", template="plotly_dark", height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Autarky chart
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=df['date'], y=df['autarky'], marker_color=df['autarky'],
                                 marker_colorscale='RdYlGn', marker_cmin=0, marker_cmax=100))
            fig2.update_layout(title="Daily Autarky", template="plotly_dark", height=400)
            st.plotly_chart(fig2, use_container_width=True)


# ============================================================================
# PAGE 5: QUICK ANALYSIS
# ============================================================================

def quick_analysis():
    """Quick analysis tool."""
    
    st.header("📊 Quick Energy Analysis")
    st.markdown("Get instant insights about your energy system")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pv_power = st.number_input("PV Power (kWp)", 1.0, 30.0, 10.0, 0.5)
        battery = st.number_input("Battery (kWh)", 0.0, 30.0, 10.0, 1.0)
    
    with col2:
        consumption = st.number_input("Annual Consumption (kWh)", 1000, 10000, 4000, 100)
        elec_price = st.number_input("Electricity Price (€/kWh)", 0.10, 0.50, 0.30, 0.01)
    
    with col3:
        st.markdown("### Quick Results")
        
        pv_gen = pv_power * 1000
        rate = 0.30 + (battery / pv_power) * 0.10 if battery > 0 else 0.30
        self_consumed = min(pv_gen * rate, consumption)
        autarky = (self_consumed / consumption * 100) if consumption > 0 else 0
        savings = self_consumed * elec_price
        
        st.metric("Autarky", f"{autarky:.1f}%")
        st.metric("Annual Savings", f"€{savings:.0f}")
        st.metric("CO₂ Saved", f"{self_consumed * 0.4:.0f} kg")
    
    # Recommendations
    st.markdown("### 💡 Quick Recommendations")
    
    if autarky < 50:
        st.warning("⚠️ Low autarky. Consider adding battery storage or increasing PV size.")
    elif autarky < 70:
        st.info("ℹ️ Good autarky. Battery storage could improve it further.")
    else:
        st.success("✅ Excellent autarky! Your system is well-optimized.")
    
    if battery == 0:
        potential_savings = (pv_gen * 0.40 - self_consumed) * elec_price
        st.info(f"💰 Adding a 10 kWh battery could save an additional €{potential_savings:.0f}/year")


# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()