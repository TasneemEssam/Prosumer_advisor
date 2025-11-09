"""HTW Berlin-style Solar Calculator Suite - Streamlit App

Comprehensive solar calculators including:
1. Solarisator - Complete PV + Heat Pump + EV calculator
2. Independence Calculator - Autarky degree calculation
3. Solar Mobility Tool - EV charging with solar
4. Battery Storage Inspector - Storage system sizing
5. Plug-in Solar Simulator - Balcony solar systems

Run with: streamlit run solar_calculator_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple


# Page configuration
st.set_page_config(
    page_title="Solar Calculator Suite",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)


def main():
    """Main application."""
    
    # Header
    st.title("☀️ Solar Calculator Suite")
    st.markdown("**Professional solar energy calculators for prosumers**")
    st.markdown("---")
    
    # Calculator selection
    calculator = st.sidebar.selectbox(
        "Select Calculator",
        [
            "🏠 Solarisator (Complete System)",
            "📊 Independence Calculator",
            "🚗 Solar Mobility Tool",
            "🔋 Battery Storage Inspector",
            "🔌 Plug-in Solar Simulator"
        ]
    )
    
    if "Solarisator" in calculator:
        solarisator()
    elif "Independence" in calculator:
        independence_calculator()
    elif "Mobility" in calculator:
        solar_mobility_tool()
    elif "Battery" in calculator:
        battery_inspector()
    elif "Plug-in" in calculator:
        plugin_solar_simulator()


def solarisator():
    """Complete system calculator: PV + Heat Pump + EV."""
    
    st.header("🏠 Solarisator - Complete System Calculator")
    st.markdown("Calculate your energy independence with PV, heat pump, and electric vehicle")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("⚙️ System Configuration")
        
        # Location
        st.markdown("**📍 Location**")
        location = st.selectbox(
            "City",
            ["Berlin", "Munich", "Hamburg", "Cologne", "Frankfurt", "Custom"]
        )
        
        if location == "Custom":
            lat = st.number_input("Latitude", value=52.52, format="%.2f")
            lon = st.number_input("Longitude", value=13.40, format="%.2f")
        else:
            # Predefined coordinates
            coords = {
                "Berlin": (52.52, 13.40),
                "Munich": (48.14, 11.58),
                "Hamburg": (53.55, 10.00),
                "Cologne": (50.94, 6.96),
                "Frankfurt": (50.11, 8.68)
            }
            lat, lon = coords[location]
        
        # PV System
        st.markdown("**☀️ PV System**")
        pv_power = st.slider("PV Power (kWp)", 1.0, 30.0, 10.0, 0.5)
        roof_tilt = st.slider("Roof Tilt (°)", 0, 90, 30, 5)
        roof_azimuth = st.selectbox(
            "Roof Orientation",
            ["South (0°)", "SE/SW (±45°)", "East/West (±90°)"]
        )
        azimuth = {"South (0°)": 0, "SE/SW (±45°)": 45, "East/West (±90°)": 90}[roof_azimuth]
        
        # Battery
        st.markdown("**🔋 Battery Storage**")
        battery_enabled = st.checkbox("Include Battery", value=True)
        if battery_enabled:
            battery_capacity = st.slider("Battery Capacity (kWh)", 0.0, 30.0, 10.0, 1.0)
        else:
            battery_capacity = 0.0
        
        # Heat Pump
        st.markdown("**🌡️ Heat Pump**")
        heat_pump_enabled = st.checkbox("Include Heat Pump", value=True)
        if heat_pump_enabled:
            heating_demand = st.slider("Annual Heating Demand (kWh)", 5000, 30000, 15000, 1000)
            cop = st.slider("COP (Coefficient of Performance)", 2.0, 5.0, 3.5, 0.1)
        else:
            heating_demand = 0
            cop = 3.5
        
        # Electric Vehicle
        st.markdown("**🚗 Electric Vehicle**")
        ev_enabled = st.checkbox("Include Electric Vehicle", value=True)
        if ev_enabled:
            annual_km = st.slider("Annual Driving (km)", 5000, 30000, 15000, 1000)
            consumption = st.slider("Consumption (kWh/100km)", 10.0, 25.0, 18.0, 0.5)
        else:
            annual_km = 0
            consumption = 18.0
        
        # Household
        st.markdown("**🏡 Household**")
        household_consumption = st.slider("Annual Consumption (kWh)", 1000, 8000, 3500, 100)
        
        # Economics
        st.markdown("**💰 Economics**")
        electricity_price = st.number_input("Electricity Price (€/kWh)", 0.10, 0.50, 0.30, 0.01)
        feed_in_tariff = st.number_input("Feed-in Tariff (€/kWh)", 0.05, 0.20, 0.08, 0.01)
        
        calculate_btn = st.button("🚀 Calculate", type="primary")
    
    with col2:
        if calculate_btn:
            with st.spinner("Calculating your energy system..."):
                results = calculate_solarisator(
                    pv_power, battery_capacity, household_consumption,
                    heating_demand, cop, annual_km, consumption,
                    electricity_price, feed_in_tariff, lat, lon
                )
                
                display_solarisator_results(results)


def calculate_solarisator(
    pv_power, battery_capacity, household_consumption,
    heating_demand, cop, annual_km, ev_consumption,
    electricity_price, feed_in_tariff, lat, lon
) -> Dict[str, Any]:
    """Calculate complete system performance."""
    
    # Estimate annual PV generation (simplified)
    # Germany average: ~1000 kWh/kWp
    pv_generation = pv_power * 1000  # kWh/year
    
    # Calculate total consumption
    ev_consumption_kwh = (annual_km / 100) * ev_consumption
    heat_pump_consumption = heating_demand / cop if heating_demand > 0 else 0
    total_consumption = household_consumption + heat_pump_consumption + ev_consumption_kwh
    
    # Self-consumption calculation (simplified)
    # Without battery: ~30% self-consumption
    # With battery: up to 70% self-consumption
    if battery_capacity > 0:
        self_consumption_rate = min(0.70, 0.30 + (battery_capacity / pv_power) * 0.10)
    else:
        self_consumption_rate = 0.30
    
    self_consumed = min(pv_generation * self_consumption_rate, total_consumption)
    grid_import = max(0, total_consumption - self_consumed)
    grid_export = max(0, pv_generation - self_consumed)
    
    # Autarky degree
    autarky = (self_consumed / total_consumption * 100) if total_consumption > 0 else 0
    
    # Self-consumption degree
    self_consumption_degree = (self_consumed / pv_generation * 100) if pv_generation > 0 else 0
    
    # Economics
    savings = self_consumed * electricity_price
    feed_in_revenue = grid_export * feed_in_tariff
    grid_costs = grid_import * electricity_price
    annual_benefit = savings + feed_in_revenue - grid_costs
    
    # CO2 savings (Germany grid: ~400g CO2/kWh)
    co2_savings = self_consumed * 0.4  # kg CO2
    
    return {
        "pv_generation": pv_generation,
        "total_consumption": total_consumption,
        "household_consumption": household_consumption,
        "heat_pump_consumption": heat_pump_consumption,
        "ev_consumption": ev_consumption_kwh,
        "self_consumed": self_consumed,
        "grid_import": grid_import,
        "grid_export": grid_export,
        "autarky": autarky,
        "self_consumption_degree": self_consumption_degree,
        "annual_benefit": annual_benefit,
        "savings": savings,
        "feed_in_revenue": feed_in_revenue,
        "grid_costs": grid_costs,
        "co2_savings": co2_savings
    }


def display_solarisator_results(results: Dict[str, Any]):
    """Display comprehensive results."""
    
    st.subheader("📊 Results")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Autarky Degree",
            f"{results['autarky']:.1f}%",
            help="Percentage of consumption covered by own PV"
        )
    
    with col2:
        st.metric(
            "Self-Consumption",
            f"{results['self_consumption_degree']:.1f}%",
            help="Percentage of PV used directly"
        )
    
    with col3:
        st.metric(
            "Annual Benefit",
            f"€{results['annual_benefit']:.0f}",
            help="Savings + Feed-in - Grid costs"
        )
    
    with col4:
        st.metric(
            "CO₂ Savings",
            f"{results['co2_savings']:.0f} kg",
            help="Annual CO₂ emissions avoided"
        )
    
    # Energy flow diagram
    st.subheader("⚡ Energy Flow")
    
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=["PV Generation", "Self-Consumption", "Grid Export", 
                   "Household", "Heat Pump", "EV", "Grid Import"],
            color=["gold", "green", "orange", "blue", "red", "purple", "gray"]
        ),
        link=dict(
            source=[0, 0, 1, 1, 1, 6, 6, 6],
            target=[1, 2, 3, 4, 5, 3, 4, 5],
            value=[
                results['self_consumed'],
                results['grid_export'],
                results['household_consumption'] - min(results['household_consumption'], results['self_consumed']/3),
                results['heat_pump_consumption'] - min(results['heat_pump_consumption'], results['self_consumed']/3),
                results['ev_consumption'] - min(results['ev_consumption'], results['self_consumed']/3),
                min(results['household_consumption'], results['grid_import']/3),
                min(results['heat_pump_consumption'], results['grid_import']/3),
                min(results['ev_consumption'], results['grid_import']/3)
            ]
        )
    ))
    
    fig.update_layout(title="Annual Energy Flow (kWh)", height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔋 Energy Balance")
        
        balance_data = pd.DataFrame({
            "Category": ["PV Generation", "Total Consumption", "Self-Consumed", 
                        "Grid Import", "Grid Export"],
            "kWh/year": [
                results['pv_generation'],
                results['total_consumption'],
                results['self_consumed'],
                results['grid_import'],
                results['grid_export']
            ]
        })
        
        st.dataframe(balance_data, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("💰 Economic Analysis")
        
        economic_data = pd.DataFrame({
            "Item": ["Electricity Savings", "Feed-in Revenue", "Grid Costs", "Net Benefit"],
            "€/year": [
                results['savings'],
                results['feed_in_revenue'],
                -results['grid_costs'],
                results['annual_benefit']
            ]
        })
        
        st.dataframe(economic_data, use_container_width=True, hide_index=True)
    
    # Consumption breakdown
    st.subheader("📈 Consumption Breakdown")
    
    consumption_df = pd.DataFrame({
        "Category": ["Household", "Heat Pump", "Electric Vehicle"],
        "kWh/year": [
            results['household_consumption'],
            results['heat_pump_consumption'],
            results['ev_consumption']
        ]
    })
    
    fig = px.pie(
        consumption_df,
        values="kWh/year",
        names="Category",
        title="Annual Consumption Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)


def independence_calculator():
    """Calculate autarky degree based on PV and battery size."""
    
    st.header("📊 Independence Calculator")
    st.markdown("Calculate your energy independence (autarky degree)")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("⚙️ Configuration")
        
        annual_consumption = st.slider(
            "Annual Consumption (kWh)",
            1000, 10000, 4000, 100
        )
        
        pv_power = st.slider(
            "PV System Size (kWp)",
            1.0, 20.0, 5.0, 0.5
        )
        
        battery_capacity = st.slider(
            "Battery Capacity (kWh)",
            0.0, 20.0, 5.0, 0.5
        )
        
        calculate_btn = st.button("Calculate Independence", type="primary")
    
    with col2:
        if calculate_btn:
            # Calculate autarky for different scenarios
            pv_range = np.linspace(1, 20, 20)
            battery_range = np.linspace(0, 20, 21)
            
            autarky_matrix = np.zeros((len(battery_range), len(pv_range)))
            
            for i, batt in enumerate(battery_range):
                for j, pv in enumerate(pv_range):
                    # Simplified autarky calculation
                    pv_gen = pv * 1000
                    base_self_consumption = 0.30
                    battery_bonus = min(0.40, (batt / pv) * 0.10) if batt > 0 else 0
                    self_consumption_rate = base_self_consumption + battery_bonus
                    
                    self_consumed = min(pv_gen * self_consumption_rate, annual_consumption)
                    autarky_matrix[i, j] = (self_consumed / annual_consumption) * 100
            
            # Heatmap
            fig = go.Figure(data=go.Heatmap(
                z=autarky_matrix,
                x=pv_range,
                y=battery_range,
                colorscale='RdYlGn',
                colorbar=dict(title="Autarky %")
            ))
            
            fig.update_layout(
                title="Autarky Degree Matrix",
                xaxis_title="PV System Size (kWp)",
                yaxis_title="Battery Capacity (kWh)",
                height=500
            )
            
            # Add current configuration marker
            fig.add_trace(go.Scatter(
                x=[pv_power],
                y=[battery_capacity],
                mode='markers',
                marker=dict(size=15, color='blue', symbol='star'),
                name='Your Configuration'
            ))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Current autarky
            current_autarky = autarky_matrix[
                int(battery_capacity),
                int((pv_power - 1) * 20 / 19)
            ]
            
            st.success(f"🎯 Your Autarky Degree: **{current_autarky:.1f}%**")
            
            # Recommendations
            st.subheader("💡 Recommendations")
            
            if current_autarky < 30:
                st.warning("⚠️ Low independence. Consider increasing PV size or adding battery storage.")
            elif current_autarky < 60:
                st.info("ℹ️ Moderate independence. Adding battery storage could significantly improve autarky.")
            else:
                st.success("✅ Good independence! You're covering most of your consumption with solar.")


def solar_mobility_tool():
    """Calculate solar EV charging potential."""
    
    st.header("🚗 Solar Mobility Tool")
    st.markdown("Calculate how much you can charge your EV with solar power")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚗 Vehicle")
        annual_km = st.slider("Annual Driving (km)", 5000, 50000, 15000, 1000)
        consumption = st.slider("Consumption (kWh/100km)", 10.0, 30.0, 18.0, 0.5)
        battery_size = st.slider("Vehicle Battery (kWh)", 20, 100, 60, 5)
        
        st.subheader("☀️ Solar System")
        pv_power = st.slider("PV Power (kWp)", 1.0, 20.0, 8.0, 0.5)
        home_consumption = st.slider("Home Consumption (kWh/year)", 2000, 8000, 3500, 100)
    
    with col2:
        # Calculate
        ev_demand = (annual_km / 100) * consumption
        pv_generation = pv_power * 1000
        
        # Assume 40% of PV can be used for EV charging
        solar_ev_charging = min(ev_demand, pv_generation * 0.40)
        grid_charging = ev_demand - solar_ev_charging
        
        solar_percentage = (solar_ev_charging / ev_demand * 100) if ev_demand > 0 else 0
        
        # Display results
        st.metric("Annual EV Demand", f"{ev_demand:.0f} kWh")
        st.metric("Solar Charging", f"{solar_ev_charging:.0f} kWh ({solar_percentage:.1f}%)")
        st.metric("Grid Charging", f"{grid_charging:.0f} kWh")
        
        # Visualization
        fig = go.Figure(data=[
            go.Bar(name='Solar Charging', x=['EV Charging'], y=[solar_ev_charging], marker_color='gold'),
            go.Bar(name='Grid Charging', x=['EV Charging'], y=[grid_charging], marker_color='gray')
        ])
        
        fig.update_layout(
            title="EV Charging Sources",
            yaxis_title="kWh/year",
            barmode='stack',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Economics
        electricity_price = 0.30
        savings = solar_ev_charging * electricity_price
        
        st.success(f"💰 Annual Savings: €{savings:.0f}")
        st.info(f"🌍 CO₂ Avoided: {solar_ev_charging * 0.4:.0f} kg/year")


def battery_inspector():
    """Battery storage sizing tool."""
    
    st.header("🔋 Battery Storage Inspector")
    st.markdown("Find the right battery size for your system")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("System Parameters")
        
        pv_power = st.slider("PV Power (kWp)", 1.0, 20.0, 6.0, 0.5)
        annual_consumption = st.slider("Annual Consumption (kWh)", 2000, 8000, 4000, 100)
        
        st.subheader("Battery Options")
        
        battery_options = {
            "Small (5 kWh)": 5,
            "Medium (10 kWh)": 10,
            "Large (15 kWh)": 15,
            "Custom": 0
        }
        
        selected_size = st.selectbox("Battery Size", list(battery_options.keys()))
        
        if selected_size == "Custom":
            battery_capacity = st.slider("Custom Capacity (kWh)", 1.0, 30.0, 10.0, 0.5)
        else:
            battery_capacity = battery_options[selected_size]
    
    with col2:
        # Calculate performance metrics
        pv_generation = pv_power * 1000
        
        # Without battery
        autarky_no_battery = min(30, (pv_generation / annual_consumption) * 30)
        
        # With battery
        battery_bonus = min(40, (battery_capacity / pv_power) * 10)
        autarky_with_battery = min(80, autarky_no_battery + battery_bonus)
        
        improvement = autarky_with_battery - autarky_no_battery
        
        # Display comparison
        st.subheader("📊 Performance Comparison")
        
        comparison_df = pd.DataFrame({
            "Scenario": ["Without Battery", "With Battery", "Improvement"],
            "Autarky (%)": [autarky_no_battery, autarky_with_battery, improvement]
        })
        
        fig = px.bar(
            comparison_df,
            x="Scenario",
            y="Autarky (%)",
            title="Autarky Comparison",
            color="Scenario",
            color_discrete_sequence=['lightgray', 'green', 'blue']
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.subheader("💡 Recommendation")
        
        optimal_size = pv_power * 1.0  # Rule of thumb: 1 kWh per kWp
        
        if battery_capacity < optimal_size * 0.7:
            st.warning(f"⚠️ Battery might be undersized. Consider {optimal_size:.1f} kWh for optimal performance.")
        elif battery_capacity > optimal_size * 1.5:
            st.info(f"ℹ️ Battery might be oversized. {optimal_size:.1f} kWh could be more cost-effective.")
        else:
            st.success(f"✅ Good sizing! Battery capacity matches your PV system well.")


def plugin_solar_simulator():
    """Balcony/plug-in solar calculator."""
    
    st.header("🔌 Plug-in Solar Simulator")
    st.markdown("Calculate savings with balcony solar systems")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("System Configuration")
        
        num_modules = st.slider("Number of Modules", 1, 4, 2)
        module_power = st.slider("Module Power (Wp)", 300, 500, 400, 10)
        
        location_type = st.selectbox(
            "Installation Location",
            ["Balcony (South)", "Balcony (East/West)", "Roof", "Wall"]
        )
        
        # Location factor
        location_factors = {
            "Balcony (South)": 0.85,
            "Balcony (East/West)": 0.70,
            "Roof": 1.00,
            "Wall": 0.60
        }
        
        factor = location_factors[location_type]
        
        st.subheader("Economics")
        electricity_price = st.number_input("Electricity Price (€/kWh)", 0.20, 0.50, 0.30, 0.01)
        system_cost = st.number_input("System Cost (€)", 300, 2000, 800, 50)
    
    with col2:
        # Calculate generation
        total_power = (num_modules * module_power) / 1000  # kWp
        annual_generation = total_power * 1000 * factor  # kWh/year
        
        # Assume 70% self-consumption for plug-in systems
        self_consumed = annual_generation * 0.70
        
        # Economics
        annual_savings = self_consumed * electricity_price
        payback_years = system_cost / annual_savings if annual_savings > 0 else 0
        
        # Display results
        st.metric("System Power", f"{total_power:.2f} kWp")
        st.metric("Annual Generation", f"{annual_generation:.0f} kWh")
        st.metric("Self-Consumption", f"{self_consumed:.0f} kWh")
        st.metric("Annual Savings", f"€{annual_savings:.0f}")
        st.metric("Payback Period", f"{payback_years:.1f} years")
        
        # 25-year projection
        years = np.arange(0, 26)
        cumulative_savings = years * annual_savings
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=years,
            y=cumulative_savings,
            mode='lines',
            name='Cumulative Savings',
            line=dict(color='green', width=3)
        ))
        
        fig.add_hline(
            y=system_cost,
            line_dash="dash",
            line_color="red",
            annotation_text="System Cost"
        )
        
        fig.update_layout(
            title="25-Year Financial Projection",
            xaxis_title="Years",
            yaxis_title="Cumulative Savings (€)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # CO2 savings
        co2_savings = self_consumed * 0.4  # kg CO2/year
        
        st.success(f"🌍 Annual CO₂ Savings: {co2_savings:.0f} kg")


if __name__ == "__main__":
    main()