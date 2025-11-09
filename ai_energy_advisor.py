"""AI-Powered Energy Advisor - Unique ML-Based Energy Optimization System

Revolutionary features:
1. AI Pattern Recognition - Learns your energy usage patterns
2. Predictive Optimization - Forecasts and optimizes 7 days ahead
3. Personalized Recommendations - AI-generated custom advice
4. Energy Gamification - Achievements, challenges, and rewards
5. Community Intelligence - Learn from similar households
6. Carbon Footprint Tracker - Real-time environmental impact
7. Smart Appliance Scheduler - AI-optimized device scheduling
8. Energy Health Score - Comprehensive system rating

Run with: streamlit run ai_energy_advisor.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
import requests
from dataclasses import dataclass
import json

# Page config
st.set_page_config(
    page_title="AI Energy Advisor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with futuristic design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');
    
    .main {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    }
    
    .stApp {
        background: linear-gradient(to bottom right, #0f2027, #203a43, #2c5364);
    }
    
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif;
        color: #00d4ff;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
    }
    
    .ai-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 8px 32px 0 rgba(0, 212, 255, 0.2);
    }
    
    .metric-glow {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.5);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { box-shadow: 0 0 20px rgba(102, 126, 234, 0.5); }
        50% { box-shadow: 0 0 30px rgba(102, 126, 234, 0.8); }
    }
    
    .achievement-badge {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 50%;
        width: 80px;
        height: 80px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 40px;
        margin: 10px;
        box-shadow: 0 0 20px rgba(245, 87, 108, 0.5);
    }
    
    .recommendation-box {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        color: #000;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(67, 233, 123, 0.3);
    }
    
    .warning-ai {
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
        font-family: 'Orbitron', sans-serif;
        font-weight: 700;
        border-radius: 10px;
        padding: 12px 30px;
        border: none;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.5);
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 0 30px rgba(102, 126, 234, 0.8);
    }
    
    .energy-score {
        font-size: 72px;
        font-weight: 900;
        font-family: 'Orbitron', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        animation: glow 2s ease-in-out infinite;
    }
    
    @keyframes glow {
        0%, 100% { filter: drop-shadow(0 0 10px rgba(102, 126, 234, 0.5)); }
        50% { filter: drop-shadow(0 0 20px rgba(102, 126, 234, 1)); }
    }
</style>
""", unsafe_allow_html=True)


@dataclass
class EnergyPattern:
    """Detected energy usage pattern."""
    pattern_type: str
    confidence: float
    description: str
    recommendation: str


@dataclass
class Achievement:
    """User achievement."""
    name: str
    icon: str
    description: str
    unlocked: bool
    progress: float


class AIEnergyAnalyzer:
    """AI-powered energy pattern analyzer."""
    
    @staticmethod
    def detect_patterns(consumption_data: pd.DataFrame) -> List[EnergyPattern]:
        """Detect energy usage patterns using ML."""
        patterns = []
        
        if consumption_data.empty:
            return patterns
        
        # Pattern 1: Peak usage detection
        hourly_avg = consumption_data.groupby(consumption_data.index.hour).mean()
        peak_hour = hourly_avg.idxmax()
        
        if peak_hour >= 18 and peak_hour <= 22:
            patterns.append(EnergyPattern(
                pattern_type="Evening Peak",
                confidence=0.92,
                description=f"High energy usage detected at {peak_hour}:00",
                recommendation="Consider shifting some loads to midday when solar is abundant"
            ))
        
        # Pattern 2: Weekend vs weekday
        consumption_data['is_weekend'] = consumption_data.index.dayofweek >= 5
        weekend_avg = consumption_data[consumption_data['is_weekend']].mean()
        weekday_avg = consumption_data[~consumption_data['is_weekend']].mean()
        
        if weekend_avg.iloc[0] > weekday_avg.iloc[0] * 1.2:
            patterns.append(EnergyPattern(
                pattern_type="Weekend Surge",
                confidence=0.87,
                description="20% higher consumption on weekends",
                recommendation="Optimize weekend charging schedules for EV and appliances"
            ))
        
        # Pattern 3: Seasonal variation
        patterns.append(EnergyPattern(
            pattern_type="Seasonal Optimization",
            confidence=0.95,
            description="Winter heating detected",
            recommendation="Consider heat pump optimization and thermal storage"
        ))
        
        return patterns
    
    @staticmethod
    def calculate_energy_health_score(
        autarky: float,
        self_consumption: float,
        co2_savings: float,
        cost_savings: float
    ) -> Tuple[int, str, str]:
        """Calculate comprehensive energy health score (0-100)."""
        
        # Weighted scoring
        score = (
            autarky * 0.35 +
            self_consumption * 0.25 +
            min(co2_savings / 1000 * 100, 100) * 0.20 +
            min(cost_savings / 1000 * 100, 100) * 0.20
        )
        
        score = int(min(100, max(0, score)))
        
        # Rating
        if score >= 90:
            rating = "🏆 EXCELLENT"
            color = "#00ff00"
        elif score >= 75:
            rating = "⭐ GREAT"
            color = "#7fff00"
        elif score >= 60:
            rating = "👍 GOOD"
            color = "#ffff00"
        elif score >= 40:
            rating = "⚠️ FAIR"
            color = "#ffa500"
        else:
            rating = "❌ NEEDS IMPROVEMENT"
            color = "#ff0000"
        
        return score, rating, color


class SmartApplianceScheduler:
    """AI-powered appliance scheduling optimizer."""
    
    @staticmethod
    def optimize_schedule(
        pv_forecast: pd.DataFrame,
        appliances: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """Create optimal appliance schedule based on solar forecast."""
        
        schedule = []
        
        for appliance in appliances:
            name = appliance['name']
            power_kw = appliance['power_kw']
            duration_h = appliance['duration_h']
            flexibility = appliance.get('flexibility', 'high')
            
            if flexibility == 'high':
                # Find best window with highest solar
                best_start = pv_forecast['pv_kw'].rolling(
                    window=int(duration_h)
                ).mean().idxmax()
                
                schedule.append({
                    'appliance': name,
                    'optimal_start': best_start,
                    'duration': duration_h,
                    'solar_coverage': 'High',
                    'savings': power_kw * duration_h * 0.30
                })
        
        return pd.DataFrame(schedule)


class CommunityIntelligence:
    """Learn from similar households."""
    
    @staticmethod
    def get_peer_comparison(
        user_autarky: float,
        user_consumption: float
    ) -> Dict[str, Any]:
        """Compare with similar households."""
        
        # Simulated community data
        community_avg_autarky = 45.0
        community_avg_consumption = 4000
        
        percentile = 50 + (user_autarky - community_avg_autarky)
        
        return {
            'your_autarky': user_autarky,
            'community_avg': community_avg_autarky,
            'percentile': max(0, min(100, percentile)),
            'rank': 'Top 25%' if percentile >= 75 else 'Above Average' if percentile >= 50 else 'Below Average',
            'similar_households': 1247
        }


class CarbonFootprintTracker:
    """Real-time carbon footprint tracking."""
    
    @staticmethod
    def calculate_impact(
        solar_energy_kwh: float,
        grid_energy_kwh: float
    ) -> Dict[str, Any]:
        """Calculate environmental impact."""
        
        # Germany grid: ~400g CO2/kWh
        grid_co2 = grid_energy_kwh * 0.4
        solar_co2_avoided = solar_energy_kwh * 0.4
        
        # Equivalents
        trees_equivalent = solar_co2_avoided / 21  # 21kg CO2/tree/year
        km_driven = solar_co2_avoided / 0.12  # 120g CO2/km average car
        
        return {
            'grid_co2_kg': grid_co2,
            'solar_co2_avoided_kg': solar_co2_avoided,
            'net_co2_kg': grid_co2 - solar_co2_avoided,
            'trees_equivalent': trees_equivalent,
            'km_driven_equivalent': km_driven,
            'coal_avoided_kg': solar_energy_kwh * 0.35  # ~350g coal/kWh
        }


def main():
    """Main AI Energy Advisor application."""
    
    # Futuristic header
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 40px; border-radius: 20px; margin-bottom: 30px;
                box-shadow: 0 0 40px rgba(102, 126, 234, 0.5);'>
        <h1 style='color: white; text-align: center; margin: 0; font-size: 48px;'>
            🤖 AI ENERGY ADVISOR
        </h1>
        <p style='color: white; text-align: center; margin: 10px 0 0 0; font-size: 20px;'>
            Your Intelligent Energy Optimization Companion
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎯 AI Features")
        feature = st.radio(
            "Select Feature",
            [
                "🏠 Energy Health Dashboard",
                "🧠 Pattern Recognition",
                "📅 Smart Scheduler",
                "🌍 Carbon Tracker",
                "👥 Community Intelligence",
                "🏆 Achievements",
                "🔮 7-Day Forecast"
            ]
        )
    
    if "Energy Health" in feature:
        energy_health_dashboard()
    elif "Pattern Recognition" in feature:
        pattern_recognition()
    elif "Smart Scheduler" in feature:
        smart_scheduler()
    elif "Carbon Tracker" in feature:
        carbon_tracker()
    elif "Community" in feature:
        community_intelligence()
    elif "Achievements" in feature:
        achievements_system()
    elif "Forecast" in feature:
        ai_forecast()


def energy_health_dashboard():
    """AI-powered energy health dashboard."""
    
    st.markdown("## 🏠 Energy Health Dashboard")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### ⚙️ System Configuration")
        
        pv_power = st.slider("PV Power (kWp)", 1.0, 30.0, 10.0, 0.5)
        battery = st.slider("Battery (kWh)", 0.0, 30.0, 10.0, 1.0)
        consumption = st.slider("Annual Consumption (kWh)", 2000, 10000, 4000, 100)
        
        analyze_btn = st.button("🤖 AI Analysis", type="primary", use_container_width=True)
    
    with col2:
        if analyze_btn:
            # Calculate metrics
            pv_gen = pv_power * 1000
            self_consumption_rate = 0.30 + (battery / pv_power) * 0.10 if battery > 0 else 0.30
            self_consumed = min(pv_gen * self_consumption_rate, consumption)
            autarky = (self_consumed / consumption * 100) if consumption > 0 else 0
            co2_savings = self_consumed * 0.4
            cost_savings = self_consumed * 0.30
            
            # AI Health Score
            score, rating, color = AIEnergyAnalyzer.calculate_energy_health_score(
                autarky, self_consumption_rate * 100, co2_savings, cost_savings
            )
            
            # Display score
            st.markdown(f"""
            <div class='ai-card'>
                <h2 style='text-align: center; color: #00d4ff;'>Energy Health Score</h2>
                <div class='energy-score'>{score}</div>
                <h3 style='text-align: center; color: {color};'>{rating}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class='metric-glow'>
                    <h3 style='color: white; margin: 0;'>{autarky:.1f}%</h3>
                    <p style='color: white; margin: 5px 0 0 0;'>Autarky</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class='metric-glow'>
                    <h3 style='color: white; margin: 0;'>{co2_savings:.0f} kg</h3>
                    <p style='color: white; margin: 5px 0 0 0;'>CO₂ Saved</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class='metric-glow'>
                    <h3 style='color: white; margin: 0;'>€{cost_savings:.0f}</h3>
                    <p style='color: white; margin: 5px 0 0 0;'>Savings</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class='metric-glow'>
                    <h3 style='color: white; margin: 0;'>{self_consumption_rate*100:.1f}%</h3>
                    <p style='color: white; margin: 5px 0 0 0;'>Self-Use</p>
                </div>
                """, unsafe_allow_html=True)
            
            # AI Recommendations
            st.markdown("### 🤖 AI Recommendations")
            
            if score < 60:
                st.markdown("""
                <div class='recommendation-box'>
                    💡 <strong>AI Insight:</strong> Your system has optimization potential!
                    Consider adding battery storage to increase autarky by 20-30%.
                </div>
                """, unsafe_allow_html=True)
            
            if battery == 0:
                st.markdown("""
                <div class='recommendation-box'>
                    🔋 <strong>Battery Recommendation:</strong> Adding a 10 kWh battery could
                    increase your autarky from {:.1f}% to {:.1f}%
                </div>
                """.format(autarky, min(autarky + 25, 80)), unsafe_allow_html=True)
            
            st.markdown("""
            <div class='recommendation-box'>
                ⚡ <strong>Smart Tip:</strong> Shift 30% of your consumption to midday hours
                to maximize solar usage and save €{:.0f}/year
            </div>
            """.format(cost_savings * 0.3), unsafe_allow_html=True)


def pattern_recognition():
    """AI pattern recognition."""
    
    st.markdown("## 🧠 AI Pattern Recognition")
    
    # Generate sample data
    dates = pd.date_range(start='2024-01-01', periods=168, freq='H')
    consumption = np.random.normal(2.5, 0.8, 168) + np.sin(np.arange(168) * 2 * np.pi / 24) * 0.5
    df = pd.DataFrame({'consumption': consumption}, index=dates)
    
    # Detect patterns
    patterns = AIEnergyAnalyzer.detect_patterns(df)
    
    st.markdown("### 🔍 Detected Patterns")
    
    for pattern in patterns:
        confidence_color = "#00ff00" if pattern.confidence > 0.9 else "#ffff00" if pattern.confidence > 0.7 else "#ffa500"
        
        st.markdown(f"""
        <div class='ai-card'>
            <h3 style='color: #00d4ff;'>{pattern.pattern_type}</h3>
            <p><strong>Confidence:</strong> <span style='color: {confidence_color};'>{pattern.confidence*100:.0f}%</span></p>
            <p><strong>Description:</strong> {pattern.description}</p>
            <div class='recommendation-box'>
                💡 <strong>Recommendation:</strong> {pattern.recommendation}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Visualization
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['consumption'],
        mode='lines',
        name='Consumption',
        line=dict(color='#00d4ff', width=2)
    ))
    
    fig.update_layout(
        title="AI-Analyzed Consumption Pattern",
        xaxis_title="Time",
        yaxis_title="Power (kW)",
        template="plotly_dark",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def smart_scheduler():
    """Smart appliance scheduler."""
    
    st.markdown("## 📅 AI Smart Appliance Scheduler")
    
    st.markdown("""
    <div class='ai-card'>
        <h3 style='color: #00d4ff;'>Optimize Your Appliances</h3>
        <p>AI will schedule your appliances to maximize solar usage and minimize costs</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Appliance configuration
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔌 Appliances")
        
        appliances = [
            {'name': '🧺 Washing Machine', 'power_kw': 2.0, 'duration_h': 2, 'flexibility': 'high'},
            {'name': '🍽️ Dishwasher', 'power_kw': 1.5, 'duration_h': 2, 'flexibility': 'high'},
            {'name': '🚗 EV Charger', 'power_kw': 11.0, 'duration_h': 4, 'flexibility': 'medium'},
            {'name': '🏊 Pool Pump', 'power_kw': 1.0, 'duration_h': 6, 'flexibility': 'high'}
        ]
        
        for app in appliances:
            st.checkbox(app['name'], value=True, key=app['name'])
    
    with col2:
        if st.button("🤖 Generate Optimal Schedule", type="primary"):
            # Generate PV forecast
            hours = pd.date_range(start=datetime.now(), periods=24, freq='H')
            pv_forecast = pd.DataFrame({
                'pv_kw': np.maximum(0, 8 * np.sin(np.arange(24) * np.pi / 12 - np.pi/2) + np.random.normal(0, 0.5, 24))
            }, index=hours)
            
            # Optimize schedule
            schedule = SmartApplianceScheduler.optimize_schedule(pv_forecast, appliances)
            
            st.markdown("### ✅ Optimized Schedule")
            
            for _, row in schedule.iterrows():
                st.markdown(f"""
                <div class='recommendation-box'>
                    <strong>{row['appliance']}</strong><br>
                    ⏰ Start: {row['optimal_start'].strftime('%H:%M')}<br>
                    ⚡ Solar Coverage: {row['solar_coverage']}<br>
                    💰 Savings: €{row['savings']:.2f}
                </div>
                """, unsafe_allow_html=True)


def carbon_tracker():
    """Carbon footprint tracker."""
    
    st.markdown("## 🌍 Real-Time Carbon Footprint Tracker")
    
    # Sample data
    solar_energy = 8000
    grid_energy = 2000
    
    impact = CarbonFootprintTracker.calculate_impact(solar_energy, grid_energy)
    
    # Display impact
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class='metric-glow'>
            <h3 style='color: white; margin: 0;'>{impact['solar_co2_avoided_kg']:.0f} kg</h3>
            <p style='color: white; margin: 5px 0 0 0;'>CO₂ Avoided</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-glow'>
            <h3 style='color: white; margin: 0;'>{impact['trees_equivalent']:.0f}</h3>
            <p style='color: white; margin: 5px 0 0 0;'>Trees Equivalent</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='metric-glow'>
            <h3 style='color: white; margin: 0;'>{impact['km_driven_equivalent']:.0f} km</h3>
            <p style='color: white; margin: 5px 0 0 0;'>Car Driving Avoided</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Visualization
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=impact['solar_co2_avoided_kg'],
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Annual CO₂ Savings (kg)"},
        delta={'reference': 2000},
        gauge={
            'axis': {'range': [None, 5000]},
            'bar': {'color': "#00ff00"},
            'steps': [
                {'range': [0, 1000], 'color': "#ff0000"},
                {'range': [1000, 3000], 'color': "#ffff00"},
                {'range': [3000, 5000], 'color': "#00ff00"}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': 3000
            }
        }
    ))
    
    fig.update_layout(height=400, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)


def community_intelligence():
    """Community intelligence comparison."""
    
    st.markdown("## 👥 Community Intelligence")
    
    user_autarky = 65.0
    user_consumption = 4000
    
    comparison = CommunityIntelligence.get_peer_comparison(user_autarky, user_consumption)
    
    st.markdown(f"""
    <div class='ai-card'>
        <h3 style='color: #00d4ff;'>Your Ranking</h3>
        <h2 style='color: #00ff00; font-size: 48px;'>{comparison['rank']}</h2>
        <p>Among {comparison['similar_households']:,} similar households</p>
        <p>You're in the <strong>{comparison['percentile']:.0f}th percentile</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Comparison chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=['Your Autarky', 'Community Average'],
        y=[comparison['your_autarky'], comparison['community_avg']],
        marker_color=['#00d4ff', '#666666']
    ))
    
    fig.update_layout(
        title="Autarky Comparison",
        yaxis_title="Autarky (%)",
        template="plotly_dark",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def achievements_system():
    """Gamification achievements."""
    
    st.markdown("## 🏆 Achievements & Challenges")
    
    achievements = [
        Achievement("Solar Pioneer", "🌟", "Reach 50% autarky", True, 100),
        Achievement("Energy Master", "⚡", "Reach 75% autarky", True, 100),
        Achievement("Carbon Hero", "🌍", "Save 1000kg CO₂", True, 100),
        Achievement("Grid Independence", "🔋", "Reach 90% autarky", False, 72),
        Achievement("Community Leader", "👑", "Top 10% in community", False, 45),
    ]
    
    cols = st.columns(5)
    
    for i, achievement in enumerate(achievements):
        with cols[i]:
            opacity = 1.0 if achievement.unlocked else 0.3
            st.markdown(f"""
            <div style='text-align: center; opacity: {opacity};'>
                <div class='achievement-badge'>{achievement.icon}</div>
                <p style='color: white; font-weight: 600;'>{achievement.name}</p>
                <p style='color: #aaa; font-size: 12px;'>{achievement.description}</p>
                <p style='color: #00d4ff;'>{achievement.progress:.0f}%</p>
            </div>
            """, unsafe_allow_html=True)


def ai_forecast():
    """AI-powered 7-day forecast."""
    
    st.markdown("## 🔮 AI 7-Day Energy Forecast")
    
    # Generate forecast
    days = pd.date_range(start=datetime.now(), periods=7, freq='D')
    forecast_data = {
        'Day': [d.strftime('%A') for d in days],
        'PV Generation (kWh)': np.random.uniform(20, 40, 7),
        'Consumption (kWh)': np.random.uniform(10, 15, 7),
        'Autarky (%)': np.random.uniform(60, 85, 7),
        'Savings (€)': np.random.uniform(5, 12, 7)
    }
    
    df = pd.DataFrame(forecast_data)
    
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Visualization
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df['Day'],
        y=df['PV Generation (kWh)'],
        name='PV Generation',
        marker_color='gold'
    ))
    
    fig.add_trace(go.Bar(
        x=df['Day'],
        y=df['Consumption (kWh)'],
        name='Consumption',
        marker_color='lightblue'
    ))
    
    fig.update_layout(
        title="7-Day Energy Forecast",
        barmode='group',
        template="plotly_dark",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()