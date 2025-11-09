"""Visualization module for prosumer energy advisor.

This module provides functions to create various plots for analyzing
energy flows, action recommendations, and system performance.
"""

from typing import Dict, Any
import pandas as pd
import matplotlib.pyplot as plt


def plot_overview(df: pd.DataFrame, cfg: Dict[str, Any]) -> None:
    """Create overview plot of PV generation vs load with action shading.
    
    Args:
        df: DataFrame with columns: pv_kW, load_kW, recommended_action
        cfg: Configuration dictionary with visualization settings
    """
    tz_name = cfg.get('location', {}).get('tz', 'Europe/Berlin')
    # Plot PV and Load, shade background by recommended action category
    fig, ax = plt.subplots(figsize=(10,5))
    if 'pv_kW' in df:
        ax.plot(df.index, df['pv_kW'], label='PV generation (kW)', color='orange')
    if 'load_kW' in df:
        ax.plot(df.index, df['load_kW'], label='Load (kW)', color='red')
    # Shade by action
    if 'recommended_action' in df:
        action_colors = {
            'use_pv_direct': '#b3de69',   # light green
            'sell_to_grid': '#fdbf6f',    # light orange/yellow
            'charge_battery': '#a6cee3',  # light blue
            'charge_ev': '#fb9a99',       # light pink
            'idle': '#dddddd'            # light gray
        }
        # For each contiguous block of same action, shade it
        last_action = None
        block_start = None
        for time, action in df['recommended_action'].items():
            if action != last_action:
                # end previous block
                if last_action is not None:
                    ax.axvspan(block_start, time, color=action_colors.get(last_action, '#eeeeee'), alpha=0.3)
                # start new block
                block_start = time
                last_action = action
        # shade last block till end
        if last_action is not None:
            ax.axvspan(block_start, df.index.max(), color=action_colors.get(last_action, '#eeeeee'), alpha=0.3)
        # Create a custom legend for actions
        handles = [plt.Rectangle((0,0),1,1, color=color, alpha=0.5) for color in action_colors.values()]
        labels = list(action_colors.keys())
        ax.legend(handles, labels, title="Action (shading)", loc='upper right')
    else:
        ax.legend(loc='upper left')
    ax.set_ylabel("Power (kW)")
    ax.set_xlabel("Time")
    ax.set_title("PV Generation vs Load (with Action Highlights)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig(f"{cfg.get('visualization', {}).get('plot_dir', '.')}/pv_load_actions.png")
    plt.close(fig)

def plot_action_distribution(df: pd.DataFrame, cfg: Dict[str, Any]) -> None:
    """Plot bar chart of action frequency distribution.
    
    Args:
        df: DataFrame with 'recommended_action' column
        cfg: Configuration dictionary with visualization settings
    """
    if 'recommended_action' not in df:
        return
    action_counts = df['recommended_action'].value_counts()
    fig, ax = plt.subplots()
    action_counts.plot(kind='bar', ax=ax, color='skyblue')
    ax.set_title("Recommended Action Frequency")
    ax.set_ylabel("Hours")
    plt.tight_layout()
    fig.savefig(f"{cfg.get('visualization', {}).get('plot_dir', '.')}/action_frequency.png")
    plt.close(fig)

def plot_energy_flow(df: pd.DataFrame, cfg: Dict[str, Any]) -> None:
    """Plot daily energy flow analysis.
    
    Creates two stacked bar charts:
    1. PV usage: self-consumed vs exported
    2. Grid import: for load vs for charging
    
    Args:
        df: DataFrame with columns: pv_kW, load_kW, recommended_action
        cfg: Configuration dictionary with oracle and visualization settings
    """
    # Validate required columns
    required_cols = ['pv_kW', 'load_kW', 'recommended_action']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"[plot_energy_flow] Skipping: missing columns {missing_cols}")
        return
    # Initialize energy flow columns
    df_hourly = df.copy()
    df_hourly['pv_used_on_site'] = 0.0
    df_hourly['pv_exported'] = 0.0
    df_hourly['grid_import_for_load'] = 0.0
    df_hourly['grid_import_for_storage'] = 0.0
    
    # Battery and EV parameters
    oracle_cfg = cfg.get('oracle', {})
    cap_batt = oracle_cfg.get('battery_capacity_kwh', 10.0)
    cap_ev = oracle_cfg.get('ev_capacity_kwh', 50.0)
    charge_rate = 5.0  # Max charge rate in kW
    
    # Track state of charge
    soc_batt = 0.0
    soc_ev = 0.0
    # Calculate energy flows for each hour
    for i, row in df_hourly.iterrows():
        pv = row['pv_kW']
        load = row['load_kW']
        action = row['recommended_action']
        
        # PV allocation: first serve load
        used = min(pv, load)
        pv_surplus = pv - used
        
        # Grid import for load (if PV < load)
        grid_for_load = max(load - pv, 0.0)
        
        # Export logic
        export = 0.0
        if action == 'sell_to_grid':
            export = pv_surplus
            pv_surplus = 0.0
        
        # Charging logic
        grid_for_storage = 0.0
        
        if action == 'charge_battery':
            # Determine how much to charge
            available_capacity = cap_batt - soc_batt
            to_store = min(charge_rate, available_capacity)
            
            if pv_surplus >= to_store:
                # Use PV surplus for charging
                pv_surplus -= to_store
                soc_batt = min(cap_batt, soc_batt + to_store)
            else:
                # Use all PV surplus + grid import
                needed_from_grid = to_store - pv_surplus
                pv_surplus = 0.0
                soc_batt = min(cap_batt, soc_batt + to_store)
                grid_for_storage = needed_from_grid
                
        elif action == 'charge_ev':
            # EV charging (typically from grid at night)
            available_capacity = cap_ev - soc_ev
            to_store = min(charge_rate, available_capacity)
            soc_ev = min(cap_ev, soc_ev + to_store)
            grid_for_storage = to_store
        
        # Update hourly data
        df_hourly.at[i, 'pv_used_on_site'] = used
        df_hourly.at[i, 'pv_exported'] = export
        df_hourly.at[i, 'grid_import_for_load'] = grid_for_load
        df_hourly.at[i, 'grid_import_for_storage'] = grid_for_storage
    # Aggregate to daily totals
    daily = df_hourly.resample('D').sum(numeric_only=True)
    
    if daily.empty:
        print("[plot_energy_flow] No daily data to plot")
        return
    
    plot_dir = cfg.get('visualization', {}).get('plot_dir', 'outputs')
    
    # Plot 1: PV usage (self-consumed vs exported)
    fig, ax = plt.subplots(figsize=(10, 5))
    daily[['pv_used_on_site', 'pv_exported']].plot(
        kind='bar',
        stacked=True,
        ax=ax,
        color=['green', 'gold'],
        width=0.8
    )
    ax.set_ylabel("Energy (kWh)")
    ax.set_xlabel("Date")
    ax.set_title("Daily PV Usage: Self-Consumed vs Exported")
    ax.legend(['Self-Consumed', 'Exported to Grid'])
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig(f"{plot_dir}/pv_usage.png", dpi=150)
    plt.close(fig)
    
    # Plot 2: Grid import (for load vs for charging)
    fig2, bx = plt.subplots(figsize=(10, 5))
    daily[['grid_import_for_load', 'grid_import_for_storage']].plot(
        kind='bar',
        stacked=True,
        ax=bx,
        color=['gray', 'cyan'],
        width=0.8
    )
    bx.set_ylabel("Energy (kWh)")
    bx.set_xlabel("Date")
    bx.set_title("Daily Grid Import: for Load vs for Charging")
    bx.legend(['For Load', 'For Charging'])
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig2.savefig(f"{plot_dir}/grid_import.png", dpi=150)
    plt.close(fig2)
