"""Cost-based oracle for prosumer energy optimization.

This module provides a cost-minimization oracle that can be used as an
alternative to ML-based action selection (Mode B).
"""

from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd


class CostOracle:
    """Cost-based decision oracle for energy management.
    
    Makes myopic (single-step) cost-minimization decisions for each hour,
    considering battery and EV state of charge.
    
    Attributes:
        batt_capacity: Battery capacity in kWh
        batt_soc: Current battery state of charge in kWh
        batt_eff: Battery round-trip efficiency (0-1)
        batt_max_charge_kw: Maximum battery charging power in kW
        batt_max_discharge_kw: Maximum battery discharging power in kW
        ev_capacity: EV battery capacity in kWh
        ev_soc: Current EV state of charge in kWh
    """
    
    def __init__(self, cfg: Dict[str, Any]):
        """Initialize cost oracle with configuration.
        
        Args:
            cfg: Configuration dictionary with oracle settings
        """
        oracle_cfg = cfg.get('oracle', {})
        
        # Battery parameters
        self.batt_capacity = oracle_cfg.get('battery_capacity_kwh', 10.0)
        self.batt_soc = 0.0  # Initial state of charge (kWh)
        self.batt_eff = 0.95  # Round-trip efficiency
        self.batt_max_charge_kw = 5.0
        self.batt_max_discharge_kw = 5.0
        
        # EV parameters
        self.ev_capacity = oracle_cfg.get('ev_capacity_kwh', 50.0)
        self.ev_soc = 0.0
        
        # Price series (to be set externally if needed)
        self.import_price: Optional[pd.Series] = None
        self.export_price: Optional[pd.Series] = None
        self.carbon_intensity: Optional[pd.Series] = None

    def decide_action(
        self,
        pv_kw: float,
        load_kw: float,
        price: float,
        hour: int
    ) -> Tuple[str, Dict[str, float]]:
        """Decide optimal action based on instantaneous cost minimization.
        
        This is a myopic (single-step) optimization. For better results,
        consider implementing multi-step lookahead optimization.
        
        Args:
            pv_kw: PV generation in kW
            load_kw: Household load in kW
            price: Electricity price in EUR/kWh
            hour: Hour of day (0-23)
            
        Returns:
            Tuple of (best_action, cost_dict) where:
                - best_action: Recommended action name
                - cost_dict: Dictionary of costs for each action
        """
        # Compute net grid import if idle: idle_cost = (load - pv if positive) * price.
        net_load = load_kw - pv_kw
        idle_import = max(net_load, 0)
        idle_export = max(pv_kw - load_kw, 0)
        # idle cost = import * price - export * price (revenue)
        idle_cost = idle_import * price - idle_export * price
        actions_cost = {}
        # Cost of use_pv_direct: essentially same as idle except you potentially spill surplus (no revenue)
        use_pv_cost = idle_import * price  # if surplus, it's just not monetized but cost is zero for that portion
        actions_cost['use_pv_direct'] = use_pv_cost
        # Cost of sell_to_grid: if surplus, you get revenue
        sell_cost = idle_import * price - idle_export * price  # actually same formula as idle_cost, since idle already assumed export revenue
        actions_cost['sell_to_grid'] = sell_cost
        # Cost of charge_battery: import extra power (up to batt_max_charge_kw or until battery full) if surplus < 0 or if we want to soak surplus
        charge_amount = min(self.batt_max_charge_kw, self.batt_capacity - self.batt_soc)
        # If PV surplus, use that first, any remainder from grid
        surplus = pv_kw - load_kw
        if surplus >= charge_amount:
            # we can charge fully just from surplus (no cost, just lost opportunity of selling)
            charge_import = 0
            surplus -= charge_amount  # leftover surplus after charging
        else:
            # use all surplus plus import the rest
            charge_import = charge_amount - max(surplus, 0)
            surplus = 0
        # Update SOC for cost calc (not permanently updating here, just hypothetically for this hour)
        # Battery charged by charge_amount
        batt_cost = (max(net_load, 0) + charge_import) * price - 0 * price  # paying for import (including extra for charge), no immediate revenue
        actions_cost['charge_battery'] = batt_cost
        # Cost of charge_ev: similar to battery, but maybe only do at night, treat similarly for cost
        charge_ev_amount = min(self.batt_max_charge_kw, self.ev_capacity - self.ev_soc)
        ev_import = charge_ev_amount  # assume all from grid for simplicity
        ev_cost = (max(net_load, 0) + ev_import) * price
        actions_cost['charge_ev'] = ev_cost
        # Cost of idle: already computed as idle_cost
        actions_cost['idle'] = idle_cost
        # Choose min cost action
        best_action = min(actions_cost, key=actions_cost.get)
        return best_action, actions_cost

    def decide_actions_for_series(
        self,
        pv_series: pd.Series,
        load_series: pd.Series,
        price_series: pd.Series
    ) -> List[str]:
        """Apply cost oracle across time series.
        
        Sequentially applies myopic cost minimization for each hour,
        updating battery and EV state of charge as actions are taken.
        
        Args:
            pv_series: Time series of PV generation (kW)
            load_series: Time series of load (kW)
            price_series: Time series of electricity prices (EUR/kWh)
            
        Returns:
            List of recommended actions for each timestep
        """
        actions = []
        
        for t in range(len(pv_series)):
            pv = pv_series.iloc[t]
            load = load_series.iloc[t]
            price = price_series.iloc[t]
            hour = pv_series.index[t].hour
            
            # Decide action
            action, cost_dict = self.decide_action(pv, load, price, hour)
            
            # Update state of charge based on action
            if action == 'charge_battery':
                charge_amount = min(
                    self.batt_max_charge_kw,
                    self.batt_capacity - self.batt_soc
                )
                self.batt_soc = min(
                    self.batt_capacity,
                    self.batt_soc + charge_amount * self.batt_eff
                )
            elif action == 'charge_ev':
                charge_amount = min(
                    self.batt_max_charge_kw,
                    self.ev_capacity - self.ev_soc
                )
                self.ev_soc = min(
                    self.ev_capacity,
                    self.ev_soc + charge_amount
                )
            # Note: Discharge actions not explicitly implemented
            # Could be added for battery-to-home or V2G scenarios
            
            actions.append(action)
        
        return actions
