"""Counterfactual simulation primitives shared by the generator and the ROI simulator."""

from mro_intelligence.simulation.inventory import (
    InventoryPath,
    PolicyParams,
    simulate_sku_inventory,
)

__all__ = ["InventoryPath", "PolicyParams", "simulate_sku_inventory"]
