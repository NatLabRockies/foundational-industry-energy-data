"""Source Classification Codes

Methods to download and apply EPA's Source Classification Codes for characterizing units.
"""
import polars as pl

from .core import scc_unit_and_fuel_types

__all__ = ["scc_unit_and_fuel_types"]
