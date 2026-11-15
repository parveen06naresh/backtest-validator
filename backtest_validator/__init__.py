"""Backtest Validator: a battery of tests that tries to kill false positives
in trading-signal research. Research and education only; not advice."""
from .data import Market, load
from .engine import Costs
from .pipeline import Config, Report, validate
from .strategies import Rule, dense_family, standard_family

__version__ = "0.2.0"
__all__ = ["Market", "load", "Costs", "Config", "Report", "validate", "Rule",
           "standard_family", "dense_family"]
