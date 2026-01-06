"""
Standalone DSPy optimization module for drill performance.

This module provides tools for:
- Converting drills to datasets
- Optimizing DSPy modules using GEPA
- Evaluating optimized modules
- Using optimized modules in the main system
"""

from .signature import CatanDrillSignature
from .dataset import DrillDataset
from .agent import DSPyDrillAgent
from .optimizer import DrillOptimizer

__all__ = [
    "CatanDrillSignature",
    "DrillDataset",
    "DSPyDrillAgent",
    "DrillOptimizer",
]

