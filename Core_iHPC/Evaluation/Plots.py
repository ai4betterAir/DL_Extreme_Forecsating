"""
Backward-compatible plotting module.

Some parts of the codebase still import ``Core_iHPC.Evaluation.Plots``.
This shim preserves that import path while delegating to ``OldPlots``.
"""

from .OldPlots import Plot_Class

