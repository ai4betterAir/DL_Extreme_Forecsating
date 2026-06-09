"""
Backward-compatible shim.

The training evaluation helpers were moved to `Core_iHPC/Evaluation` to keep
`Core_iHPC/Processing` focused on data processing + orchestration.

Old imports (kept working):
    from Core_iHPC.Processing import Training_evaluations
"""

from Core_iHPC.Evaluation.Training_evaluations import Training_evaluations_Class

__all__ = ["Training_evaluations_Class"]
