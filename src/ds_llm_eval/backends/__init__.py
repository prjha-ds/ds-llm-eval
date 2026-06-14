"""Optional delegation to validated external evaluation engines (lazy)."""

from .ir import IRBackend, PyTrecEvalBackend, RanxBackend, evaluate_with_backend, get_ir_backend

__all__ = [
    "IRBackend",
    "PyTrecEvalBackend",
    "RanxBackend",
    "evaluate_with_backend",
    "get_ir_backend",
]
