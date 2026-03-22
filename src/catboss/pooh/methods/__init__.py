"""
POOH flagging methods.

Available methods:
- SumThreshold: Multi-scale combinatorial thresholding (Offringa et al. 2010)
- IQR: Inter-quartile range outlier detection
- MAD: Median Absolute Deviation flagging

All methods support batch processing via flag_batch() interface.
"""

from typing import Dict, Any, Optional
from .base import BaseFlaggingMethod
from .sumthreshold import SumThresholdMethod
from .iqr import IQRMethod
from .mad import MADMethod

# Method registry
METHODS = {
    'sumthreshold': SumThresholdMethod,
    'iqr': IQRMethod,
    'mad': MADMethod,
}


def get_method(name: str, params: Optional[Dict[str, Any]] = None, logger=None) -> BaseFlaggingMethod:
    """
    Get instantiated method by name.
    
    Args:
        name: Method name (sumthreshold, iqr, mad)
        params: Parameters to pass to method
        logger: Optional logger
        
    Returns:
        Instantiated method object
    """
    name_lower = name.lower()
    if name_lower not in METHODS:
        available = ', '.join(METHODS.keys())
        raise ValueError(f"Unknown method '{name}'. Available: {available}")
    
    method_class = METHODS[name_lower]
    return method_class(params or {}, logger=logger)


def list_methods() -> list:
    """List available method names."""
    return list(METHODS.keys())


__all__ = [
    'BaseFlaggingMethod',
    'SumThresholdMethod',
    'IQRMethod',
    'MADMethod',
    'METHODS',
    'get_method',
    'list_methods',
]
