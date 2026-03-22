"""
NIMKI - Non-linear Interference Modeling and Korrection Interface

UV-domain RFI flagging using Gabor basis fitting.

Model: V(r) = Σ Aᵢ · exp(-(r/σᵢ)²/2) · cos(ωᵢ·r + φᵢ)

Author: Arpan Pal
Institution: NCRA-TIFR
"""

from .engine import hunt_ms
from .core_functions import (
    calculate_uv_distances,
    collect_data_single_corr,
    fit_gabor,
    fit_gabor_adaptive,
    flag_outliers,
    is_cpp_available,
)

__all__ = [
    'hunt_ms',
    'calculate_uv_distances',
    'collect_data_single_corr',
    'fit_gabor',
    'fit_gabor_adaptive',
    'flag_outliers',
    'is_cpp_available',
]
