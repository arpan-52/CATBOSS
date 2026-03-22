"""
POOH - Parallelized Optimized Outlier Hunter

Dynamic spectra-based RFI flagging for radio astronomy.

Features:
- Batch GPU processing (entire batch transferred to GPU together)
- Async prefetch (load next batch while GPU processes current)
- Three flagging methods: SumThreshold, IQR, MAD
- Time-frequency chunking
- Multi-pass processing

Author: Arpan Pal
Institution: NCRA-TIFR
"""

from .engine import hunt_ms
from .bandpass import normalize_bandpass, compute_polynomial_bandpass
from .thresholds import calculate_robust_thresholds, calculate_sumthreshold_levels
from .methods import get_method, list_methods, METHODS

__all__ = [
    'hunt_ms',
    'normalize_bandpass',
    'compute_polynomial_bandpass',
    'calculate_robust_thresholds',
    'calculate_sumthreshold_levels',
    'get_method',
    'list_methods',
    'METHODS',
]
