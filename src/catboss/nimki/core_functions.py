"""
NIMKI Core Functions - C++ Interface Wrappers

Thin Python wrappers around C++ implementations for:
- UV distance calculation
- Data collection
- Gabor basis fitting
- Outlier detection

NIMKI strictly requires the compiled C++ extension (`_nimki_core`). If the
extension is not importable, importing this module raises ImportError with
a hint to rebuild it. There is no Python fallback — the polynomial
approximation that used to live here silently produced fake component dicts
and materially different flag sets.

Author: Arpan Pal
Institution: NCRA-TIFR
"""

import numpy as np
from typing import Dict, Any, Tuple

# Relative import: the extension is built into this package
# (catboss.nimki._nimki_core). A bare `import _nimki_core` only resolves when
# the package directory itself happens to be on sys.path, which is true for an
# in-place build run from this directory but false for any normal install -
# that is how a pip-installed catboss ended up with no working NIMKI.
try:
    from . import _nimki_core
except ImportError as e:
    raise ImportError(
        "NIMKI requires the compiled C++ extension `_nimki_core` but it "
        "could not be imported. It is built automatically by\n"
        "    pip install .\n"
        "from the repository root. For a manual in-place build:\n"
        "    cd src/catboss/nimki && python setup_cpp.py build_ext --inplace\n"
        f"(underlying import error: {e})"
    ) from e

_nami_core = _nimki_core  # backwards-compat alias for any external caller


def is_cpp_available() -> bool:
    return True


def calculate_uv_distances(uvw: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
    """Calculate UV distances for all rows and channels (wavelengths)."""
    return _nimki_core.calculate_uv_distances(
        np.ascontiguousarray(uvw, dtype=np.float64),
        np.ascontiguousarray(wavelengths, dtype=np.float64)
    )


def collect_data_single_corr(
    data: np.ndarray,
    flags: np.ndarray,
    uv_distances: np.ndarray,
    spw_row_indices: np.ndarray,
    corr_index: int
) -> Dict[str, np.ndarray]:
    """Collect unflagged amplitudes for a single correlation."""
    return _nimki_core.collect_data(
        np.ascontiguousarray(data, dtype=np.complex64),
        np.ascontiguousarray(flags, dtype=bool),
        np.ascontiguousarray(uv_distances, dtype=np.float64),
        np.asarray(spw_row_indices, dtype=np.int32),
        int(corr_index)
    )


def fit_gabor(
    uv_dists: np.ndarray,
    amplitudes: np.ndarray,
    n_components: int = 5,
    max_iter: int = 500,
    tol: float = 1e-8,
    n_restarts: int = 2
) -> Dict[str, Any]:
    """
    Fit Gabor basis model: V(r) = Σ Aᵢ · exp(-(r/σᵢ)²/2) · cos(ωᵢ·r + φᵢ)
    """
    return _nimki_core.fit_gabor(
        np.ascontiguousarray(uv_dists, dtype=np.float64),
        np.ascontiguousarray(amplitudes, dtype=np.float64),
        int(n_components),
        int(max_iter),
        float(tol),
        int(n_restarts)
    )


def fit_gabor_adaptive(
    uv_dists: np.ndarray,
    amplitudes: np.ndarray,
    n_components: int = 5,
    max_components: int = 12,
    min_improvement: float = 0.05,
    max_iter: int = 500,
    tol: float = 1e-8
) -> Dict[str, Any]:
    """Adaptive Gabor fit: add components until relative gain < min_improvement."""
    return _nimki_core.fit_gabor_adaptive(
        np.ascontiguousarray(uv_dists, dtype=np.float64),
        np.ascontiguousarray(amplitudes, dtype=np.float64),
        int(n_components),
        int(max_components),
        float(min_improvement),
        int(max_iter),
        float(tol)
    )


def flag_outliers(
    amplitudes: np.ndarray,
    predicted: np.ndarray,
    sigma_threshold: float
) -> Tuple[np.ndarray, np.ndarray, float]:
    """MAD-based outlier flagging on (amplitudes - predicted)."""
    result = _nimki_core.flag_outliers(
        np.ascontiguousarray(amplitudes, dtype=np.float64),
        np.ascontiguousarray(predicted, dtype=np.float64),
        float(sigma_threshold)
    )
    return result['outliers'], result['residuals'], result['mad_sigma']
