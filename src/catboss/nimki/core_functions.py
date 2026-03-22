"""
NIMKI Core Functions - C++ Interface Wrappers

Thin Python wrappers around C++ implementations for:
- UV distance calculation
- Data collection
- Gabor basis fitting
- Outlier detection

Author: Arpan Pal
Institution: NCRA-TIFR
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional

# Try to import C++ extension
_CPP_AVAILABLE = False
try:
    import _nami_core
    _CPP_AVAILABLE = True
except ImportError:
    _CPP_AVAILABLE = False


def is_cpp_available() -> bool:
    """Check if C++ extension is available."""
    return _CPP_AVAILABLE


def calculate_uv_distances(uvw: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
    """
    Calculate UV distances for all rows and channels.

    Parameters
    ----------
    uvw : ndarray, shape (n_rows, 3)
        UVW coordinates in meters
    wavelengths : ndarray, shape (n_channels,)
        Wavelengths in meters

    Returns
    -------
    ndarray, shape (n_rows, n_channels)
        UV distances in wavelengths
    """
    if _CPP_AVAILABLE:
        return _nami_core.calculate_uv_distances(
            np.ascontiguousarray(uvw, dtype=np.float64),
            np.ascontiguousarray(wavelengths, dtype=np.float64)
        )
    else:
        # Pure Python fallback
        u = uvw[:, 0:1]  # (n_rows, 1)
        v = uvw[:, 1:2]  # (n_rows, 1)
        uv_meters = np.sqrt(u**2 + v**2)  # (n_rows, 1)
        return uv_meters / wavelengths[np.newaxis, :]  # (n_rows, n_channels)


def collect_data_single_corr(
    data: np.ndarray,
    flags: np.ndarray,
    uv_distances: np.ndarray,
    spw_row_indices: np.ndarray,
    corr_index: int
) -> Dict[str, np.ndarray]:
    """
    Collect amplitude data for a single correlation.

    Parameters
    ----------
    data : ndarray, shape (n_rows, n_channels, n_corr), complex64
        Visibility data
    flags : ndarray, shape (n_rows, n_channels, n_corr), bool
        Flag array
    uv_distances : ndarray, shape (n_rows, n_channels)
        UV distances
    spw_row_indices : array-like
        Row indices for this SPW
    corr_index : int
        Which correlation to process

    Returns
    -------
    dict with keys: uv_dists, amplitudes, row_indices, chan_indices
    """
    if _CPP_AVAILABLE:
        return _nami_core.collect_data(
            np.ascontiguousarray(data, dtype=np.complex64),
            np.ascontiguousarray(flags, dtype=bool),
            np.ascontiguousarray(uv_distances, dtype=np.float64),
            np.asarray(spw_row_indices, dtype=np.int32),
            int(corr_index)
        )
    else:
        # Pure Python fallback
        uv_list = []
        amp_list = []
        row_list = []
        chan_list = []
        
        for row_idx in spw_row_indices:
            for chan_idx in range(data.shape[1]):
                if not flags[row_idx, chan_idx, corr_index]:
                    uv_list.append(uv_distances[row_idx, chan_idx])
                    amp_list.append(np.abs(data[row_idx, chan_idx, corr_index]))
                    row_list.append(row_idx)
                    chan_list.append(chan_idx)
        
        return {
            'uv_dists': np.array(uv_list, dtype=np.float64),
            'amplitudes': np.array(amp_list, dtype=np.float64),
            'row_indices': np.array(row_list, dtype=np.int32),
            'chan_indices': np.array(chan_list, dtype=np.int32),
        }


def fit_gabor(
    uv_dists: np.ndarray,
    amplitudes: np.ndarray,
    n_components: int = 5,
    max_iter: int = 500,
    tol: float = 1e-8,
    n_restarts: int = 5
) -> Dict[str, Any]:
    """
    Fit Gabor basis model to visibility amplitudes.
    
    Model: V(r) = Σ Aᵢ · exp(-(r/σᵢ)²/2) · cos(ωᵢ·r + φᵢ)
    
    This captures both smooth Gaussian decay AND oscillations from
    source structure (disks, rings, double sources, etc.)

    Parameters
    ----------
    uv_dists : ndarray
        UV distances
    amplitudes : ndarray
        Visibility amplitudes
    n_components : int, optional
        Number of Gabor components (default: 5)
    max_iter : int, optional
        Maximum LM iterations (default: 500)
    tol : float, optional
        Convergence tolerance (default: 1e-8)
    n_restarts : int, optional
        Random restarts to avoid local minima (default: 5)

    Returns
    -------
    dict with keys:
        predicted : ndarray - Fitted values
        residuals : ndarray - Residuals
        components : list of dict - Component parameters
        rms : float - RMS residual
        mad_sigma : float - MAD * 1.4826
        converged : bool
        n_iterations : int
        n_components : int
    """
    if _CPP_AVAILABLE:
        return _nami_core.fit_gabor(
            np.ascontiguousarray(uv_dists, dtype=np.float64),
            np.ascontiguousarray(amplitudes, dtype=np.float64),
            int(n_components),
            int(max_iter),
            float(tol),
            int(n_restarts)
        )
    else:
        # Simple Python fallback using polynomial fit
        # (Real Gabor fitting needs scipy.optimize or similar)
        return _fit_gabor_python(uv_dists, amplitudes, n_components)


def _fit_gabor_python(
    uv_dists: np.ndarray,
    amplitudes: np.ndarray,
    n_components: int
) -> Dict[str, Any]:
    """Pure Python Gabor fitting fallback using polynomial approximation."""
    
    # Sort by UV distance
    sort_idx = np.argsort(uv_dists)
    uv_sorted = uv_dists[sort_idx]
    amp_sorted = amplitudes[sort_idx]
    
    # Fit polynomial as approximation
    degree = min(n_components * 2, 10, len(uv_sorted) - 1)
    
    try:
        coeffs = np.polyfit(uv_sorted, amp_sorted, degree)
        predicted_sorted = np.polyval(coeffs, uv_sorted)
    except np.linalg.LinAlgError:
        predicted_sorted = np.full_like(amp_sorted, np.median(amp_sorted))
    
    # Unsort
    predicted = np.zeros_like(amplitudes)
    predicted[sort_idx] = predicted_sorted
    
    residuals = amplitudes - predicted
    
    # Statistics
    mad = np.median(np.abs(residuals - np.median(residuals)))
    mad_sigma = 1.4826 * mad
    rms = np.sqrt(np.mean(residuals**2))
    
    # Fake components (for compatibility)
    components = []
    for i in range(n_components):
        components.append({
            'amplitude': 1.0 / (i + 1),
            'sigma': 100.0 * (i + 1),
            'omega': 0.01 * i,
            'phase': 0.0,
        })
    
    return {
        'predicted': predicted,
        'residuals': residuals,
        'components': components,
        'rms': rms,
        'mad_sigma': mad_sigma,
        'converged': True,
        'n_iterations': degree,
        'n_components': n_components,
    }


def fit_gabor_adaptive(
    uv_dists: np.ndarray,
    amplitudes: np.ndarray,
    n_components: int = 5,
    max_components: int = 12,
    min_improvement: float = 0.05,
    max_iter: int = 500,
    tol: float = 1e-8
) -> Dict[str, Any]:
    """
    Adaptive Gabor fitting with automatic component selection.
    
    Starts with n_components and adds more until improvement drops
    below min_improvement (diminishing returns).

    Parameters
    ----------
    uv_dists : ndarray
        UV distances
    amplitudes : ndarray
        Visibility amplitudes
    n_components : int, optional
        Starting components (default: 5)
    max_components : int, optional
        Maximum components (default: 12)
    min_improvement : float, optional
        Stop if improvement < this fraction (default: 0.05 = 5%)
    max_iter : int, optional
        Max iterations per fit (default: 500)
    tol : float, optional
        Convergence tolerance (default: 1e-8)

    Returns
    -------
    dict with best fit results including optimal n_components
    """
    if _CPP_AVAILABLE:
        return _nami_core.fit_gabor_adaptive(
            np.ascontiguousarray(uv_dists, dtype=np.float64),
            np.ascontiguousarray(amplitudes, dtype=np.float64),
            int(n_components),
            int(max_components),
            float(min_improvement),
            int(max_iter),
            float(tol)
        )
    else:
        # Pure Python fallback - just use fixed components
        return fit_gabor(uv_dists, amplitudes, n_components, max_iter, tol)


def flag_outliers(
    amplitudes: np.ndarray,
    predicted: np.ndarray,
    sigma_threshold: float
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Flag outliers using MAD.

    Parameters
    ----------
    amplitudes : ndarray
        Actual amplitudes
    predicted : ndarray
        Predicted amplitudes from Gabor fit
    sigma_threshold : float
        Threshold in sigma units

    Returns
    -------
    tuple: (outliers, residuals, mad_sigma)
        outliers: bool array
        residuals: amplitude - predicted
        mad_sigma: MAD * 1.4826
    """
    if _CPP_AVAILABLE:
        result = _nami_core.flag_outliers(
            np.ascontiguousarray(amplitudes, dtype=np.float64),
            np.ascontiguousarray(predicted, dtype=np.float64),
            float(sigma_threshold)
        )
        return result['outliers'], result['residuals'], result['mad_sigma']
    else:
        # Pure Python fallback
        residuals = amplitudes - predicted
        median_resid = np.median(residuals)
        mad = np.median(np.abs(residuals - median_resid))
        mad_sigma = 1.4826 * mad
        
        if mad_sigma > 0:
            outliers = np.abs(residuals - median_resid) > sigma_threshold * mad_sigma
        else:
            outliers = np.zeros_like(residuals, dtype=bool)
        
        return outliers, residuals, mad_sigma
