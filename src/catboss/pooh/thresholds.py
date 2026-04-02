"""
Threshold calculation utilities for POOH.

Robust per-channel threshold estimation.

Author: Arpan Pal
Institution: NCRA-TIFR
"""

import numpy as np
from typing import Optional
from ..utils.gpu import get_jit, get_prange

jit = get_jit()
prange = get_prange()


@jit(nopython=True, parallel=True)
def calculate_channel_medians_parallel(
    amp: np.ndarray,
    flags: np.ndarray,
    sigma_factor: float
) -> np.ndarray:
    """
    Calculate per-channel thresholds using median.
    
    Numba-accelerated with parallel processing.
    
    Args:
        amp: Amplitude array (time × frequency)
        flags: Flag array
        sigma_factor: Threshold multiplier
        
    Returns:
        Per-channel thresholds
    """
    n_time, n_chan = amp.shape
    thresholds = np.zeros(n_chan, dtype=np.float32)
    
    for j in prange(n_chan):
        # Count unflagged
        n_unflagged = 0
        for i in range(n_time):
            if not flags[i, j]:
                n_unflagged += 1
        
        if n_unflagged > 0:
            unflagged = np.empty(n_unflagged, dtype=amp.dtype)
            idx = 0
            for i in range(n_time):
                if not flags[i, j]:
                    unflagged[idx] = amp[i, j]
                    idx += 1
            
            median = np.median(unflagged)
            thresholds[j] = median * sigma_factor
        else:
            thresholds[j] = -1.0  # Mark for interpolation
    
    return thresholds


@jit(nopython=True, parallel=True)
def calculate_channel_mad_parallel(
    amp: np.ndarray,
    flags: np.ndarray,
    sigma_factor: float
) -> np.ndarray:
    """
    Calculate per-channel thresholds using MAD (Median Absolute Deviation).
    
    Args:
        amp: Amplitude array (time × frequency)
        flags: Flag array
        sigma_factor: Threshold multiplier
        
    Returns:
        Per-channel thresholds based on MAD
    """
    n_time, n_chan = amp.shape
    thresholds = np.zeros(n_chan, dtype=np.float32)
    
    for j in prange(n_chan):
        # Count unflagged
        n_unflagged = 0
        for i in range(n_time):
            if not flags[i, j]:
                n_unflagged += 1
        
        if n_unflagged > 0:
            unflagged = np.empty(n_unflagged, dtype=amp.dtype)
            idx = 0
            for i in range(n_time):
                if not flags[i, j]:
                    unflagged[idx] = amp[i, j]
                    idx += 1
            
            median = np.median(unflagged)
            
            # MAD calculation
            deviations = np.empty(n_unflagged, dtype=amp.dtype)
            for k in range(n_unflagged):
                deviations[k] = np.abs(unflagged[k] - median)
            
            mad = np.median(deviations)
            sigma_mad = 1.4826 * mad  # Convert to Gaussian equivalent
            
            thresholds[j] = median + sigma_factor * sigma_mad
        else:
            thresholds[j] = -1.0
    
    return thresholds


@jit(nopython=True, parallel=True)
def calculate_channel_iqr_parallel(
    amp: np.ndarray,
    flags: np.ndarray,
    iqr_factor: float
) -> np.ndarray:
    """
    Calculate per-channel thresholds using IQR (Inter-Quartile Range).
    
    Args:
        amp: Amplitude array (time × frequency)
        flags: Flag array
        iqr_factor: IQR multiplier
        
    Returns:
        Per-channel upper thresholds (Q3 + factor * IQR)
    """
    n_time, n_chan = amp.shape
    thresholds = np.zeros(n_chan, dtype=np.float32)
    
    for j in prange(n_chan):
        n_unflagged = 0
        for i in range(n_time):
            if not flags[i, j]:
                n_unflagged += 1
        
        if n_unflagged >= 4:  # Need at least 4 points for quartiles
            unflagged = np.empty(n_unflagged, dtype=amp.dtype)
            idx = 0
            for i in range(n_time):
                if not flags[i, j]:
                    unflagged[idx] = amp[i, j]
                    idx += 1
            
            # Sort for percentiles
            unflagged.sort()
            
            # Q1 (25th percentile)
            q1_idx = int(n_unflagged * 0.25)
            q1 = unflagged[q1_idx]
            
            # Q3 (75th percentile)
            q3_idx = int(n_unflagged * 0.75)
            q3 = unflagged[q3_idx]
            
            iqr = q3 - q1
            thresholds[j] = q3 + iqr_factor * iqr
        else:
            thresholds[j] = -1.0
    
    return thresholds


def calculate_robust_thresholds(
    amp: np.ndarray,
    flags: np.ndarray,
    sigma_factor: float = 6.0,
    method: str = 'median'
) -> np.ndarray:
    """
    Calculate robust per-channel thresholds.
    
    Args:
        amp: Amplitude array (time × frequency)
        flags: Flag array
        sigma_factor: Threshold multiplier
        method: 'median', 'mad', or 'iqr'
        
    Returns:
        Per-channel thresholds
    """
    if method == 'mad':
        thresholds = calculate_channel_mad_parallel(amp, flags, sigma_factor)
    elif method == 'iqr':
        thresholds = calculate_channel_iqr_parallel(amp, flags, sigma_factor)
    else:  # median
        thresholds = calculate_channel_medians_parallel(amp, flags, sigma_factor)
    
    # Interpolate flagged channels
    flagged_channels = np.where(thresholds < 0)[0]
    
    if len(flagged_channels) > 0:
        valid_channels = np.where(thresholds > 0)[0]
        
        if len(valid_channels) > 0:
            for j in flagged_channels:
                if len(valid_channels) == 1:
                    thresholds[j] = thresholds[valid_channels[0]]
                else:
                    # Interpolate from nearest valid
                    left_idx = valid_channels[valid_channels < j]
                    right_idx = valid_channels[valid_channels > j]
                    
                    if len(left_idx) > 0 and len(right_idx) > 0:
                        left_val = thresholds[left_idx[-1]]
                        right_val = thresholds[right_idx[0]]
                        thresholds[j] = (left_val + right_val) / 2
                    elif len(left_idx) > 0:
                        thresholds[j] = thresholds[left_idx[-1]]
                    else:
                        thresholds[j] = thresholds[right_idx[0]]
        else:
            # All flagged - use global (guard against NaN / zero)
            finite_amp = amp[np.isfinite(amp) & (amp > 0)]
            if len(finite_amp) > 0:
                global_median = np.median(finite_amp) * sigma_factor
            else:
                global_median = 1e10  # Effectively flag nothing
            thresholds[:] = global_median
    
    return thresholds


def calculate_sumthreshold_levels(
    base_threshold: np.ndarray,
    rho: float,
    combinations: list
) -> dict:
    """
    Calculate threshold levels for each SumThreshold window size.
    
    Based on Offringa et al. 2010: χᵢ = χ₁ / ρ^(log₂(i))
    
    Args:
        base_threshold: Base per-channel thresholds (χ₁)
        rho: Reduction factor (default 1.5)
        combinations: List of window sizes [1, 2, 4, 8, ...]
        
    Returns:
        Dictionary mapping window size to threshold array
    """
    levels = {}
    
    for M in combinations:
        if M == 1:
            levels[M] = base_threshold.copy()
        else:
            # χₘ = χ₁ / ρ^(log₂(M))
            factor = rho ** np.log2(M)
            levels[M] = base_threshold / factor
    
    return levels
