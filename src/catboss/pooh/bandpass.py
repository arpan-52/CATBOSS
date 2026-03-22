"""
Bandpass normalization for POOH.

Iterative polynomial fitting with sigma-clipping.

Author: Arpan Pal
Institution: NCRA-TIFR
"""

import numpy as np
from typing import Tuple, Optional
from ..utils.gpu import get_jit, get_prange, is_gpu_available

jit = get_jit()
prange = get_prange()


@jit(nopython=True, parallel=True)
def calculate_bandpass_parallel(amp: np.ndarray, flags: np.ndarray) -> np.ndarray:
    """
    Calculate bandpass (median across time for each channel).
    
    Numba-accelerated with parallel processing.
    
    Args:
        amp: Amplitude array (time × frequency)
        flags: Flag array (same shape)
        
    Returns:
        Bandpass array (frequency,)
    """
    n_time, n_chan = amp.shape
    bandpass = np.zeros(n_chan, dtype=np.float32)
    
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
            
            bandpass[j] = np.median(unflagged)
        else:
            bandpass[j] = 0.0
    
    return bandpass


def compute_polynomial_bandpass(
    amp: np.ndarray,
    flags: np.ndarray,
    deviation_threshold: float = 5.0,
    poly_order: int = 5,
    max_iterations: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute iterative polynomial fit for bandpass normalization.
    
    Args:
        amp: Amplitude array (time × frequency)
        flags: Flag array
        deviation_threshold: Threshold in sigma for flagging bad channels
        poly_order: Polynomial order for fitting
        max_iterations: Maximum fitting iterations
        
    Returns:
        poly_fit: Fitted polynomial values for each channel
        bad_channels: Boolean mask of bad channels
    """
    n_chan = amp.shape[1]
    
    # Step 1: Calculate time-averaged spectrum
    spectrum = calculate_bandpass_parallel(amp, flags)
    
    # Step 2: Iterative polynomial fitting with sigma-clipping
    valid_channels = spectrum > 0
    bad_channels = ~valid_channels  # Start with invalid = bad
    poly_fit = spectrum.copy()
    
    if np.sum(valid_channels) > poly_order + 1:
        channel_indices = np.arange(n_chan, dtype=np.float64)
        
        for iteration in range(max_iterations):
            fit_mask = valid_channels & ~bad_channels
            
            if np.sum(fit_mask) <= poly_order + 1:
                break
            
            # Fit polynomial to good channels
            fit_indices = channel_indices[fit_mask]
            fit_spectrum = spectrum[fit_mask].astype(np.float64)
            
            try:
                poly_coeffs = np.polyfit(fit_indices, fit_spectrum, poly_order)
                poly_fit = np.polyval(poly_coeffs, channel_indices).astype(np.float32)
            except np.linalg.LinAlgError:
                break
            
            # Calculate residuals
            residuals = spectrum - poly_fit
            
            # Robust sigma (MAD)
            fit_residuals = residuals[fit_mask]
            if len(fit_residuals) > 0:
                mad = np.median(np.abs(fit_residuals - np.median(fit_residuals)))
                sigma = 1.4826 * mad
                
                if sigma == 0:
                    break
                
                # Find outliers
                new_bad = np.zeros(n_chan, dtype=bool)
                for j in range(n_chan):
                    if fit_mask[j] and np.abs(residuals[j]) > deviation_threshold * sigma:
                        new_bad[j] = True
                
                # Check convergence
                if not np.any(new_bad & ~bad_channels):
                    break
                
                bad_channels |= new_bad
        
        # Final fit
        final_mask = valid_channels & ~bad_channels
        if np.sum(final_mask) > poly_order + 1:
            fit_indices = channel_indices[final_mask]
            fit_spectrum = spectrum[final_mask].astype(np.float64)
            try:
                poly_coeffs = np.polyfit(fit_indices, fit_spectrum, poly_order)
                poly_fit = np.polyval(poly_coeffs, channel_indices).astype(np.float32)
            except np.linalg.LinAlgError:
                pass
    
    return poly_fit, bad_channels


@jit(nopython=True, parallel=True)
def apply_bandpass_normalization_parallel(
    amp: np.ndarray,
    flags: np.ndarray,
    bandpass: np.ndarray,
    bad_channels: np.ndarray
):
    """
    Apply bandpass normalization in-place.
    
    Args:
        amp: Amplitude array (modified in-place)
        flags: Flag array (modified in-place)
        bandpass: Fitted bandpass values
        bad_channels: Boolean mask of bad channels
    """
    n_time, n_chan = amp.shape
    
    for j in prange(n_chan):
        if bad_channels[j]:
            # Flag entire bad channel
            for i in range(n_time):
                flags[i, j] = True
        elif bandpass[j] > 0:
            # Normalize good channel
            for i in range(n_time):
                if not flags[i, j]:
                    amp[i, j] = amp[i, j] / bandpass[j]


def normalize_bandpass(
    amp: np.ndarray,
    flags: np.ndarray,
    poly_order: int = 5,
    deviation_threshold: float = 5.0,
    logger=None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Full bandpass normalization pipeline.
    
    Args:
        amp: Amplitude array (time × frequency) - modified in-place
        flags: Flag array - modified in-place
        poly_order: Polynomial order
        deviation_threshold: Sigma threshold for bad channels
        logger: Optional logger
        
    Returns:
        amp: Normalized amplitude (same array, modified)
        flags: Updated flags (same array, modified)
        bandpass: Fitted bandpass
        bad_channels: Bad channel mask
    """
    n_chan = amp.shape[1]
    
    # Compute polynomial fit
    bandpass, bad_channels = compute_polynomial_bandpass(
        amp, flags, deviation_threshold, poly_order
    )
    
    n_bad = int(np.sum(bad_channels))
    # Just log the count - no verbose channel list
    
    # Apply normalization
    apply_bandpass_normalization_parallel(amp, flags, bandpass, bad_channels)
    
    return amp, flags, bandpass, bad_channels, n_bad


# GPU kernel for bandpass application (if available)
if is_gpu_available():
    from ..utils.gpu import get_cuda
    cuda = get_cuda()
    
    if cuda is not None:
        @cuda.jit
        def bandpass_normalize_kernel(amp, flags, bandpass, bad_channels):
            """CUDA kernel for bandpass normalization."""
            j = cuda.grid(1)
            
            if j < amp.shape[1]:
                if bad_channels[j]:
                    for i in range(amp.shape[0]):
                        flags[i, j] = True
                elif bandpass[j] > 0:
                    for i in range(amp.shape[0]):
                        if not flags[i, j]:
                            amp[i, j] = amp[i, j] / bandpass[j]
