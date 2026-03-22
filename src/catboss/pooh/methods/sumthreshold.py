"""
SumThreshold RFI flagging method.

Based on: Offringa et al. (2010) "Post-correlation radio frequency 
interference classification methods" (arXiv:1002.1957)

The SumThreshold method uses combinatorial thresholding:
- χᵢ = χ₁ / ρ^(log₂(i)) where i is the window size
- Default: χ₁ (sigma) = 6.0, ρ (rho) = 1.5
- Window sizes M = {1, 2, 4, 8, 16, 32, 64}

Supports both single-baseline and batch processing on GPU.

Author: Arpan Pal
Institution: NCRA-TIFR
"""

import numpy as np
from typing import Dict, Any, Optional, List
from .base import BaseFlaggingMethod
from ..thresholds import calculate_robust_thresholds, calculate_sumthreshold_levels
from ...utils.gpu import (
    get_jit, get_prange, is_gpu_available, get_cuda,
    get_optimal_block_size, get_optimal_block_size_3d
)

jit = get_jit()
prange = get_prange()


# ==================== CPU IMPLEMENTATIONS ====================

@jit(nopython=True, parallel=True)
def _sum_threshold_cpu_batch(
    amp: np.ndarray,
    flags: np.ndarray,
    thresholds: np.ndarray,
    combinations: np.ndarray,
    rho: float
):
    """
    SumThreshold for batch of baselines (CPU, parallel over baselines).
    
    Args:
        amp: Amplitude array (n_baselines × n_time × n_freq), float32
        flags: Flag array (n_baselines × n_time × n_freq), uint8, modified in-place
        thresholds: Base thresholds (n_baselines × n_freq), float32
        combinations: Window sizes array
        rho: Reduction factor
    """
    n_bl, n_time, n_freq = amp.shape
    
    # Process each baseline in parallel
    for bl in prange(n_bl):
        # Process each window size
        for m_idx in range(len(combinations)):
            M = combinations[m_idx]
            
            # Calculate threshold scaling: χₘ = χ₁ / ρ^(log₂(M))
            if M > 1:
                scale = rho ** (np.log2(float(M)))
            else:
                scale = 1.0
            
            # Time direction sliding window
            for t in range(n_time):
                for f in range(n_freq - M + 1):
                    # Average threshold for this window
                    avg_thresh = 0.0
                    for k in range(M):
                        avg_thresh += thresholds[bl, f + k]
                    avg_thresh = (avg_thresh / M) / scale
                    
                    # Sum unflagged samples
                    window_sum = 0.0
                    count = 0
                    for k in range(M):
                        if flags[bl, t, f + k] == 0:
                            window_sum += amp[bl, t, f + k]
                            count += 1
                    
                    # Need at least 30% unflagged
                    min_unflagged = max(1, int(M * 0.3))
                    
                    if count >= min_unflagged:
                        avg_val = window_sum / count
                        if avg_val > avg_thresh:
                            # Flag samples exceeding threshold
                            for k in range(M):
                                if flags[bl, t, f + k] == 0:
                                    if amp[bl, t, f + k] > thresholds[bl, f + k] / scale:
                                        flags[bl, t, f + k] = 1
            
            # Frequency direction sliding window
            for t in range(n_time - M + 1):
                for f in range(n_freq):
                    thresh = thresholds[bl, f] / scale
                    
                    window_sum = 0.0
                    count = 0
                    for k in range(M):
                        if flags[bl, t + k, f] == 0:
                            window_sum += amp[bl, t + k, f]
                            count += 1
                    
                    min_unflagged = max(1, int(M * 0.3))
                    
                    if count >= min_unflagged:
                        avg_val = window_sum / count
                        if avg_val > thresh:
                            for k in range(M):
                                if flags[bl, t + k, f] == 0:
                                    if amp[bl, t + k, f] > thresh:
                                        flags[bl, t + k, f] = 1


# ==================== GPU IMPLEMENTATIONS ====================

# GPU kernels are defined conditionally
_gpu_kernels_defined = False
_sumthreshold_gpu_time_kernel = None
_sumthreshold_gpu_freq_kernel = None

if is_gpu_available():
    cuda = get_cuda()
    
    if cuda is not None:
        @cuda.jit
        def _sumthreshold_gpu_time_3d(amp, flags, thresholds, M, scale):
            """
            CUDA kernel for SumThreshold in time direction (3D batch).
            
            Grid: (n_baselines, n_time_blocks, n_freq_blocks)
            Each thread handles one (bl, t, f) position.
            """
            bl, t, f = cuda.grid(3)
            
            n_bl, n_time, n_freq = amp.shape
            
            if bl < n_bl and t < n_time and f < n_freq - M + 1:
                # Calculate average threshold for window
                avg_thresh = 0.0
                for k in range(M):
                    avg_thresh += thresholds[bl, f + k]
                avg_thresh = (avg_thresh / M) / scale
                
                # Sum unflagged samples in window
                window_sum = 0.0
                count = 0
                for k in range(M):
                    if flags[bl, t, f + k] == 0:
                        window_sum += amp[bl, t, f + k]
                        count += 1
                
                min_unflagged = max(1, int(M * 0.3))
                
                if count >= min_unflagged:
                    avg_val = window_sum / count
                    if avg_val > avg_thresh:
                        # Flag individual samples exceeding threshold
                        for k in range(M):
                            if flags[bl, t, f + k] == 0:
                                sample_thresh = thresholds[bl, f + k] / scale
                                if amp[bl, t, f + k] > sample_thresh:
                                    flags[bl, t, f + k] = 1

        @cuda.jit
        def _sumthreshold_gpu_freq_3d(amp, flags, thresholds, M, scale):
            """
            CUDA kernel for SumThreshold in frequency direction (3D batch).
            """
            bl, t, f = cuda.grid(3)
            
            n_bl, n_time, n_freq = amp.shape
            
            if bl < n_bl and t < n_time - M + 1 and f < n_freq:
                thresh = thresholds[bl, f] / scale
                
                # Sum unflagged samples
                window_sum = 0.0
                count = 0
                for k in range(M):
                    if flags[bl, t + k, f] == 0:
                        window_sum += amp[bl, t + k, f]
                        count += 1
                
                min_unflagged = max(1, int(M * 0.3))
                
                if count >= min_unflagged:
                    avg_val = window_sum / count
                    if avg_val > thresh:
                        for k in range(M):
                            if flags[bl, t + k, f] == 0:
                                if amp[bl, t + k, f] > thresh:
                                    flags[bl, t + k, f] = 1
        
        _gpu_kernels_defined = True
        _sumthreshold_gpu_time_kernel = _sumthreshold_gpu_time_3d
        _sumthreshold_gpu_freq_kernel = _sumthreshold_gpu_freq_3d


class SumThresholdMethod(BaseFlaggingMethod):
    """
    SumThreshold flagging method.
    
    Implements combinatorial thresholding from Offringa et al. (2010).
    Supports both CPU (parallel) and GPU (batch) execution.
    """
    
    name = "sumthreshold"
    
    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        return {
            'sigma': 6.0,
            'rho': 1.5,
            'combinations': [1, 2, 4, 8, 16, 32, 64],
            'use_gpu': True,
        }
    
    @classmethod
    def get_param_help(cls) -> Dict[str, str]:
        return {
            'sigma': 'Base threshold multiplier χ₁ (default: 6.0)',
            'rho': 'Window reduction factor ρ (default: 1.5)',
            'combinations': 'Window sizes M (default: 1,2,4,8,16,32,64)',
            'use_gpu': 'Use GPU if available (default: True)',
        }
    
    def flag_batch(
        self,
        amp_batch: np.ndarray,
        flags_batch: np.ndarray,
        thresholds_batch: np.ndarray,
        gpu_arrays=None
    ) -> np.ndarray:
        """
        Apply SumThreshold flagging to a batch of baselines.
        
        This is the primary interface for batch processing.
        
        Args:
            amp_batch: Amplitude array (n_baselines × n_time × n_freq), float32
            flags_batch: Existing flags (n_baselines × n_time × n_freq), uint8
            thresholds_batch: Base thresholds (n_baselines × n_freq), float32
            
        Returns:
            Updated flags array (modified in-place and returned)
        """
        rho = self.params.get('rho', 1.5)
        combinations = self.params.get('combinations', [1, 2, 4, 8, 16, 32, 64])
        use_gpu = self.params.get('use_gpu', True) and is_gpu_available() and _gpu_kernels_defined
        
        if use_gpu:
            self._flag_batch_gpu(amp_batch, flags_batch, thresholds_batch, combinations, rho, gpu_arrays=gpu_arrays)
        else:
            self._flag_batch_cpu(amp_batch, flags_batch, thresholds_batch, combinations, rho)
        
        return flags_batch
    
    def _flag_batch_cpu(
        self,
        amp: np.ndarray,
        flags: np.ndarray,
        thresholds: np.ndarray,
        combinations: List[int],
        rho: float
    ):
        """CPU batch flagging with Numba parallel."""
        combi_arr = np.array(combinations, dtype=np.int32)
        
        # Ensure correct dtypes
        amp = amp.astype(np.float32)
        flags = flags.astype(np.uint8)
        thresholds = thresholds.astype(np.float32)
        
        _sum_threshold_cpu_batch(amp, flags, thresholds, combi_arr, rho)
        
        self.log(f"    SumThreshold CPU: {amp.shape[0]} baselines, {len(combinations)} windows")
    
    def _flag_batch_gpu(
        self,
        amp: np.ndarray,
        flags: np.ndarray,
        thresholds: np.ndarray,
        combinations: List[int],
        rho: float,
        gpu_arrays=None
    ):
        """GPU batch flagging - all baselines processed together."""
        cuda = get_cuda()

        if cuda is None:
            self._flag_batch_cpu(amp, flags, thresholds, combinations, rho)
            return

        n_bl, n_time, n_freq = amp.shape
        amp_c = np.ascontiguousarray(amp, dtype=np.float32)
        flags_c = np.ascontiguousarray(flags, dtype=np.uint8)

        try:
            # Use pre-allocated GPU arrays if available and shapes match
            if (gpu_arrays is not None and
                    gpu_arrays['d_amp'].shape[0] >= n_bl and
                    gpu_arrays['d_amp'].shape[1] == n_time and
                    gpu_arrays['d_amp'].shape[2] == n_freq):
                d_amp = gpu_arrays['d_amp'][:n_bl]
                d_flags = gpu_arrays['d_flags'][:n_bl]
                d_amp.copy_to_device(amp_c)
                d_flags.copy_to_device(flags_c)
            else:
                d_amp = cuda.to_device(amp_c)
                d_flags = cuda.to_device(flags_c)
            d_thresh = cuda.to_device(np.ascontiguousarray(thresholds, dtype=np.float32))
            
            # Calculate grid dimensions
            threads = (1, 16, 16)  # (baseline, time, freq)
            blocks = (
                n_bl,
                (n_time + threads[1] - 1) // threads[1],
                (n_freq + threads[2] - 1) // threads[2]
            )
            
            # Process each window size
            for M in combinations:
                # Calculate scale factor
                if M > 1:
                    scale = rho ** np.log2(float(M))
                else:
                    scale = 1.0
                
                # Time direction kernel
                _sumthreshold_gpu_time_kernel[blocks, threads](
                    d_amp, d_flags, d_thresh, M, scale
                )
                
                # Frequency direction kernel
                _sumthreshold_gpu_freq_kernel[blocks, threads](
                    d_amp, d_flags, d_thresh, M, scale
                )
            
            # Single transfer back
            cuda.synchronize()
            d_flags.copy_to_host(flags)
            
            self.log(f"    SumThreshold GPU: {n_bl} baselines, {len(combinations)} windows")
            
        except Exception as e:
            self.log(f"    GPU failed ({e}), falling back to CPU", level='warning')
            self._flag_batch_cpu(amp, flags, thresholds, combinations, rho)
    
    # Legacy single-baseline interface
    def flag(
        self,
        amp: np.ndarray,
        flags: np.ndarray,
        thresholds: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Apply SumThreshold flagging to single baseline.
        
        Args:
            amp: Amplitude array (n_time × n_freq)
            flags: Existing flags
            thresholds: Optional pre-calculated base thresholds
            
        Returns:
            Combined flag array
        """
        sigma = self.params.get('sigma', 6.0)
        
        # Calculate thresholds if not provided
        if thresholds is None:
            thresholds = calculate_robust_thresholds(amp, flags.astype(bool), sigma)
        
        # Reshape to batch format (1, n_time, n_freq)
        amp_batch = amp[np.newaxis, :, :].astype(np.float32)
        flags_batch = flags[np.newaxis, :, :].astype(np.uint8)
        thresh_batch = thresholds[np.newaxis, :].astype(np.float32)
        
        # Process as batch of 1
        self.flag_batch(amp_batch, flags_batch, thresh_batch)
        
        return flags_batch[0].astype(bool)
