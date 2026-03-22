"""
IQR (Inter-Quartile Range) RFI flagging method.

Flags samples outside the range [Q1 - factor*IQR, Q3 + factor*IQR]
where IQR = Q3 - Q1 and default factor = 1.5.

Supports both CPU (parallel) and GPU (batch) execution.

Author: Arpan Pal
Institution: NCRA-TIFR
"""

import numpy as np
from typing import Dict, Any, Optional
from .base import BaseFlaggingMethod
from ...utils.gpu import get_jit, get_prange, is_gpu_available, get_cuda

jit = get_jit()
prange = get_prange()


# ==================== CPU IMPLEMENTATIONS ====================

@jit(nopython=True, parallel=True)
def _iqr_flag_cpu_batch(
    amp: np.ndarray,
    flags: np.ndarray,
    q1: np.ndarray,
    q3: np.ndarray,
    factor: float
):
    """
    IQR flagging for batch of baselines (CPU, parallel over baselines).
    
    Args:
        amp: Amplitude array (n_baselines × n_time × n_freq), float32
        flags: Flag array (n_baselines × n_time × n_freq), uint8, modified in-place
        q1: First quartile per channel (n_baselines × n_freq), float32
        q3: Third quartile per channel (n_baselines × n_freq), float32
        factor: IQR multiplier
    """
    n_bl, n_time, n_freq = amp.shape
    
    for bl in prange(n_bl):
        for f in range(n_freq):
            iqr = q3[bl, f] - q1[bl, f]
            lower = q1[bl, f] - factor * iqr
            upper = q3[bl, f] + factor * iqr
            
            for t in range(n_time):
                if flags[bl, t, f] == 0:  # Only flag unflagged samples
                    val = amp[bl, t, f]
                    if val < lower or val > upper:
                        flags[bl, t, f] = 1


@jit(nopython=True)
def _calculate_quartiles_single(amp: np.ndarray, flags: np.ndarray) -> tuple:
    """
    Calculate Q1, Q3 per channel for a single baseline.
    
    Args:
        amp: Amplitude array (n_time × n_freq)
        flags: Flag array (n_time × n_freq)
        
    Returns:
        Tuple of (q1, q3) arrays, each (n_freq,)
    """
    n_time, n_freq = amp.shape
    q1 = np.zeros(n_freq, dtype=np.float32)
    q3 = np.zeros(n_freq, dtype=np.float32)
    
    for f in range(n_freq):
        # Collect unflagged samples
        valid = []
        for t in range(n_time):
            if not flags[t, f]:
                valid.append(amp[t, f])
        
        if len(valid) >= 4:
            # Sort and get quartiles
            valid_arr = np.array(valid)
            valid_arr.sort()
            n = len(valid_arr)
            q1[f] = valid_arr[n // 4]
            q3[f] = valid_arr[(3 * n) // 4]
        else:
            # Not enough samples, use median ± some range
            if len(valid) > 0:
                valid_arr = np.array(valid)
                med = np.median(valid_arr)
                q1[f] = med * 0.5
                q3[f] = med * 1.5
            else:
                q1[f] = 0.0
                q3[f] = 1e10  # Will flag nothing
    
    return q1, q3


@jit(nopython=True, parallel=True)
def _calculate_quartiles_parallel(
    amp: np.ndarray,
    flags: np.ndarray,
    q1_out: np.ndarray,
    q3_out: np.ndarray,
):
    """
    Numba-parallel quartile calculation. Each baseline runs on its own thread.
    No large intermediate arrays — tiny stack allocation per (baseline, channel).

    Args:
        amp:    (n_bl × n_time × n_freq), float32
        flags:  (n_bl × n_time × n_freq), uint8
        q1_out: (n_bl × n_freq), float32  — pre-allocated, written in-place
        q3_out: (n_bl × n_freq), float32  — pre-allocated (init 1e10), written in-place
    """
    n_bl, n_time, n_freq = amp.shape

    for bl in prange(n_bl):
        for f in range(n_freq):
            # Count unflagged samples
            count = 0
            for t in range(n_time):
                if flags[bl, t, f] == 0:
                    count += 1

            if count == 0:
                # All flagged — leave defaults (q1=0, q3=1e10): flags nothing
                continue

            # Collect unflagged values
            valid = np.empty(count, dtype=np.float32)
            idx = 0
            for t in range(n_time):
                if flags[bl, t, f] == 0:
                    valid[idx] = amp[bl, t, f]
                    idx += 1

            valid.sort()

            if count >= 4:
                q1_out[bl, f] = valid[count // 4]
                q3_out[bl, f] = valid[(3 * count) // 4]
            else:
                # Too few samples — use median ± half range as fallback
                med = valid[count // 2]
                q1_out[bl, f] = med * 0.5
                q3_out[bl, f] = med * 1.5


def calculate_quartiles_batch(
    amp_batch: np.ndarray,
    flags_batch: np.ndarray
) -> tuple:
    """
    Calculate Q1, Q3 for a batch of baselines.

    Parallelised over baselines via Numba prange. No large intermediate
    arrays — each thread works on one baseline at a time.

    Args:
        amp_batch:   (n_baselines × n_time × n_freq), float32
        flags_batch: (n_baselines × n_time × n_freq), uint8

    Returns:
        Tuple of (q1_batch, q3_batch), each (n_baselines × n_freq)
    """
    n_bl, _, n_freq = amp_batch.shape
    q1_batch = np.zeros((n_bl, n_freq), dtype=np.float32)
    q3_batch = np.full((n_bl, n_freq), 1e10, dtype=np.float32)

    _calculate_quartiles_parallel(
        np.ascontiguousarray(amp_batch, dtype=np.float32),
        np.ascontiguousarray(flags_batch, dtype=np.uint8),
        q1_batch, q3_batch,
    )
    return q1_batch, q3_batch


# ==================== GPU IMPLEMENTATIONS ====================

_gpu_kernel_defined = False
_iqr_gpu_kernel = None

if is_gpu_available():
    cuda = get_cuda()
    
    if cuda is not None:
        @cuda.jit
        def _iqr_flag_gpu_3d(amp, flags, q1, q3, factor):
            """
            CUDA kernel for IQR flagging (3D batch).
            
            Each thread handles one (baseline, time, freq) position.
            """
            bl, t, f = cuda.grid(3)
            
            n_bl, n_time, n_freq = amp.shape
            
            if bl < n_bl and t < n_time and f < n_freq:
                if flags[bl, t, f] == 0:  # Only check unflagged
                    iqr = q3[bl, f] - q1[bl, f]
                    lower = q1[bl, f] - factor * iqr
                    upper = q3[bl, f] + factor * iqr
                    
                    val = amp[bl, t, f]
                    if val < lower or val > upper:
                        flags[bl, t, f] = 1
        
        _gpu_kernel_defined = True
        _iqr_gpu_kernel = _iqr_flag_gpu_3d


class IQRMethod(BaseFlaggingMethod):
    """
    IQR-based flagging method.
    
    Flags samples outside [Q1 - factor*IQR, Q3 + factor*IQR].
    Supports both CPU and GPU batch processing.
    """
    
    name = "iqr"
    
    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        return {
            'iqr_factor': 1.5,
            'use_gpu': True,
        }
    
    @classmethod
    def get_param_help(cls) -> Dict[str, str]:
        return {
            'iqr_factor': 'IQR multiplier for outlier detection (default: 1.5)',
            'use_gpu': 'Use GPU if available (default: True)',
        }
    
    def flag_batch(
        self,
        amp_batch: np.ndarray,
        flags_batch: np.ndarray,
        thresholds_batch: np.ndarray = None,
        gpu_arrays=None
    ) -> np.ndarray:
        """
        Apply IQR flagging to a batch of baselines.
        
        Args:
            amp_batch: Amplitude array (n_baselines × n_time × n_freq), float32
            flags_batch: Existing flags (n_baselines × n_time × n_freq), uint8
            thresholds_batch: Ignored (quartiles calculated internally)
            
        Returns:
            Updated flags array (modified in-place and returned)
        """
        factor = self.params.get('iqr_factor', 1.5)
        use_gpu = self.params.get('use_gpu', True) and is_gpu_available() and _gpu_kernel_defined
        
        # Calculate quartiles (always on CPU - involves sorting)
        q1_batch, q3_batch = calculate_quartiles_batch(amp_batch, flags_batch)
        
        if use_gpu:
            self._flag_batch_gpu(amp_batch, flags_batch, q1_batch, q3_batch, factor, gpu_arrays=gpu_arrays)
        else:
            self._flag_batch_cpu(amp_batch, flags_batch, q1_batch, q3_batch, factor)
        
        return flags_batch
    
    def _flag_batch_cpu(
        self,
        amp: np.ndarray,
        flags: np.ndarray,
        q1: np.ndarray,
        q3: np.ndarray,
        factor: float
    ):
        """CPU batch flagging with Numba parallel."""
        amp = amp.astype(np.float32)
        flags_work = flags.astype(np.uint8)
        q1 = q1.astype(np.float32)
        q3 = q3.astype(np.float32)
        
        _iqr_flag_cpu_batch(amp, flags_work, q1, q3, factor)
        
        # Copy back
        flags[:] = flags_work
        
        self.log(f"    IQR CPU: {amp.shape[0]} baselines, factor={factor}")
    
    def _flag_batch_gpu(
        self,
        amp: np.ndarray,
        flags: np.ndarray,
        q1: np.ndarray,
        q3: np.ndarray,
        factor: float,
        gpu_arrays=None
    ):
        """GPU batch flagging."""
        cuda = get_cuda()

        if cuda is None:
            self._flag_batch_cpu(amp, flags, q1, q3, factor)
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
            d_q1 = cuda.to_device(np.ascontiguousarray(q1, dtype=np.float32))
            d_q3 = cuda.to_device(np.ascontiguousarray(q3, dtype=np.float32))
            
            # Grid configuration
            threads = (1, 16, 16)
            blocks = (
                n_bl,
                (n_time + threads[1] - 1) // threads[1],
                (n_freq + threads[2] - 1) // threads[2]
            )
            
            # Launch kernel
            _iqr_gpu_kernel[blocks, threads](d_amp, d_flags, d_q1, d_q3, factor)
            
            # Copy back
            cuda.synchronize()
            d_flags.copy_to_host(flags)
            
            self.log(f"    IQR GPU: {n_bl} baselines, factor={factor}")
            
        except Exception as e:
            self.log(f"    GPU failed ({e}), falling back to CPU", level='warning')
            self._flag_batch_cpu(amp, flags, q1, q3, factor)
    
    # Legacy single-baseline interface
    def flag(
        self,
        amp: np.ndarray,
        flags: np.ndarray,
        thresholds: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Apply IQR flagging to single baseline.
        
        Args:
            amp: Amplitude array (n_time × n_freq)
            flags: Existing flags
            thresholds: Ignored
            
        Returns:
            Combined flag array
        """
        # Reshape to batch format
        amp_batch = amp[np.newaxis, :, :].astype(np.float32)
        flags_batch = flags[np.newaxis, :, :].astype(np.uint8)
        
        self.flag_batch(amp_batch, flags_batch)
        
        return flags_batch[0].astype(bool)
