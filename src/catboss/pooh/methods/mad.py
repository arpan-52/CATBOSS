"""
MAD (Median Absolute Deviation) RFI flagging method.

Flags samples where |value - median| > sigma * MAD * 1.4826
where MAD = median(|values - median|) and 1.4826 is the consistency
constant for Gaussian distributions.

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

# Consistency constant for Gaussian
MAD_SCALE = 1.4826


# ==================== CPU IMPLEMENTATIONS ====================

@jit(nopython=True, parallel=True)
def _mad_flag_cpu_batch(
    amp: np.ndarray,
    flags: np.ndarray,
    medians: np.ndarray,
    mads: np.ndarray,
    sigma: float
):
    """
    MAD flagging for batch of baselines (CPU, parallel over baselines).
    
    Args:
        amp: Amplitude array (n_baselines × n_time × n_freq), float32
        flags: Flag array (n_baselines × n_time × n_freq), uint8, modified in-place
        medians: Median per channel (n_baselines × n_freq), float32
        mads: MAD per channel (n_baselines × n_freq), float32
        sigma: Sigma threshold
    """
    n_bl, n_time, n_freq = amp.shape
    
    for bl in prange(n_bl):
        for f in range(n_freq):
            med = medians[bl, f]
            threshold = med + sigma * mads[bl, f] * MAD_SCALE
            
            for t in range(n_time):
                if flags[bl, t, f] == 0:  # Only flag unflagged samples
                    if amp[bl, t, f] > threshold:
                        flags[bl, t, f] = 1


@jit(nopython=True, parallel=True)
def _calculate_mad_parallel(
    amp: np.ndarray,
    flags: np.ndarray,
    medians_out: np.ndarray,
    mads_out: np.ndarray,
):
    """
    Numba-parallel median and MAD calculation. Each baseline runs on its own thread.
    No large intermediate arrays — tiny stack allocation per (baseline, channel).

    Args:
        amp:         (n_bl × n_time × n_freq), float32
        flags:       (n_bl × n_time × n_freq), uint8
        medians_out: (n_bl × n_freq), float32  — written in-place
        mads_out:    (n_bl × n_freq), float32  — pre-allocated (init 1e10), written in-place
    """
    n_bl, n_time, n_freq = amp.shape

    for bl in prange(n_bl):
        for f in range(n_freq):
            count = 0
            for t in range(n_time):
                if flags[bl, t, f] == 0:
                    count += 1

            if count == 0:
                # All flagged — leave defaults: flags nothing
                continue

            valid = np.empty(count, dtype=np.float32)
            idx = 0
            for t in range(n_time):
                if flags[bl, t, f] == 0:
                    valid[idx] = amp[bl, t, f]
                    idx += 1

            valid.sort()

            # Median
            if count % 2 == 0:
                med = (valid[count // 2 - 1] + valid[count // 2]) * 0.5
            else:
                med = valid[count // 2]
            medians_out[bl, f] = med

            if count < 3:
                mads_out[bl, f] = med * 0.1
                continue

            # MAD: median of absolute deviations
            devs = np.empty(count, dtype=np.float32)
            for k in range(count):
                devs[k] = abs(valid[k] - med)
            devs.sort()

            if count % 2 == 0:
                mad = (devs[count // 2 - 1] + devs[count // 2]) * 0.5
            else:
                mad = devs[count // 2]

            mads_out[bl, f] = mad if mad > 1e-10 else 1e-10


def calculate_mad_batch(
    amp_batch: np.ndarray,
    flags_batch: np.ndarray
) -> tuple:
    """
    Calculate median and MAD for a batch of baselines.

    Parallelised over baselines via Numba prange. No large intermediate
    arrays — each thread works on one baseline at a time.

    Args:
        amp_batch:   (n_baselines × n_time × n_freq), float32
        flags_batch: (n_baselines × n_time × n_freq), uint8

    Returns:
        Tuple of (medians, mads), each (n_baselines × n_freq)
    """
    n_bl, _, n_freq = amp_batch.shape
    medians = np.zeros((n_bl, n_freq), dtype=np.float32)
    mads = np.full((n_bl, n_freq), 1e10, dtype=np.float32)

    _calculate_mad_parallel(
        np.ascontiguousarray(amp_batch, dtype=np.float32),
        np.ascontiguousarray(flags_batch, dtype=np.uint8),
        medians, mads,
    )
    return medians, mads


# ==================== GPU IMPLEMENTATIONS ====================

_gpu_kernel_defined = False
_mad_gpu_kernel = None

if is_gpu_available():
    cuda = get_cuda()
    
    if cuda is not None:
        @cuda.jit
        def _mad_flag_gpu_3d(amp, flags, medians, mads, sigma, mad_scale):
            """
            CUDA kernel for MAD flagging (3D batch).
            
            Each thread handles one (baseline, time, freq) position.
            """
            bl, t, f = cuda.grid(3)
            
            n_bl, n_time, n_freq = amp.shape
            
            if bl < n_bl and t < n_time and f < n_freq:
                if flags[bl, t, f] == 0:  # Only check unflagged
                    med = medians[bl, f]
                    threshold = med + sigma * mads[bl, f] * mad_scale
                    
                    if amp[bl, t, f] > threshold:
                        flags[bl, t, f] = 1
        
        _gpu_kernel_defined = True
        _mad_gpu_kernel = _mad_flag_gpu_3d


class MADMethod(BaseFlaggingMethod):
    """
    MAD-based flagging method.
    
    Flags samples where value > median + sigma * MAD * 1.4826.
    Supports both CPU and GPU batch processing.
    """
    
    name = "mad"
    
    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        return {
            'mad_sigma': 5.0,
            'use_gpu': True,
        }
    
    @classmethod
    def get_param_help(cls) -> Dict[str, str]:
        return {
            'mad_sigma': 'Sigma threshold for MAD flagging (default: 5.0)',
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
        Apply MAD flagging to a batch of baselines.
        
        Args:
            amp_batch: Amplitude array (n_baselines × n_time × n_freq), float32
            flags_batch: Existing flags (n_baselines × n_time × n_freq), uint8
            thresholds_batch: Ignored (median/MAD calculated internally)
            
        Returns:
            Updated flags array (modified in-place and returned)
        """
        sigma = self.params.get('mad_sigma', 5.0)
        use_gpu = self.params.get('use_gpu', True) and is_gpu_available() and _gpu_kernel_defined
        
        # Calculate median and MAD (always on CPU - involves sorting)
        medians, mads = calculate_mad_batch(amp_batch, flags_batch)
        
        if use_gpu:
            self._flag_batch_gpu(amp_batch, flags_batch, medians, mads, sigma, gpu_arrays=gpu_arrays)
        else:
            self._flag_batch_cpu(amp_batch, flags_batch, medians, mads, sigma)
        
        return flags_batch
    
    def _flag_batch_cpu(
        self,
        amp: np.ndarray,
        flags: np.ndarray,
        medians: np.ndarray,
        mads: np.ndarray,
        sigma: float
    ):
        """CPU batch flagging with Numba parallel."""
        amp = amp.astype(np.float32)
        flags_work = flags.astype(np.uint8)
        medians = medians.astype(np.float32)
        mads = mads.astype(np.float32)
        
        _mad_flag_cpu_batch(amp, flags_work, medians, mads, sigma)
        
        # Copy back
        flags[:] = flags_work
        
        self.log(f"    MAD CPU: {amp.shape[0]} baselines, sigma={sigma}")
    
    def _flag_batch_gpu(
        self,
        amp: np.ndarray,
        flags: np.ndarray,
        medians: np.ndarray,
        mads: np.ndarray,
        sigma: float,
        gpu_arrays=None
    ):
        """GPU batch flagging."""
        cuda = get_cuda()

        if cuda is None:
            self._flag_batch_cpu(amp, flags, medians, mads, sigma)
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
            d_medians = cuda.to_device(np.ascontiguousarray(medians, dtype=np.float32))
            d_mads = cuda.to_device(np.ascontiguousarray(mads, dtype=np.float32))
            
            # Grid configuration
            threads = (1, 16, 16)
            blocks = (
                n_bl,
                (n_time + threads[1] - 1) // threads[1],
                (n_freq + threads[2] - 1) // threads[2]
            )
            
            # Launch kernel
            _mad_gpu_kernel[blocks, threads](
                d_amp, d_flags, d_medians, d_mads, sigma, MAD_SCALE
            )
            
            # Copy back
            cuda.synchronize()
            d_flags.copy_to_host(flags)
            
            self.log(f"    MAD GPU: {n_bl} baselines, sigma={sigma}")
            
        except Exception as e:
            self.log(f"    GPU failed ({e}), falling back to CPU", level='warning')
            self._flag_batch_cpu(amp, flags, medians, mads, sigma)
    
    # Legacy single-baseline interface
    def flag(
        self,
        amp: np.ndarray,
        flags: np.ndarray,
        thresholds: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Apply MAD flagging to single baseline.
        
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
