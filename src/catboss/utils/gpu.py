"""
GPU utilities for CATBOSS.

Handles GPU detection, memory management, batch sizing, and CUDA operations.

Author: Arpan Pal
Institution: NCRA-TIFR
"""

import numpy as np
import psutil
from typing import Tuple, Optional, Any, Dict

# Global GPU state
CUDA_CONTEXT = None
GPU_AVAILABLE = False
cuda = None
jit = None
prange = None


def _init_gpu():
    """Initialize GPU state on module load."""
    global CUDA_CONTEXT, GPU_AVAILABLE, cuda, jit, prange
    
    try:
        from numba import cuda as numba_cuda
        from numba import jit as numba_jit
        from numba import prange as numba_prange
        
        cuda = numba_cuda
        jit = numba_jit
        prange = numba_prange
        
        if cuda.is_available():
            try:
                CUDA_CONTEXT = cuda.current_context()
                free, total = CUDA_CONTEXT.get_memory_info()
                GPU_AVAILABLE = True
                print(f"GPU: {total/1e9:.2f} GB VRAM available")
            except Exception as e:
                print(f"✗ GPU context test failed: {e} - using CPU only")
                GPU_AVAILABLE = False
                CUDA_CONTEXT = None
        else:
            print("✗ No GPU detected - will use CPU processing")
            
    except ImportError:
        print("✗ Numba not installed - will use CPU processing")
        # Fallback
        def jit_fallback(*args, **kwargs):
            def decorator(func):
                return func
            if len(args) == 1 and callable(args[0]):
                return args[0]
            return decorator
        
        jit = jit_fallback
        prange = range
        GPU_AVAILABLE = False

# Initialize on import
_init_gpu()


def is_gpu_available() -> bool:
    """Check if GPU is available."""
    return GPU_AVAILABLE


def get_cuda():
    """Get cuda module (or None if unavailable)."""
    return cuda


def get_jit():
    """Get jit decorator."""
    return jit


def get_prange():
    """Get prange function."""
    return prange


def get_cuda_context():
    """Get the cached CUDA context."""
    return CUDA_CONTEXT


def print_gpu_info(logger=None):
    """Print hardware information."""
    def log(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)
    
    log("  HARDWARE INFORMATION")
    
    if not GPU_AVAILABLE:
        log("  Mode: CPU only (no GPU available)")
        # Still show RAM
        try:
            mem = psutil.virtual_memory()
            log(f"  RAM: {mem.total/1e9:.2f} GB total, {mem.available/1e9:.2f} GB available")
        except:
            pass
        return

    try:
        device = cuda.get_current_device()
        log(f"  GPU: {device.name}")
        free, total = CUDA_CONTEXT.get_memory_info()
        log(f"  VRAM: {total/1e9:.2f} GB total, {free/1e9:.2f} GB free")
        mem = psutil.virtual_memory()
        log(f"  RAM:  {mem.total/1e9:.2f} GB total, {mem.available/1e9:.2f} GB free")
    except Exception as e:
        log(f"  Error getting GPU info: {str(e)}")


def get_memory_info(logger=None) -> Tuple[float, float]:
    """
    Get available GPU and system memory.
    
    Returns:
        Tuple of (gpu_free_mem, system_free_mem) in bytes
    """
    gpu_free_mem = 0.0
    
    if GPU_AVAILABLE and CUDA_CONTEXT is not None:
        try:
            free_mem, total_mem = CUDA_CONTEXT.get_memory_info()
            gpu_free_mem = free_mem
        except Exception:
            gpu_free_mem = 0.0

    # System memory
    try:
        mem = psutil.virtual_memory()
        system_free_mem = mem.available
    except Exception:
        system_free_mem = 8e9  # Assume 8GB

    return gpu_free_mem, system_free_mem


def calculate_batch_size(
    n_time: int,
    n_freq: int,
    n_corr: int,
    gpu_free_mem: float,
    system_free_mem: float,
    use_gpu: bool = True,
    logger=None
) -> Tuple[int, Dict[str, Any]]:
    """
    Calculate optimal batch size (number of baselines) based on available memory.
    
    Memory model:
    - GPU arrays: amp (float32), flags (bool→uint8), thresholds (float32)
      Per baseline: n_time × n_freq × (4 + 1 + 4) = 9 bytes per element
      Plus threshold array: n_freq × 4 bytes
    
    - RAM arrays: complex64 data, flags, working copies
      Per baseline: n_time × n_freq × n_corr × 8 bytes × 2 (double buffer for prefetch)
    
    Args:
        n_time: Number of time samples per baseline
        n_freq: Number of frequency channels
        n_corr: Number of correlations
        gpu_free_mem: Free GPU memory in bytes
        system_free_mem: Free system memory in bytes
        use_gpu: Whether GPU will be used
        logger: Optional logger
        
    Returns:
        Tuple of (batch_size, info_dict)
    """
    # GPU memory per baseline (for flagging arrays)
    # amp: float32 (4), flags: uint8 (1), thresh: float32 (4) per freq
    gpu_per_bl = n_time * n_freq * (4 + 1) + n_freq * 4  # ~5 bytes per element + thresh
    gpu_per_bl = int(gpu_per_bl * 1.2)  # 20% overhead for CUDA allocations
    
    # RAM per baseline (for I/O and working data)
    # complex64 data: 8 bytes × n_corr
    # flags: 1 byte × n_corr  
    # Working copy for prefetch: ×2
    ram_per_bl = n_time * n_freq * n_corr * (8 + 1) * 2
    ram_per_bl = int(ram_per_bl * 1.1)  # 10% overhead
    
    # Calculate limits with safety margins
    if use_gpu and gpu_free_mem > 0:
        # GPU mode: limited by VRAM (use 80%)
        gpu_usable = gpu_free_mem * 0.8
        gpu_batch_limit = max(1, int(gpu_usable / gpu_per_bl))
    else:
        gpu_batch_limit = 999999  # No GPU limit
    
    # RAM limit (use 60% to leave room for OS and prefetch)
    ram_usable = system_free_mem * 0.6
    ram_batch_limit = max(1, int(ram_usable / ram_per_bl))
    
    # Take the minimum
    batch_size = min(gpu_batch_limit, ram_batch_limit)
    
    # Apply final safety margin
    batch_size = max(1, int(batch_size * 0.8))
    
    info = {
        'gpu_per_bl_mb': gpu_per_bl / 1e6,
        'ram_per_bl_mb': ram_per_bl / 1e6,
        'gpu_batch_limit': gpu_batch_limit,
        'ram_batch_limit': ram_batch_limit,
        'final_batch_size': batch_size,
        'limiting_factor': 'VRAM' if gpu_batch_limit < ram_batch_limit else 'RAM',
    }
    
    if logger:
        logger.info(f"  Memory per baseline: GPU={gpu_per_bl/1e6:.1f} MB, RAM={ram_per_bl/1e6:.1f} MB")
        logger.info(f"  Batch limits: GPU={gpu_batch_limit}, RAM={ram_batch_limit}")
        logger.info(f"  Selected batch size: {batch_size} baselines (limited by {info['limiting_factor']})")
    
    return batch_size, info


def allocate_gpu_batch_arrays(
    n_baselines: int,
    n_time: int,
    n_freq: int,
    stream=None
) -> Tuple[Any, Any, Any]:
    """
    Allocate contiguous GPU arrays for a batch of baselines.
    
    Args:
        n_baselines: Number of baselines in batch
        n_time: Time samples per baseline
        n_freq: Frequency channels
        stream: Optional CUDA stream
        
    Returns:
        Tuple of (d_amp, d_flags, d_thresh) device arrays
    """
    if not GPU_AVAILABLE or cuda is None:
        raise RuntimeError("GPU not available")
    
    # Allocate on device
    d_amp = cuda.device_array((n_baselines, n_time, n_freq), dtype=np.float32)
    d_flags = cuda.device_array((n_baselines, n_time, n_freq), dtype=np.uint8)
    d_thresh = cuda.device_array((n_baselines, n_freq), dtype=np.float32)
    
    return d_amp, d_flags, d_thresh


def copy_batch_to_gpu(
    amp_list: list,
    flags_list: list,
    thresh_list: list,
    d_amp: Any,
    d_flags: Any,
    d_thresh: Any,
    stream=None
):
    """
    Copy batch of baseline data to pre-allocated GPU arrays.
    
    Args:
        amp_list: List of amplitude arrays (n_time × n_freq each)
        flags_list: List of flag arrays
        thresh_list: List of threshold arrays (n_freq each)
        d_amp, d_flags, d_thresh: Pre-allocated device arrays
        stream: Optional CUDA stream
    """
    if not GPU_AVAILABLE or cuda is None:
        return
    
    n_bl = len(amp_list)
    
    # Stack into contiguous arrays
    amp_batch = np.stack(amp_list, axis=0).astype(np.float32)
    flags_batch = np.stack(flags_list, axis=0).astype(np.uint8)
    thresh_batch = np.stack(thresh_list, axis=0).astype(np.float32)
    
    # Copy to device
    if stream is not None:
        cuda.to_device(amp_batch, to=d_amp, stream=stream)
        cuda.to_device(flags_batch, to=d_flags, stream=stream)
        cuda.to_device(thresh_batch, to=d_thresh, stream=stream)
    else:
        cuda.to_device(amp_batch, to=d_amp)
        cuda.to_device(flags_batch, to=d_flags)
        cuda.to_device(thresh_batch, to=d_thresh)


def copy_batch_from_gpu(
    d_flags: Any,
    flags_list: list,
    stream=None
):
    """
    Copy flag results from GPU back to host arrays.
    
    Args:
        d_flags: Device flag array (n_baselines × n_time × n_freq)
        flags_list: List of host flag arrays to copy into
        stream: Optional CUDA stream
    """
    if not GPU_AVAILABLE or cuda is None:
        return
    
    # Copy to host
    if stream is not None:
        flags_batch = d_flags.copy_to_host(stream=stream)
        stream.synchronize()
    else:
        flags_batch = d_flags.copy_to_host()
    
    # Unpack to individual arrays
    for i, flags in enumerate(flags_list):
        flags[:] = flags_batch[i].astype(bool)


def create_cuda_stream():
    """Create a new CUDA stream if GPU is available."""
    if GPU_AVAILABLE and cuda is not None:
        try:
            return cuda.stream()
        except Exception:
            return None
    return None


def cuda_synchronize():
    """Synchronize CUDA if available."""
    if GPU_AVAILABLE and cuda is not None:
        try:
            cuda.synchronize()
        except Exception:
            pass


def free_gpu_memory():
    """Force GPU memory cleanup by flushing Numba's deallocation queue."""
    if GPU_AVAILABLE and cuda is not None:
        try:
            cuda.synchronize()
            ctx = cuda.current_context()
            # Flush Numba's pending deallocation queue so memory is actually returned
            try:
                ctx.memory_manager.deallocations.clear()
            except AttributeError:
                try:
                    ctx.deallocations.clear()
                except AttributeError:
                    pass
        except Exception:
            pass


def get_optimal_block_size(n_time: int, n_freq: int) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """
    Calculate optimal CUDA block and grid dimensions for 2D data.
    
    Args:
        n_time: Time dimension
        n_freq: Frequency dimension
        
    Returns:
        Tuple of (threads_per_block, blocks_per_grid)
    """
    # Standard 2D block size (16×16 = 256 threads, good for most GPUs)
    threads = (16, 16)
    
    blocks_t = (n_time + threads[0] - 1) // threads[0]
    blocks_f = (n_freq + threads[1] - 1) // threads[1]
    blocks = (blocks_t, blocks_f)
    
    return threads, blocks


def get_optimal_block_size_3d(
    n_baselines: int, 
    n_time: int, 
    n_freq: int
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """
    Calculate optimal CUDA block and grid dimensions for 3D batch processing.
    
    Args:
        n_baselines: Number of baselines in batch
        n_time: Time dimension
        n_freq: Frequency dimension
        
    Returns:
        Tuple of (threads_per_block, blocks_per_grid)
    """
    # For 3D: (baselines, time, freq)
    # Keep baseline dim small, maximize time/freq parallelism
    threads = (1, 16, 16)  # 256 threads per block
    
    blocks_bl = n_baselines
    blocks_t = (n_time + threads[1] - 1) // threads[1]
    blocks_f = (n_freq + threads[2] - 1) // threads[2]
    blocks = (blocks_bl, blocks_t, blocks_f)
    
    return threads, blocks
