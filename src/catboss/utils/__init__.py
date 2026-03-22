"""
Utility functions for CATBOSS.
"""

from .gpu import (
    is_gpu_available,
    get_cuda,
    get_jit,
    get_prange,
    get_cuda_context,
    print_gpu_info,
    get_memory_info,
    calculate_batch_size,
    allocate_gpu_batch_arrays,
    copy_batch_to_gpu,
    copy_batch_from_gpu,
    create_cuda_stream,
    cuda_synchronize,
    free_gpu_memory,
    get_optimal_block_size,
    get_optimal_block_size_3d,
    GPU_AVAILABLE,
    CUDA_CONTEXT,
)

__all__ = [
    'is_gpu_available',
    'get_cuda',
    'get_jit',
    'get_prange',
    'get_cuda_context',
    'print_gpu_info',
    'get_memory_info',
    'calculate_batch_size',
    'allocate_gpu_batch_arrays',
    'copy_batch_to_gpu',
    'copy_batch_from_gpu',
    'create_cuda_stream',
    'cuda_synchronize',
    'free_gpu_memory',
    'get_optimal_block_size',
    'get_optimal_block_size_3d',
    'GPU_AVAILABLE',
    'CUDA_CONTEXT',
]
