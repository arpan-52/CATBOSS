"""
POOH - Parallelized Optimized Outlier Hunter

Main processing engine for dynamic spectra RFI flagging.

Features:
- Process ALL correlations per baseline together
- Batch GPU processing (all baselines × all corrs shipped together)
- Async prefetch (load next batch while GPU processes current)
- Multi-pass processing with flag accumulation
- Memory-aware batch sizing

Author: Arpan Pal
Institution: NCRA-TIFR
"""

import numpy as np
import time
import gc
import os
from typing import Dict, Any, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor

from ..io import (
    get_ms_info, get_unique_baselines, get_frequencies,
    read_baseline_data, parse_selection, parse_baseline_selection,
    write_field_flags, print_ms_summary,
)
from ..utils import (
    get_memory_info, calculate_batch_size, print_gpu_info,
    is_gpu_available, cuda_synchronize, free_gpu_memory,
)
from ..utils.gpu import get_cuda
from .bandpass import normalize_bandpass
from .thresholds import calculate_robust_thresholds
from .methods import get_method


def prepare_baseline_all_corrs(
    baseline_data: Dict[Tuple[int, int], Dict],
    corr_indices: List[int],
    normalize: bool,
    poly_order: int,
    deviation_threshold: float,
    sigma: float,
    logger=None
) -> Dict[str, Any]:
    """
    Prepare a batch of baselines with ALL correlations for GPU processing.
    
    Each correlation is processed independently but kept together per baseline.
    
    Args:
        baseline_data: Dict mapping baseline -> {'data': complex, 'flags': bool}
        corr_indices: Which correlations to process
        normalize: Whether to apply bandpass normalization
        poly_order: Polynomial order for bandpass
        deviation_threshold: Bad channel detection threshold
        sigma: Base sigma for threshold calculation
        logger: Optional logger
        
    Returns:
        Dict with:
        - amp_batch: (n_bl × n_corr × n_time × n_freq), float32
        - flags_batch: (n_bl × n_corr × n_time × n_freq), uint8
        - existing_flags_batch: same shape, original flags
        - thresh_batch: (n_bl × n_corr × n_freq), float32
        - baselines: List of baseline tuples
        - n_time, n_freq, n_corr: dimensions
        - total_bad_channels: int
    """
    baselines = list(baseline_data.keys())
    n_bl = len(baselines)
    
    if n_bl == 0:
        return None
    
    # Get dimensions
    first_bl = baselines[0]
    first_data = baseline_data[first_bl]['data']
    n_time, n_freq, _ = first_data.shape
    n_corr = len(corr_indices)
    
    # Output arrays: (n_bl, n_corr, n_time, n_freq)
    amp_batch = np.zeros((n_bl, n_corr, n_time, n_freq), dtype=np.float32)
    flags_batch = np.zeros((n_bl, n_corr, n_time, n_freq), dtype=np.uint8)
    existing_flags_batch = np.zeros((n_bl, n_corr, n_time, n_freq), dtype=np.uint8)
    thresh_batch = np.zeros((n_bl, n_corr, n_freq), dtype=np.float32)
    
    total_bad_channels = 0
    
    for bl_idx, bl in enumerate(baselines):
        bl_data = baseline_data[bl]['data']
        bl_flags = baseline_data[bl]['flags']
        
        for c_idx, corr_idx in enumerate(corr_indices):
            data = bl_data[:, :, corr_idx]
            flags = bl_flags[:, :, corr_idx].astype(bool)
            
            # Store existing flags BEFORE any processing
            existing_flags_batch[bl_idx, c_idx] = flags.astype(np.uint8)
            
            # Calculate amplitude
            amp = np.abs(data).astype(np.float32)
            
            # Bandpass normalization (respects existing flags)
            if normalize:
                amp, flags, _, _, n_bad = normalize_bandpass(
                    amp, flags,
                    poly_order=poly_order,
                    deviation_threshold=deviation_threshold,
                    logger=None
                )
                total_bad_channels += n_bad
            
            # Calculate thresholds (respects flags)
            thresh = calculate_robust_thresholds(amp, flags, sigma)
            
            amp_batch[bl_idx, c_idx] = amp
            flags_batch[bl_idx, c_idx] = flags.astype(np.uint8)
            thresh_batch[bl_idx, c_idx] = thresh
    
    return {
        'amp_batch': amp_batch,
        'flags_batch': flags_batch,
        'existing_flags_batch': existing_flags_batch,
        'thresh_batch': thresh_batch,
        'baselines': baselines,
        'n_time': n_time,
        'n_freq': n_freq,
        'n_corr': n_corr,
        'corr_indices': corr_indices,
        'total_bad_channels': total_bad_channels,
    }


def process_batch_all_corrs(
    prep_data: Dict[str, Any],
    passes_config: List[Dict],
    time_chunks: List[Tuple[int, int]],
    freq_chunks: List[Tuple[int, int]],
    logger=None,
    gpu_arrays=None
) -> np.ndarray:
    """
    Process batch with all correlations on GPU.
    
    Reshapes (n_bl, n_corr, n_time, n_freq) -> (n_bl * n_corr, n_time, n_freq)
    for efficient GPU processing, then reshapes back.
    
    Args:
        prep_data: Dict from prepare_baseline_all_corrs
        passes_config: List of pass configurations
        time_chunks, freq_chunks: Chunking info
        logger: Optional logger
        
    Returns:
        Updated flags_batch (n_bl × n_corr × n_time × n_freq)
    """
    amp_batch = prep_data['amp_batch']
    flags_batch = prep_data['flags_batch']
    thresh_batch = prep_data['thresh_batch']

    n_bl, n_corr, n_time, n_freq = amp_batch.shape

    # Reshape to flat: (n_bl * n_corr, n_time, n_freq)
    # amp_flat is static — never changes across passes
    # flags_flat accumulates across passes
    amp_flat = amp_batch.reshape(n_bl * n_corr, n_time, n_freq)
    flags_flat = flags_batch.reshape(n_bl * n_corr, n_time, n_freq)
    thresh_flat = thresh_batch.reshape(n_bl * n_corr, n_freq)

    # Pre-allocate contiguous chunk buffers (reused across all passes/chunks)
    # Find max chunk dimensions to allocate once
    max_t_chunk = max((t_end - t_start) for t_start, t_end in time_chunks)
    max_f_chunk = max((f_end - f_start) for f_start, f_end in freq_chunks)
    n_flat = n_bl * n_corr
    amp_chunk_buf = np.empty((n_flat, max_t_chunk, max_f_chunk), dtype=np.float32)
    flags_chunk_buf = np.empty((n_flat, max_t_chunk, max_f_chunk), dtype=np.uint8)
    thresh_chunk_buf = np.empty((n_flat, max_f_chunk), dtype=np.float32)

    # Passes are the OUTER loop so each pass sees flags from the full previous pass
    for pass_idx, pass_config in enumerate(passes_config):
        method_name = pass_config.get('method', 'sumthreshold')
        method = get_method(method_name, pass_config, logger=logger)

        # timebin/freqbin define local stat regions — iterate over them
        for t_start, t_end in time_chunks:
            t_len = t_end - t_start
            for f_start, f_end in freq_chunks:
                f_len = f_end - f_start

                # Copy into pre-allocated contiguous buffers (avoids allocation)
                amp_chunk = amp_chunk_buf[:, :t_len, :f_len]
                flags_chunk = flags_chunk_buf[:, :t_len, :f_len]
                thresh_chunk = thresh_chunk_buf[:, :f_len]

                amp_chunk[:] = amp_flat[:, t_start:t_end, f_start:f_end]
                flags_chunk[:] = flags_flat[:, t_start:t_end, f_start:f_end]
                thresh_chunk[:] = thresh_flat[:, f_start:f_end]

                method.flag_batch(amp_chunk, flags_chunk, thresh_chunk, gpu_arrays=gpu_arrays)

                # Write updated flags back — amp unchanged
                flags_flat[:, t_start:t_end, f_start:f_end] = flags_chunk

    return flags_flat.reshape(n_bl, n_corr, n_time, n_freq)


def calculate_chunks(
    n_time: int,
    n_freq: int,
    timebin: Optional[int],
    freqbin: Optional[int]
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Calculate chunk boundaries."""
    if timebin is None or timebin <= 0 or timebin >= n_time:
        time_chunks = [(0, n_time)]
    else:
        time_chunks = [(t, min(t + timebin, n_time)) for t in range(0, n_time, timebin)]
    
    if freqbin is None or freqbin <= 0 or freqbin >= n_freq:
        freq_chunks = [(0, n_freq)]
    else:
        freq_chunks = [(f, min(f + freqbin, n_freq)) for f in range(0, n_freq, freqbin)]
    
    return time_chunks, freq_chunks


def calculate_batch_size_all_corrs(
    n_time: int,
    n_freq: int,
    n_corr: int,
    gpu_free_mem: float,
    sys_free_mem: float,
    use_gpu: bool,
    logger=None
) -> int:
    """
    Calculate batch size (number of baselines) considering all correlations.
    
    Memory per baseline (all corrs):
        GPU: n_time × n_freq × n_corr × 9 bytes (amp f32 + flags u8 + thresh f32)
        RAM: n_time × n_freq × n_corr × 18 bytes (complex64 + flags + working) × 2 prefetch
    """
    # GPU: amp(4) + flags(1) + thresh(4/n_time) ≈ 5.5 bytes per element
    gpu_per_bl = n_time * n_freq * n_corr * 6  # bytes
    gpu_per_bl = int(gpu_per_bl * 1.3)  # overhead
    
    # RAM: complex(8) + flags(1) × 2 for prefetch
    ram_per_bl = n_time * n_freq * n_corr * 18
    ram_per_bl = int(ram_per_bl * 1.2)
    
    if use_gpu and gpu_free_mem > 0:
        gpu_limit = max(1, int(gpu_free_mem * 0.85 / gpu_per_bl))
    else:
        gpu_limit = 999999

    ram_limit = max(1, int(sys_free_mem * 0.6 / ram_per_bl))

    batch_size = max(1, min(gpu_limit, ram_limit))
    
    if logger:
        logger.info(f"  Memory per baseline (all {n_corr} corrs): "
                   f"GPU={gpu_per_bl/1e6:.1f}MB, RAM={ram_per_bl/1e6:.1f}MB")
        logger.info(f"  Batch limits: GPU={gpu_limit}, RAM={ram_limit}")
        logger.info(f"  Batch size: {batch_size} baselines")
    
    return batch_size


def load_batch_async(
    ms_file: str,
    field_id: int,
    baselines: List[Tuple[int, int]],
    datacolumn: str,
    chunk_size: int,
    logger=None
) -> Dict:
    """Load batch of baselines from MS (for async prefetch)."""
    return read_baseline_data(
        ms_file, field_id, baselines,
        datacolumn=datacolumn,
        spw=None,
        chunk_size=chunk_size,
        logger=None
    )


def hunt_ms(ms_file: str, options: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main POOH entry point - process entire MS file.
    
    Processes ALL correlations independently per baseline.
    
    Args:
        ms_file: Path to Measurement Set
        options: Processing options
        
    Returns:
        Results dictionary with statistics
    """
    logger = options.get('logger')
    total_start = time.time()
    
    # Get MS info
    if logger:
        logger.info("  READING MS INFORMATION")
    
    info = get_ms_info(ms_file, logger=logger)
    print_ms_summary(info, logger=logger)
    print_gpu_info(logger=logger)
    
    # Parse selections
    field_ids = parse_selection(options.get('field'), list(range(info['n_fields'])))
    if field_ids is None:
        field_ids = list(range(info['n_fields']))
    
    spw_ids = parse_selection(options.get('spw'), list(range(info['n_spw'])))
    if spw_ids is None:
        spw_ids = list(range(info['n_spw']))
    
    corr_indices = parse_selection(options.get('corr'), list(range(info['n_corr'])))
    if corr_indices is None:
        corr_indices = list(range(info['n_corr']))
    
    baseline_filter = parse_baseline_selection(
        options.get('baseline'),
        info['valid_antennas']
    )
    exclude_autocorr = options.get('exclude_autocorr', False)
    
    # Build passes config
    passes = options.get('passes_config')
    if passes is None:
        n_passes = options.get('passes', 1)
        method = options.get('method', 'sumthreshold')
        
        pass_config = {'method': method}
        if method == 'sumthreshold':
            pass_config['sigma'] = options.get('sigma', 6.0)
            pass_config['rho'] = options.get('rho', 1.5)
            combi = options.get('combinations', '1,2,4,8,16,32,64')
            pass_config['combinations'] = [int(x) for x in combi.split(',')] if isinstance(combi, str) else combi
        elif method == 'iqr':
            pass_config['iqr_factor'] = options.get('iqr_factor', 1.5)
        elif method == 'mad':
            pass_config['mad_sigma'] = options.get('mad_sigma', 5.0)
        
        pass_config['use_gpu'] = options.get('mode', 'auto') != 'cpu'
        passes = [pass_config] * n_passes
    
    # Processing options
    normalize = not options.get('no_bandpass', False)
    poly_order = options.get('poly_order', 5)
    deviation_threshold = options.get('deviation_threshold', 5.0)
    sigma = passes[0].get('sigma', 6.0) if passes else 6.0
    
    timebin = options.get('timebin')
    freqbin = options.get('freqbin')
    
    # Memory and batch sizing
    use_gpu = options.get('mode', 'auto') != 'cpu' and is_gpu_available()
    gpu_mem, sys_mem = get_memory_info()

    # Initial conservative estimate — will be recalculated per field from real data
    n_ant = info['n_antennas']
    n_baselines_est = n_ant * (n_ant - 1) // 2
    sample_n_time = max(100, info['n_rows'] // max(1, n_baselines_est * info['n_fields']))
    batch_size = calculate_batch_size_all_corrs(
        sample_n_time, info['n_chan'], len(corr_indices),
        gpu_mem, sys_mem, use_gpu, logger
    )
    
    # Output options
    dry_run = options.get('dry_run', False)
    apply_flags = options.get('apply_flags', False) and not dry_run
    
    # Plotting options
    make_plots = options.get('plots', False)
    plot_dir = options.get('plot_dir', 'catboss_plots')
    
    if logger:
        logger.info("  PROCESSING CONFIGURATION")
        logger.info(f"  Fields: {[info['field_names'][i] for i in field_ids]}")
        logger.info(f"  Correlations: {[info['corr_labels'][i] for i in corr_indices]} (all processed independently)")
        logger.info(f"  Passes: {len(passes)}")
        for i, p in enumerate(passes):
            logger.info(f"    Pass {i+1}: {p.get('method')} - {p}")
        logger.info(f"  Batch size: {batch_size} baselines × {len(corr_indices)} corrs")
        logger.info(f"  Mode: {'GPU' if use_gpu else 'CPU'}")
        logger.info(f"  Bandpass normalization: {normalize}")
        logger.info(f"  Apply flags: {apply_flags}")
        if make_plots:
            logger.info(f"  Plots: {plot_dir}")
    
    # Import plotting if needed
    if make_plots:
        os.makedirs(plot_dir, exist_ok=True)
        try:
            from ..plotting import create_field_viewer, create_master_index
        except ImportError as e:
            if logger:
                logger.warning(f"Plotting not available ({e})")
            make_plots = False
    
    # Statistics
    stats = {
        'baselines_processed': 0,
        'baselines_skipped': 0,
        'total_visibilities': 0,
        'existing_flags': 0,
        'new_flags': 0,
    }
    
    field_viewers = {}
    executor = ThreadPoolExecutor(max_workers=1)
    
    # Process each field
    for field_id in field_ids:
        field_name = info['field_names'][field_id] if field_id < len(info['field_names']) else ""
        
        if logger:
            logger.info(f"\n{'='*60}")
            logger.info(f"FIELD {field_id}: {field_name}")
            logger.info(f"{'='*60}")
        
        # Get baselines
        baselines = get_unique_baselines(
            ms_file, field_id,
            exclude_autocorr=exclude_autocorr,
            valid_antennas=info['valid_antennas'],
            logger=logger
        )
        
        if baseline_filter:
            baselines = [bl for bl in baselines if bl in baseline_filter]
        
        if not baselines:
            continue
        
        if logger:
            logger.info(f"  Baselines: {len(baselines)}")

        # Probe one baseline to get real n_time, then recalculate batch size
        real_n_time = None
        if baselines:
            try:
                probe = read_baseline_data(
                    ms_file, field_id, baselines[:1],
                    datacolumn=options.get('datacolumn', 'DATA'),
                    spw=None, chunk_size=50000, logger=None
                )
                if probe:
                    probe_key = list(probe.keys())[0]
                    real_n_time = probe[probe_key]['data'].shape[0]
                    del probe
                    gpu_mem_now, sys_mem_now = get_memory_info()
                    batch_size = calculate_batch_size_all_corrs(
                        real_n_time, info['n_chan'], len(corr_indices),
                        gpu_mem_now, sys_mem_now, use_gpu, logger
                    )
                    if logger:
                        logger.info(f"  Real n_time={real_n_time}, batch_size={batch_size} baselines")
            except Exception as e:
                if logger:
                    logger.warning(f"  Probe read failed ({e}), using estimated batch_size={batch_size}")

        # Pre-allocate GPU arrays once for this field — reused across all batches
        gpu_arrays = None
        if use_gpu and real_n_time is not None:
            _cuda = get_cuda()
            if _cuda is not None:
                try:
                    n_flat = batch_size * len(corr_indices)
                    d_amp_pre = _cuda.device_array(
                        (n_flat, real_n_time, info['n_chan']), dtype=np.float32
                    )
                    d_flags_pre = _cuda.device_array(
                        (n_flat, real_n_time, info['n_chan']), dtype=np.uint8
                    )
                    gpu_arrays = {'d_amp': d_amp_pre, 'd_flags': d_flags_pre}
                    mem_mb = n_flat * real_n_time * info['n_chan'] * 5 / 1e6
                    if logger:
                        logger.info(f"  GPU pre-allocated: {mem_mb:.0f} MB ({n_flat} flat baselines)")
                except Exception as e:
                    if logger:
                        logger.warning(f"  GPU pre-allocation failed ({e}), will allocate per batch")
                    gpu_arrays = None

        # Split into batches
        batches = [baselines[i:i + batch_size] for i in range(0, len(baselines), batch_size)]
        
        if logger:
            logger.info(f"  Batches: {len(batches)}")
        
        # Storage
        field_flags = {}
        plot_data = []  # For plotting
        
        # Prefetch first batch
        prefetch_future = executor.submit(
            load_batch_async, ms_file, field_id, batches[0],
            options.get('datacolumn', 'DATA'),
            options.get('chunk_size', 200000), None
        )
        
        for batch_idx, batch_baselines in enumerate(batches):
            if logger:
                logger.info(f"\n  Batch {batch_idx + 1}/{len(batches)}: {len(batch_baselines)} baselines")
            
            # Get current batch data
            load_start = time.time()
            baseline_data = prefetch_future.result()
            load_time = time.time() - load_start
            
            if logger:
                logger.info(f"    Load: {load_time:.2f}s")
            
            # Start prefetch of next batch
            if batch_idx + 1 < len(batches):
                prefetch_future = executor.submit(
                    load_batch_async, ms_file, field_id, batches[batch_idx + 1],
                    options.get('datacolumn', 'DATA'),
                    options.get('chunk_size', 200000), None
                )
            
            # Prepare all correlations
            prep_start = time.time()
            prep_data = prepare_baseline_all_corrs(
                baseline_data, corr_indices,
                normalize=normalize,
                poly_order=poly_order,
                deviation_threshold=deviation_threshold,
                sigma=sigma,
                logger=logger
            )
            prep_time = time.time() - prep_start
            
            if prep_data is None:
                continue
            
            n_time = prep_data['n_time']
            n_freq = prep_data['n_freq']
            n_corr = prep_data['n_corr']
            bl_order = prep_data['baselines']
            
            if logger:
                logger.info(f"    Prep: {prep_time:.2f}s")
                if normalize and prep_data['total_bad_channels'] > 0:
                    avg_bad = prep_data['total_bad_channels'] / (len(bl_order) * n_corr)
                    logger.info(f"    Bandpass: {prep_data['total_bad_channels']} bad channels ({avg_bad:.0f} avg/baseline/corr)")
                logger.info(f"    Shape: {len(bl_order)} baselines × {n_corr} corrs × {n_time} time × {n_freq} freq")
            
            # Count existing
            existing_count = int(np.sum(prep_data['flags_batch'] > 0))
            
            # Calculate chunks
            time_chunks, freq_chunks = calculate_chunks(n_time, n_freq, timebin, freqbin)
            
            # Process
            proc_start = time.time()
            flags_result = process_batch_all_corrs(
                prep_data, passes, time_chunks, freq_chunks, logger, gpu_arrays=gpu_arrays
            )
            proc_time = time.time() - proc_start
            
            # Count new
            new_count = int(np.sum(flags_result > 0)) - existing_count
            total_vis = flags_result.size
            
            if logger:
                logger.info(f"    Process: {proc_time:.2f}s")
                logger.info(f"    New flags: {new_count:,} ({100*new_count/max(1, total_vis - existing_count):.2f}%)")
            
            # Update stats
            stats['total_visibilities'] += total_vis
            stats['existing_flags'] += existing_count
            stats['new_flags'] += new_count
            stats['baselines_processed'] += len(bl_order)
            
            # Collect plot data
            if make_plots:
                for bl_idx, bl in enumerate(bl_order):
                    for c_idx, corr_idx in enumerate(corr_indices):
                        existing = prep_data['existing_flags_batch'][bl_idx, c_idx]
                        final = flags_result[bl_idx, c_idx]
                        new_only = (final > 0) & ~(existing > 0)
                        
                        pct_exist = 100 * np.sum(existing > 0) / existing.size
                        pct_new = 100 * np.sum(new_only) / new_only.size
                        pct_total = 100 * np.sum(final > 0) / final.size
                        
                        plot_data.append({
                            'baseline': bl,
                            'corr_label': info['corr_labels'][corr_idx],
                            'amp': prep_data['amp_batch'][bl_idx, c_idx],
                            'existing_flags': existing,
                            'new_flags': new_only.astype(np.uint8),
                            'pct_existing': pct_exist,
                            'pct_new': pct_new,
                            'pct_total': pct_total,
                        })
            
            # Store flags for writing (combine all corrs back)
            if apply_flags:
                for bl_idx, bl in enumerate(bl_order):
                    full_flags = np.zeros((n_time, n_freq, info['n_corr']), dtype=bool)
                    for c_idx, corr_idx in enumerate(corr_indices):
                        full_flags[:, :, corr_idx] = flags_result[bl_idx, c_idx] > 0
                    field_flags[bl] = full_flags
            
            # Cleanup
            del baseline_data, prep_data, flags_result
            gc.collect()
            cuda_synchronize()
            free_gpu_memory()
        
        # Write flags
        if apply_flags and field_flags:
            if logger:
                logger.info(f"\n  Writing flags for {len(field_flags)} baselines...")
            write_start = time.time()
            write_field_flags(ms_file, field_id, field_flags, logger=logger)
            if logger:
                logger.info(f"  Write: {time.time() - write_start:.2f}s")
        
        # Create field viewer
        if make_plots and plot_data:
            viewer_path = os.path.join(plot_dir, f"field{field_id}_{field_name}.html")
            try:
                result = create_field_viewer(plot_data, field_id, field_name, viewer_path, logger)
                if result:
                    field_viewers[field_id] = result
            except Exception as e:
                if logger:
                    logger.warning(f"  Plot failed: {e}")
    
    executor.shutdown(wait=True)
    
    # Final stats
    total_time = time.time() - total_start
    total_flagged = stats['existing_flags'] + stats['new_flags']
    total_vis = stats['total_visibilities']
    
    stats['total_processing_time'] = total_time
    stats['overall_percent_flagged'] = 100 * total_flagged / max(1, total_vis)
    stats['new_percent_flagged'] = 100 * stats['new_flags'] / max(1, total_vis)
    
    # Master index
    if make_plots and field_viewers:
        field_names_dict = {i: info['field_names'][i] for i in field_ids if i < len(info['field_names'])}
        master = create_master_index(field_viewers, field_names_dict, stats, plot_dir, logger)
        stats['plot_index'] = master
    
    # Summary
    if logger:
        logger.info(f"\n{'='*60}")
        logger.info("POOH FLAGGING COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"  Time: {total_time:.2f}s")
        logger.info(f"  Baselines: {stats['baselines_processed']}")
        logger.info(f"  Visibilities: {stats['total_visibilities']:,}")
        logger.info(f"  Existing flags: {stats['existing_flags']:,}")
        logger.info(f"  New flags: {stats['new_flags']:,}")
        logger.info(f"  Overall: {stats['overall_percent_flagged']:.2f}%")
        logger.info(f"  New: {stats['new_percent_flagged']:.2f}%")
        if 'plot_index' in stats:
            logger.info(f"  Plots: {stats['plot_index']}")
        logger.info(f"{'='*60}")
    
    return stats
