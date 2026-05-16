"""
NIMKI - Non-linear Interference Modeling and Korrection Interface

Main processing engine for UV-domain RFI flagging using Gabor basis fitting.

Model: V(r) = Σ Aᵢ · exp(-(r/σᵢ)²/2) · cos(ωᵢ·r + φᵢ)

Author: Arpan Pal
Institution: NCRA-TIFR
"""

import numpy as np
import time
import os
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict
from multiprocessing import Pool, cpu_count

from ..io import (
    get_ms_info, get_frequencies, get_field_ids,
    parse_selection, write_flags_batched, print_ms_summary,
)
from .core_functions import (
    calculate_uv_distances, collect_data_single_corr,
    fit_gabor, fit_gabor_adaptive, flag_outliers,
    is_cpp_available,
)


def _build_base_query(field_id: int, spws: List[int],
                      scan_ids: Optional[List[int]] = None) -> str:
    """Build the common TaQL filter for field + SPW + scan."""
    query = f"FIELD_ID=={field_id}"
    if spws:
        query += f" AND DATA_DESC_ID IN [{','.join(map(str, spws))}]"
    if scan_ids:
        query += f" AND SCAN_NUMBER IN [{','.join(map(str, scan_ids))}]"
    return query


def estimate_chunk_rows(ms_file: str, field_id: int, spws: List[int],
                        time_bounds: np.ndarray,
                        scan_ids: Optional[List[int]] = None) -> List[int]:
    """Query row counts per time chunk without reading data."""
    from casacore.tables import table

    counts = []
    with table(ms_file, ack=False) as tb:
        base_query = _build_base_query(field_id, spws, scan_ids)

        for i in range(len(time_bounds) - 1):
            q = f"{base_query} AND TIME>={time_bounds[i]} AND TIME<{time_bounds[i+1]}"
            with tb.query(q) as sub:
                counts.append(sub.nrows())
    return counts


def estimate_bytes_per_row(ms_file: str, field_id: int, spws: List[int],
                           datacolumn: str,
                           scan_ids: Optional[List[int]] = None) -> int:
    """Read a single row to estimate memory per row."""
    from casacore.tables import table

    with table(ms_file, ack=False) as tb:
        query = _build_base_query(field_id, spws, scan_ids)
        with tb.query(query, sortlist='', limit=1) as sub:
            if sub.nrows() == 0:
                return 0
            d = sub.getcol(datacolumn)   # (1, nchan, ncorr) complex
            f = sub.getcol('FLAG')       # (1, nchan, ncorr) bool
            u = sub.getcol('UVW')        # (1, 3) float64
            # data + flags + uvw + ddid(int32) + time(float64) + row(int64)
            return d.nbytes + f.nbytes + u.nbytes + 4 + 8 + 8


def read_chunk_batch(ms_file: str, field_id: int, t_start: float,
                     t_end: float, datacolumn: str,
                     spws: List[int],
                     scan_ids: Optional[List[int]] = None) -> Optional[Dict[str, np.ndarray]]:
    """Read a contiguous time range covering one or more chunks."""
    from casacore.tables import table

    with table(ms_file, ack=False) as tb:
        query = _build_base_query(field_id, spws, scan_ids)
        query += f" AND TIME>={t_start} AND TIME<{t_end}"

        with tb.query(query) as sub:
            if sub.nrows() == 0:
                return None
            return {
                'data': sub.getcol(datacolumn),
                'flags': sub.getcol('FLAG'),
                'uvw': sub.getcol('UVW'),
                'ddids': sub.getcol('DATA_DESC_ID'),
                'times': sub.getcol('TIME'),
                'rows': np.array(sub.rownumbers()),
            }


def split_batch_into_chunks(batch: Dict[str, np.ndarray],
                            time_bounds: np.ndarray,
                            chunk_indices: List[int],
                            ) -> List[Tuple[int, Optional[Dict[str, np.ndarray]]]]:
    """Split a pre-read batch into individual time chunks."""
    times = batch['times']
    results = []
    for ci in chunk_indices:
        mask = (times >= time_bounds[ci]) & (times < time_bounds[ci + 1])
        if not np.any(mask):
            results.append((ci, None))
        else:
            results.append((ci, {
                'data': batch['data'][mask],
                'flags': batch['flags'][mask],
                'uvw': batch['uvw'][mask],
                'ddids': batch['ddids'][mask],
                'rows': batch['rows'][mask],
            }))
    return results


def process_chunk(args: tuple) -> Dict[str, Any]:
    """
    Process one time chunk - runs in worker process.
    Receives pre-read data arrays (no MS I/O here).
    """
    (chunk, field_id, freqs, corrs, sigma,
     n_components, roam_around, max_components,
     min_improvement, max_iter, tolerance, flag_all_corr,
     do_plot, chunk_idx) = args

    if chunk is None:
        return {'empty': True, 'chunk_idx': chunk_idx, 'flags': [], 'plot_data': []}
    
    data = chunk['data']
    flags = chunk['flags']
    uvw = chunk['uvw']
    ddids = chunk['ddids']
    rows = chunk['rows']
    
    n_rows, n_chan, n_corr = data.shape
    
    # Calculate UV distances per SPW
    c = 299792458.0  # Speed of light
    uv_per_spw = {}
    for spw, freq in freqs.items():
        wavelengths = c / freq
        uv_per_spw[spw] = calculate_uv_distances(uvw, wavelengths)
    
    flag_list = []
    plot_data = []
    stats = {'n_outliers': 0, 'n_points': 0}
    
    # Process each SPW
    for spw in np.unique(ddids):
        if spw not in uv_per_spw:
            continue
        
        mask = ddids == spw
        spw_rows = np.where(mask)[0]
        uv_dist = uv_per_spw[spw]
        
        outlier_positions = defaultdict(set)
        
        # Process each correlation
        for corr in corrs:
            if corr >= n_corr:
                continue
            
            # Collect data (C++ accelerated if available)
            collected = collect_data_single_corr(
                data, flags, uv_dist,
                spw_rows.astype(np.int32),
                corr
            )
            
            uv = collected['uv_dists']
            amp = collected['amplitudes']
            row_idx = collected['row_indices']
            chan_idx = collected['chan_indices']
            
            n_points = len(uv)
            if n_points < 50:
                continue
            
            # Fit Gabor model
            if roam_around:
                fit_result = fit_gabor_adaptive(
                    uv, amp,
                    n_components=n_components,
                    max_components=max_components,
                    min_improvement=min_improvement,
                    max_iter=max_iter,
                    tol=tolerance
                )
            else:
                fit_result = fit_gabor(
                    uv, amp,
                    n_components=n_components,
                    max_iter=max_iter,
                    tol=tolerance
                )
            
            predicted = fit_result['predicted']
            
            # Flag outliers
            outliers, residuals, mad_sigma = flag_outliers(amp, predicted, sigma)
            
            n_outliers = np.sum(outliers)
            stats['n_outliers'] += n_outliers
            stats['n_points'] += n_points
            
            # Record outlier positions
            for i in np.where(outliers)[0]:
                ri, ci = row_idx[i], chan_idx[i]
                outlier_positions[(ri, ci)].add(corr)
            
            # Store plot data if requested
            if do_plot:
                plot_data.append({
                    'field_id': field_id,
                    'chunk_idx': chunk_idx,
                    'spw': int(spw),
                    'corr': corr,
                    'uv': uv.copy(),
                    'amp': amp.copy(),
                    'predicted': predicted.copy(),
                    'residuals': residuals.copy(),
                    'outliers': outliers.copy(),
                    'mad_sigma': mad_sigma,
                    'n_components': fit_result.get('n_components', n_components),
                    'components': fit_result.get('components', []),
                })
        
        # Build flag operations
        for (ri, ci), corr_set in outlier_positions.items():
            actual_row = rows[spw_rows[ri]]
            
            if flag_all_corr:
                flag_list.append({
                    'row': actual_row,
                    'chan_indices': [ci],
                    'corr': 0,  # Will flag all
                })
            else:
                for c in corr_set:
                    flag_list.append({
                        'row': actual_row,
                        'chan_indices': [ci],
                        'corr': c,
                    })
    
    return {
        'empty': False,
        'chunk_idx': chunk_idx,
        'flags': flag_list,
        'plot_data': plot_data,
        'stats': stats,
    }


def hunt_ms(ms_file: str, options: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main NIMKI entry point - process an entire MS file.
    
    Args:
        ms_file: Path to Measurement Set
        options: Processing options dictionary
        
    Returns:
        Results dictionary with statistics
    """
    logger = options.get('logger')
    total_start = time.time()
    
    # NIMKI requires the C++ extension; core_functions raises ImportError
    # at import time if it's missing, so by the time we get here the
    # extension is guaranteed to be loaded. Keep the is_cpp_available()
    # call only as a sanity log line.
    if logger and is_cpp_available():
        logger.info("C++ extension loaded - using accelerated Gabor fitting")
    
    # Get MS info
    if logger:
        logger.info(f"\nReading MS: {ms_file}")
    
    info = get_ms_info(ms_file, logger)
    print_ms_summary(info, logger)
    
    # Parse selections
    field_ids = parse_selection(options.get('field'), list(range(info['n_fields'])))
    if field_ids is None:
        field_ids = list(range(info['n_fields']))
    
    spw_ids = parse_selection(options.get('spw'), list(range(info['n_spw'])))
    if spw_ids is None:
        spw_ids = list(range(info['n_spw']))
    
    corr_ids = parse_selection(options.get('corr'), list(range(info['n_corr'])))
    if corr_ids is None:
        corr_ids = list(range(info['n_corr']))
    
    # Get parameters
    n_components = options.get('n_components', 5)
    roam_around = options.get('roam_around', False)
    max_components = options.get('max_components', 12)
    min_improvement = options.get('min_improvement', 0.05)
    max_iter = options.get('max_iter', 500)
    tolerance = options.get('tolerance', 1e-8)
    sigma = options.get('sigma', 5.0)
    timebin_min = options.get('timebin', 30.0)
    flag_all_corr = options.get('flag_all_corr', True)
    apply_flags = options.get('apply_flags', False)
    do_plot = options.get('plots', False)
    plot_dir = options.get('plot_dir', 'nimki_plots')
    ncpu = options.get('ncpu', 0)
    
    if ncpu <= 0:
        ncpu = cpu_count()
    
    datacolumn = options.get('datacolumn', 'DATA')

    # Parse scan selection
    scan_sel = options.get('scan')
    scan_ids = None
    if scan_sel:
        scan_ids = [int(s.strip()) for s in scan_sel.split(',')]

    # Get frequencies
    freqs = get_frequencies(ms_file, spw_ids)
    
    if logger:
        logger.info("  PROCESSING CONFIGURATION")
        logger.info(f"  Fields: {field_ids}")
        logger.info(f"  SPWs: {spw_ids}")
        logger.info(f"  Correlations: {[info['corr_labels'][i] for i in corr_ids]}")
        logger.info(f"  Components: {n_components}" + 
                    (f" → {max_components} (adaptive)" if roam_around else ""))
        logger.info(f"  Sigma threshold: {sigma}")
        logger.info(f"  Time bin: {timebin_min} min")
        logger.info(f"  CPUs: {ncpu}")
        logger.info(f"  Apply flags: {apply_flags}")
    
    # Statistics
    total_flags = 0
    total_points = 0
    total_outliers = 0
    all_plot_data = []
    
    # Process each field
    for fid in field_ids:
        if logger:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing Field {fid}: {info['field_names'][fid]}")
            logger.info(f"{'='*60}")
        
        # Build base query for this field (with optional scan filter)
        from casacore.tables import table
        base_field_query = f"FIELD_ID=={fid}"
        if scan_ids:
            base_field_query += f" AND SCAN_NUMBER IN [{','.join(map(str, scan_ids))}]"

        # Get time range for this field
        with table(ms_file, ack=False) as tb:
            with tb.query(base_field_query) as sub:
                if sub.nrows() == 0:
                    if logger:
                        logger.info("  No data for this field")
                    continue
                times = sub.getcol('TIME')

        t_min, t_max = times.min(), times.max()
        chunk_sec = timebin_min * 60
        time_bounds = np.arange(t_min, t_max + chunk_sec, chunk_sec)
        n_chunks = len(time_bounds) - 1

        if logger:
            logger.info(f"  Time range: {t_max - t_min:.0f}s in {n_chunks} chunks")

        # Figure out how many chunks we can fit in memory at once
        bpr = estimate_bytes_per_row(ms_file, fid, spw_ids, datacolumn, scan_ids)
        chunk_rows = estimate_chunk_rows(ms_file, fid, spw_ids, time_bounds, scan_ids)

        import psutil
        avail_mem = psutil.virtual_memory().available
        mem_budget = int(avail_mem * 0.5)  # use at most 50% of free RAM

        # Group consecutive chunks into batches that fit in budget
        batches = []  # list of (start_chunk_idx, end_chunk_idx)
        batch_start = 0
        batch_bytes = 0
        for ci in range(n_chunks):
            row_bytes = chunk_rows[ci] * bpr
            if batch_bytes + row_bytes > mem_budget and ci > batch_start:
                batches.append((batch_start, ci))
                batch_start = ci
                batch_bytes = row_bytes
            else:
                batch_bytes += row_bytes
        batches.append((batch_start, n_chunks))

        if logger:
            logger.info(f"  {len(batches)} I/O batch(es), ~{bpr} bytes/row, "
                        f"budget {mem_budget / 1e9:.1f} GB")

        # Process batch by batch: one read, then dispatch to workers
        results = []
        for b_start, b_end in batches:
            batch_data = read_chunk_batch(
                ms_file, fid,
                time_bounds[b_start], time_bounds[b_end],
                datacolumn, spw_ids, scan_ids,
            )

            chunk_indices = list(range(b_start, b_end))
            if batch_data is None:
                for ci in chunk_indices:
                    results.append({'empty': True, 'chunk_idx': ci,
                                    'flags': [], 'plot_data': []})
                continue

            split = split_batch_into_chunks(batch_data, time_bounds,
                                            chunk_indices)
            del batch_data

            tasks = [
                (chunk, fid, freqs, corr_ids, sigma,
                 n_components, roam_around, max_components,
                 min_improvement, max_iter, tolerance, flag_all_corr,
                 do_plot, ci)
                for ci, chunk in split
            ]
            del split

            n_workers = min(ncpu, len(tasks))
            if n_workers <= 1:
                batch_results = [process_chunk(t) for t in tasks]
            else:
                # Limit OMP threads per worker to avoid oversubscription.
                # Save/restore the caller's OMP_NUM_THREADS so we don't leak
                # our internal value into the rest of the process (or any
                # downstream libs spawned after NIMKI returns).
                omp_threads = max(1, ncpu // n_workers)
                _prev_omp = os.environ.get('OMP_NUM_THREADS')
                os.environ['OMP_NUM_THREADS'] = str(omp_threads)
                try:
                    with Pool(n_workers) as pool:
                        batch_results = pool.map(process_chunk, tasks)
                finally:
                    if _prev_omp is None:
                        os.environ.pop('OMP_NUM_THREADS', None)
                    else:
                        os.environ['OMP_NUM_THREADS'] = _prev_omp
            del tasks
            results.extend(batch_results)
        
        # Collect results
        field_flags = []
        for r in results:
            if not r.get('empty', True):
                field_flags.extend(r['flags'])
                if do_plot:
                    all_plot_data.extend(r.get('plot_data', []))
                stats = r.get('stats', {})
                total_points += stats.get('n_points', 0)
                total_outliers += stats.get('n_outliers', 0)
        
        if logger:
            logger.info(f"  Field {fid}: {len(field_flags)} flag operations")
        
        # Write flags
        if apply_flags and field_flags:
            if logger:
                logger.info("  Writing flags...")
            n_written = write_flags_batched(
                ms_file, field_flags, info['n_corr'],
                flag_all_corr=flag_all_corr, logger=logger
            )
            total_flags += n_written
            if logger:
                logger.info(f"  Wrote {n_written} flags")
    
    # Generate plots
    if do_plot and all_plot_data:
        if logger:
            logger.info(f"\nGenerating {len(all_plot_data)} plots...")
        os.makedirs(plot_dir, exist_ok=True)
        _create_bokeh_plots(all_plot_data, plot_dir, info['corr_labels'], logger)
    
    # Summary
    total_time = time.time() - total_start
    pct_flagged = 100 * total_outliers / max(1, total_points)
    
    results = {
        'total_processing_time': total_time,
        'total_points': total_points,
        'total_outliers': total_outliers,
        'total_flags_written': total_flags,
        'percent_flagged': pct_flagged,
    }
    
    if logger:
        logger.info(f"\n{'='*60}")
        logger.info("NIMKI FLAGGING COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"  Processing time: {total_time:.1f}s")
        logger.info(f"  Total points: {total_points:,}")
        logger.info(f"  Outliers: {total_outliers:,} ({pct_flagged:.2f}%)")
        logger.info(f"  Flags written: {total_flags:,}")
        if do_plot:
            logger.info(f"  Plots saved to: {plot_dir}/")
        logger.info(f"{'='*60}")
    
    return results


def _create_bokeh_plots(
    plot_data: List[Dict],
    plot_dir: str,
    corr_labels: List[str],
    logger=None
):
    """Generate interactive UV-plane plots via PIL/HTML viewer."""
    from ..plotting.bokeh_plots import create_nimki_viewer
    create_nimki_viewer(plot_data, plot_dir, corr_labels, logger=logger)
