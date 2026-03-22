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


def read_chunk(
    ms_file: str,
    field_id: int,
    t_start: float,
    t_end: float,
    datacolumn: str,
    spws: List[int]
) -> Optional[Dict[str, np.ndarray]]:
    """Read one time chunk from MS."""
    from casacore.tables import table
    
    with table(ms_file, ack=False) as tb:
        query = f"FIELD_ID=={field_id} AND TIME>={t_start} AND TIME<{t_end}"
        if spws:
            spw_str = ','.join(map(str, spws))
            query += f" AND DATA_DESC_ID IN [{spw_str}]"
        
        with tb.query(query) as sub:
            if sub.nrows() == 0:
                return None
            
            return {
                'data': sub.getcol(datacolumn),
                'flags': sub.getcol('FLAG'),
                'uvw': sub.getcol('UVW'),
                'ddids': sub.getcol('DATA_DESC_ID'),
                'rows': sub.rownumbers(),
            }


def process_chunk(args: tuple) -> Dict[str, Any]:
    """
    Process one time chunk - runs in worker process.
    Returns flags and statistics.
    """
    (ms_file, field_id, t_start, t_end, datacolumn, spws, freqs,
     corrs, sigma, n_components, roam_around, max_components,
     min_improvement, max_iter, tolerance, flag_all_corr, 
     do_plot, chunk_idx) = args
    
    chunk = read_chunk(ms_file, field_id, t_start, t_end, datacolumn, spws)
    
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
    
    # Check C++ availability
    if is_cpp_available():
        if logger:
            logger.info("C++ extension available - using accelerated fitting")
    else:
        if logger:
            logger.warning("✗ C++ extension not available - using Python fallback")
            logger.warning("  Performance will be reduced. Build with: pip install -e .")
    
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
        
        # Get time range for this field
        from casacore.tables import table
        with table(ms_file, ack=False) as tb:
            with tb.query(f"FIELD_ID=={fid}") as sub:
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
        
        # Build task list
        tasks = [
            (ms_file, fid, time_bounds[i], time_bounds[i+1],
             datacolumn, spw_ids, freqs, corr_ids, sigma,
             n_components, roam_around, max_components,
             min_improvement, max_iter, tolerance, flag_all_corr,
             do_plot, i)
            for i in range(n_chunks)
        ]
        
        # Process chunks
        if ncpu == 1:
            results = [process_chunk(t) for t in tasks]
        else:
            with Pool(ncpu) as pool:
                results = pool.map(process_chunk, tasks)
        
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
