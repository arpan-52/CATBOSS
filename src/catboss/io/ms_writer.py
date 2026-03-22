"""
MS file flag writing utilities.

Author: Arpan Pal
Institution: NCRA-TIFR
"""

import numpy as np
import dask
import dask.array as da
from typing import Optional, List, Dict, Tuple, Any
from collections import defaultdict


def apply_flags_to_ms(
    ms_file: str,
    bl: Tuple[int, int],
    field_id: int,
    new_flags: np.ndarray,
    spw: Optional[int] = None,
    logger=None
) -> bool:
    """
    Apply flags to MS file for a specific baseline.
    
    Args:
        ms_file: Path to MS file
        bl: Baseline tuple (ant1, ant2)
        field_id: Field ID
        new_flags: New flag array (OR'd with existing)
        spw: Optional SPW filter
        logger: Optional logger
        
    Returns:
        True if successful
    """
    from daskms import xds_from_ms, xds_to_table
    
    # Build query
    taql = f"FIELD_ID={field_id} AND ANTENNA1={bl[0]} AND ANTENNA2={bl[1]}"
    if spw is not None:
        taql += f" AND DATA_DESC_ID={spw}"
    
    try:
        ds_list = xds_from_ms(ms_file, columns=("FLAG",), taql_where=taql)
        
        if not ds_list or ds_list[0].sizes["row"] == 0:
            return False
        
        ds = ds_list[0]
        orig_flags = ds.FLAG.data.compute()
        
        # Handle shape mismatch
        if orig_flags.shape != new_flags.shape:
            combined = orig_flags.copy()
            
            if len(new_flags.shape) == 2 and len(orig_flags.shape) == 3:
                # 2D flags -> broadcast to 3D
                for corr in range(orig_flags.shape[2]):
                    min_t = min(orig_flags.shape[0], new_flags.shape[0])
                    min_f = min(orig_flags.shape[1], new_flags.shape[1])
                    combined[:min_t, :min_f, corr] = np.logical_or(
                        orig_flags[:min_t, :min_f, corr],
                        new_flags[:min_t, :min_f]
                    )
            else:
                common = tuple(min(d1, d2) for d1, d2 in zip(orig_flags.shape, new_flags.shape))
                slices = tuple(slice(0, d) for d in common)
                combined[slices] = np.logical_or(orig_flags[slices], new_flags[slices])
        else:
            combined = np.logical_or(orig_flags, new_flags)
        
        # Write back
        new_flags_dask = da.from_array(combined, chunks=ds.FLAG.data.chunks)
        updated_ds = ds.assign(FLAG=(ds.FLAG.dims, new_flags_dask))
        
        write_back = xds_to_table([updated_ds], ms_file, ["FLAG"])
        dask.compute(write_back)
        
        return True
        
    except Exception as e:
        if logger:
            logger.error(f"Error writing flags for {bl}: {e}")
        return False


def write_flags_batched(
    ms_file: str,
    flag_operations: List[Dict],
    n_corr: int,
    flag_all_corr: bool = True,
    logger=None
) -> int:
    """
    Write flags in batched mode (for NIMKI-style flagging).
    
    Args:
        ms_file: Path to MS file
        flag_operations: List of dicts with 'row', 'chan_indices', 'corr'
        n_corr: Number of correlations
        flag_all_corr: If True, flag all correlations
        logger: Optional logger
        
    Returns:
        Number of flags written
    """
    from casacore.tables import table
    
    if not flag_operations:
        return 0
    
    # Group by row
    row_flags = defaultdict(lambda: defaultdict(set))
    
    for op in flag_operations:
        row = op['row']
        for ci in op.get('chan_indices', []):
            if flag_all_corr:
                for corr in range(n_corr):
                    row_flags[row][corr].add(ci)
            else:
                corr = op.get('corr', 0)
                row_flags[row][corr].add(ci)
    
    n_written = 0
    
    with table(ms_file, readonly=False, ack=False) as tb:
        for row, corr_data in row_flags.items():
            try:
                flags = tb.getcol('FLAG', startrow=row, nrow=1)
                
                for corr, chans in corr_data.items():
                    for ci in chans:
                        if ci < flags.shape[1]:
                            flags[0, ci, corr] = True
                            n_written += 1
                
                tb.putcol('FLAG', flags, startrow=row, nrow=1)
                
            except Exception as e:
                if logger:
                    logger.warning(f"Failed to write row {row}: {e}")
    
    return n_written


def write_field_flags(
    ms_file: str,
    field_id: int,
    baseline_flags: Dict[Tuple[int, int], np.ndarray],
    spw: Optional[int] = None,
    logger=None
) -> int:
    """
    Write flags for an entire field.
    
    Args:
        ms_file: Path to MS file
        field_id: Field ID
        baseline_flags: Dict mapping baseline to flag array
        spw: Optional SPW filter
        logger: Optional logger
        
    Returns:
        Number of baselines written
    """
    n_written = 0
    
    for bl, new_flags in baseline_flags.items():
        if apply_flags_to_ms(ms_file, bl, field_id, new_flags, spw, logger):
            n_written += 1
    
    if logger:
        logger.info(f"  Wrote flags for {n_written}/{len(baseline_flags)} baselines")
    
    return n_written


def write_chunked_flags(
    ms_file: str,
    field_id: int,
    baseline: Tuple[int, int],
    time_chunks: List[Tuple[float, float]],
    chunk_flags: List[np.ndarray],
    spw: Optional[int] = None,
    logger=None
) -> bool:
    """
    Write flags for time-chunked processing.
    
    Args:
        ms_file: Path to MS file
        field_id: Field ID
        baseline: (ant1, ant2) tuple
        time_chunks: List of (t_start, t_end) boundaries
        chunk_flags: List of flag arrays per chunk
        spw: Optional SPW filter
        logger: Optional logger
        
    Returns:
        True if successful
    """
    from casacore.tables import table
    
    a1, a2 = baseline
    
    try:
        with table(ms_file, readonly=False, ack=False) as tb:
            # Build base query
            base_query = f"FIELD_ID=={field_id} AND ANTENNA1=={a1} AND ANTENNA2=={a2}"
            if spw is not None:
                base_query += f" AND DATA_DESC_ID=={spw}"
            
            for (t_start, t_end), flags in zip(time_chunks, chunk_flags):
                query = f"{base_query} AND TIME>={t_start} AND TIME<{t_end}"
                
                with tb.query(query) as sub:
                    if sub.nrows() == 0:
                        continue
                    
                    rows = sub.rownumbers()
                    
                    for i, row in enumerate(rows):
                        if i >= flags.shape[0]:
                            break
                        
                        orig_flag = tb.getcol('FLAG', startrow=row, nrow=1)
                        
                        # Combine flags
                        if len(flags.shape) == 3:
                            new_flag = np.logical_or(orig_flag[0], flags[i])
                        else:
                            # 2D flags - broadcast
                            new_flag = orig_flag[0].copy()
                            for corr in range(orig_flag.shape[2]):
                                new_flag[:, :, corr] = np.logical_or(
                                    orig_flag[0, :, :, corr] if orig_flag.ndim == 4 else orig_flag[0, :, corr],
                                    flags[i] if len(flags.shape) == 2 else flags[i, :, corr]
                                )
                        
                        tb.putcol('FLAG', new_flag.reshape(1, *new_flag.shape), startrow=row, nrow=1)
        
        return True
        
    except Exception as e:
        if logger:
            logger.error(f"Error writing chunked flags for {baseline}: {e}")
        return False
