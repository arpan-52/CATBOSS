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
            min_t = min(orig_flags.shape[0], new_flags.shape[0])

            if new_flags.ndim == 2 and orig_flags.ndim == 3:
                # 2D flags -> broadcast to all correlations
                min_f = min(orig_flags.shape[1], new_flags.shape[1])
                new_3d = np.broadcast_to(
                    new_flags[:min_t, :min_f, np.newaxis],
                    (min_t, min_f, orig_flags.shape[2])
                )
                combined[:min_t, :min_f, :] = np.logical_or(
                    orig_flags[:min_t, :min_f, :], new_3d
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

    if not row_flags:
        return 0

    # Group the rows touched by this batch into contiguous runs and do one
    # read+write per run instead of one per row. For NIMKI-style flagging
    # this collapses thousands of getcol/putcol round-trips into a handful.
    sorted_rows = sorted(row_flags.keys())

    with table(ms_file, readonly=False, ack=False) as tb:
        run_start = sorted_rows[0]
        run_end = run_start
        def flush_run(start, end):
            nonlocal n_written
            nrow = end - start + 1
            try:
                flags = tb.getcol('FLAG', startrow=start, nrow=nrow)
                for row in range(start, end + 1):
                    corr_data = row_flags.get(row)
                    if not corr_data:
                        continue
                    local = row - start
                    for corr, chans in corr_data.items():
                        for ci in chans:
                            if ci < flags.shape[1] and corr < flags.shape[2]:
                                flags[local, ci, corr] = True
                                n_written += 1
                tb.putcol('FLAG', flags, startrow=start, nrow=nrow)
            except Exception as e:
                if logger:
                    logger.warning(f"Failed to write rows {start}-{end}: {e}")

        for row in sorted_rows[1:]:
            if row == run_end + 1:
                run_end = row
            else:
                flush_run(run_start, run_end)
                run_start = row
                run_end = row
        flush_run(run_start, run_end)

    return n_written


# Rows per getcol/putcol pass in the batched flag writer. 50k rows of a
# 230-chan x 4-corr MS is ~46 MB, so the peak stays small regardless of MS size.
FLAG_WRITE_CHUNK_ROWS = 50000


def _write_field_flags_batched(
    ms_file: str,
    field_id: int,
    baseline_flags: Dict[Tuple[int, int], np.ndarray],
    spw: Optional[int],
    logger
) -> int:
    """
    Single-pass flag write for a whole field.

    One TaQL selection over the field, then one sequential read and one
    sequential write of FLAG, instead of a full selection + read + write per
    baseline. Row ordering is identical to the per-baseline path: within the
    field selection the rows of any one baseline appear in the same relative
    (table) order as they would in a baseline-restricted selection, and the
    stable argsort below preserves it. That is the same order
    read_baseline_data used to build the arrays being written back.
    """
    from casacore.tables import table

    where = f"FIELD_ID=={int(field_id)}"
    if spw is not None:
        where += f" AND DATA_DESC_ID=={int(spw)}"

    tab = table(ms_file, readonly=False, ack=False)
    try:
        sel = tab.query(where)
        try:
            n_rows = sel.nrows()
            if n_rows == 0:
                return 0

            ant1 = sel.getcol("ANTENNA1")
            ant2 = sel.getcol("ANTENNA2")

            # Group rows by baseline, preserving table order inside each group.
            key = (ant1.astype(np.int64) << 32) | ant2.astype(np.int64)
            order = np.argsort(key, kind="stable")
            skey = key[order]
            starts = np.flatnonzero(
                np.concatenate(([True], skey[1:] != skey[:-1]))
            )
            ends = np.concatenate((starts[1:], [len(skey)]))

            row_index = {}
            for s, e in zip(starts, ends):
                k = int(skey[s])
                row_index[(k >> 32, k & 0xFFFFFFFF)] = order[s:e]

            sample = sel.getcol("FLAG", 0, 1)
            n_chan, n_corr = int(sample.shape[1]), int(sample.shape[2])

            # Scatter the new flags into a field-shaped array, then OR it into
            # FLAG chunk by chunk. Only the overlapping region of each baseline
            # is touched, matching the per-baseline shape-mismatch behaviour.
            new_all = np.zeros((n_rows, n_chan, n_corr), dtype=bool)
            n_written = 0

            for bl, new_flags in baseline_flags.items():
                rows = row_index.get((int(bl[0]), int(bl[1])))
                if rows is None or len(rows) == 0:
                    continue

                arr = np.asarray(new_flags)
                if arr.ndim == 2:
                    # 2D flags -> broadcast across all correlations
                    arr = np.broadcast_to(arr[:, :, None],
                                          (arr.shape[0], arr.shape[1], n_corr))

                nt = min(len(rows), arr.shape[0])
                nf = min(n_chan, arr.shape[1])
                nc = min(n_corr, arr.shape[2])
                if nt == 0:
                    continue

                if nf == n_chan and nc == n_corr:
                    new_all[rows[:nt]] = arr[:nt].astype(bool, copy=False)
                else:
                    new_all[rows[:nt], :nf, :nc] = \
                        arr[:nt, :nf, :nc].astype(bool, copy=False)
                n_written += 1

            for start in range(0, n_rows, FLAG_WRITE_CHUNK_ROWS):
                n = min(FLAG_WRITE_CHUNK_ROWS, n_rows - start)
                flags = sel.getcol("FLAG", start, n)
                np.logical_or(flags, new_all[start:start + n], out=flags)
                sel.putcol("FLAG", flags, start, n)

            sel.flush()
        finally:
            sel.close()
    finally:
        tab.close()

    if logger:
        logger.info(f"  Wrote flags for {n_written}/{len(baseline_flags)} baselines")

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

    Does one pass over the field's rows. The previous implementation called
    apply_flags_to_ms once per baseline, and each of those ran a full
    xds_from_ms + TaQL selection + compute + write-back over the whole MS -
    1711 scans of the table for a 59-antenna array, which made writing ~25% of
    total flagging time. Falls back to that path if the batched write fails.

    Args:
        ms_file: Path to MS file
        field_id: Field ID
        baseline_flags: Dict mapping baseline to flag array
        spw: Optional SPW filter
        logger: Optional logger

    Returns:
        Number of baselines written
    """
    if not baseline_flags:
        return 0

    try:
        return _write_field_flags_batched(
            ms_file, field_id, baseline_flags, spw, logger
        )
    except Exception as e:
        if logger:
            logger.warning(
                f"  Batched flag write failed ({e}); "
                f"falling back to per-baseline writes"
            )

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
                    n_use = min(len(rows), flags.shape[0])
                    if n_use == 0:
                        continue

                    # Batch read: find contiguous runs for efficient I/O
                    rows_arr = np.array(rows[:n_use])

                    # Check if rows are contiguous
                    if len(rows_arr) > 1 and np.all(np.diff(rows_arr) == 1):
                        # Single batch read/write for contiguous rows
                        orig_flags = tb.getcol('FLAG', startrow=int(rows_arr[0]), nrow=n_use)
                        if flags.ndim == 3:
                            combined = np.logical_or(orig_flags, flags[:n_use])
                        else:
                            # 2D flags - broadcast to all correlations
                            combined = orig_flags.copy()
                            for corr in range(orig_flags.shape[-1]):
                                combined[:, :, corr] = np.logical_or(
                                    orig_flags[:, :, corr], flags[:n_use]
                                )
                        tb.putcol('FLAG', combined, startrow=int(rows_arr[0]), nrow=n_use)
                    else:
                        # Non-contiguous: fall back to per-row I/O
                        for i, row in enumerate(rows_arr):
                            orig_flag = tb.getcol('FLAG', startrow=int(row), nrow=1)
                            if flags.ndim == 3:
                                new_flag = np.logical_or(orig_flag[0], flags[i])
                            else:
                                new_flag = orig_flag[0].copy()
                                for corr in range(orig_flag.shape[-1]):
                                    new_flag[:, corr] = np.logical_or(
                                        orig_flag[0, :, corr], flags[i]
                                    )
                            tb.putcol('FLAG', new_flag.reshape(1, *new_flag.shape), startrow=int(row), nrow=1)
        
        return True
        
    except Exception as e:
        if logger:
            logger.error(f"Error writing chunked flags for {baseline}: {e}")
        return False
