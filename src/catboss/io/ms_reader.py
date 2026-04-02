"""
MS file reading utilities for CATBOSS.

Efficient reading with:
- Dummy antenna detection
- Lazy evaluation with dask
- Chunked processing
- Field/SPW/baseline selection

Author: Arpan Pal
Institution: NCRA-TIFR
"""

import numpy as np
import dask
import dask.array as da
from typing import Optional, List, Dict, Tuple, Any, Set, Union
from collections import defaultdict


def get_valid_antennas(ms_file: str, logger=None) -> List[int]:
    """
    Get valid (non-dummy) antenna indices.
    
    Detects dummy antennas by:
    1. Position is all zeros
    2. Name contains 'dummy', 'pad', etc.
    3. FLAG_ROW is True
    4. Not present in first 10000 rows of data
    
    Args:
        ms_file: Path to MS file
        logger: Optional logger
        
    Returns:
        List of valid antenna indices
    """
    from casacore.tables import table
    
    # Get antenna metadata
    with table(f"{ms_file}::ANTENNA", ack=False) as ant_tab:
        positions = ant_tab.getcol("POSITION")
        names = ant_tab.getcol("NAME")
        n_ant = ant_tab.nrows()
        
        # Check for FLAG_ROW column
        has_flag_row = "FLAG_ROW" in ant_tab.colnames()
        if has_flag_row:
            flag_rows = ant_tab.getcol("FLAG_ROW")
        else:
            flag_rows = np.zeros(n_ant, dtype=bool)
    
    # First pass: filter by metadata
    candidate_antennas = set()
    for i in range(n_ant):
        # Skip if position is all zeros
        if np.all(positions[i] == 0):
            if logger:
                logger.debug(f"  Antenna {i} ({names[i]}): skipped (zero position)")
            continue
        
        # Skip if name suggests dummy
        name_lower = names[i].lower()
        if any(x in name_lower for x in ['dummy', 'pad', 'test', 'invalid']):
            if logger:
                logger.debug(f"  Antenna {i} ({names[i]}): skipped (dummy name)")
            continue
        
        # Skip if flagged
        if flag_rows[i]:
            if logger:
                logger.debug(f"  Antenna {i} ({names[i]}): skipped (FLAG_ROW)")
            continue
        
        candidate_antennas.add(i)
    
    # Second pass: verify against actual data (first 10000 rows)
    with table(ms_file, ack=False) as tb:
        n_rows = min(10000, tb.nrows())
        if n_rows > 0:
            ant1 = tb.getcol("ANTENNA1", nrow=n_rows)
            ant2 = tb.getcol("ANTENNA2", nrow=n_rows)
            
            data_antennas = set(ant1) | set(ant2)
            
            # Keep only antennas that appear in data
            valid_antennas = candidate_antennas & data_antennas
        else:
            valid_antennas = candidate_antennas
    
    valid_list = sorted(list(valid_antennas))
    
    if logger:
        logger.info(f"  Valid antennas: {len(valid_list)}/{n_ant} "
                    f"(filtered {n_ant - len(valid_list)} dummy/unused)")
    
    return valid_list


def get_ms_info(ms_file: str, logger=None) -> Dict[str, Any]:
    """
    Get MS metadata efficiently.
    
    Args:
        ms_file: Path to MS file
        logger: Optional logger
        
    Returns:
        Dictionary with MS metadata
    """
    from casacore.tables import table
    
    info = {}
    
    # Main table info
    with table(ms_file, ack=False) as tb:
        info['n_rows'] = tb.nrows()
        
        # Get shape from first row
        if info['n_rows'] > 0:
            data_shape = tb.getcol('DATA', nrow=1).shape
            info['n_chan'] = data_shape[1]
            info['n_corr'] = data_shape[2]
        else:
            info['n_chan'] = 0
            info['n_corr'] = 0
    
    # Field info
    with table(f"{ms_file}::FIELD", ack=False) as ft:
        info['field_names'] = list(ft.getcol('NAME'))
        info['n_fields'] = ft.nrows()
    
    # SPW info
    with table(f"{ms_file}::SPECTRAL_WINDOW", ack=False) as st:
        info['n_spw'] = st.nrows()
        info['spw_chans'] = [st.getcell('NUM_CHAN', i) for i in range(st.nrows())]
        info['spw_freqs'] = []
        for i in range(st.nrows()):
            freqs = st.getcol('CHAN_FREQ', startrow=i, nrow=1)[0]
            info['spw_freqs'].append(freqs)
        # Center frequency per SPW (MHz)
        info['spw_center_mhz'] = [np.mean(f)/1e6 for f in info['spw_freqs']]
    
    # Polarization info
    with table(f"{ms_file}::POLARIZATION", ack=False) as pt:
        corr_types = pt.getcol('CORR_TYPE')[0]
        corr_map = {
            1: 'I', 2: 'Q', 3: 'U', 4: 'V',
            5: 'RR', 6: 'RL', 7: 'LR', 8: 'LL',
            9: 'XX', 10: 'XY', 11: 'YX', 12: 'YY'
        }
        info['corr_labels'] = [corr_map.get(c, str(c)) for c in corr_types]
        info['corr_types'] = list(corr_types)
    
    # Valid antennas
    info['valid_antennas'] = get_valid_antennas(ms_file, logger)
    info['n_antennas'] = len(info['valid_antennas'])
    
    return info


def get_frequencies(ms_file: str, spws: Optional[List[int]] = None) -> Dict[int, np.ndarray]:
    """
    Get frequencies for selected SPWs.
    
    Args:
        ms_file: Path to MS file
        spws: List of SPW indices (None = all)
        
    Returns:
        Dictionary mapping SPW index to frequency array (Hz)
    """
    from casacore.tables import table
    
    with table(f"{ms_file}::SPECTRAL_WINDOW", ack=False) as st:
        n_spw = st.nrows()
        if spws is None:
            spws = list(range(n_spw))
        
        freqs = {}
        for spw in spws:
            if 0 <= spw < n_spw:
                freqs[spw] = st.getcol('CHAN_FREQ', startrow=spw, nrow=1)[0]
    
    return freqs


def get_field_ids(ms_file: str) -> List[int]:
    """Get all field IDs from MS."""
    from casacore.tables import table
    
    with table(f"{ms_file}::FIELD", ack=False) as ft:
        return list(range(ft.nrows()))


def get_unique_baselines(
    ms_file: str,
    field_id: Optional[int] = None,
    valid_antennas: Optional[List[int]] = None,
    exclude_autocorr: bool = False,
    sample_rows: int = 50000,
    logger=None
) -> List[Tuple[int, int]]:
    """
    Get unique baselines efficiently by sampling.
    
    Args:
        ms_file: Path to MS file
        field_id: Optional field ID to filter
        valid_antennas: Optional list of valid antenna indices
        exclude_autocorr: If True, exclude auto-correlations
        sample_rows: Number of rows to sample
        logger: Optional logger
        
    Returns:
        List of (ant1, ant2) tuples
    """
    from casacore.tables import table
    
    baselines = set()
    
    with table(ms_file, ack=False) as tb:
        # Build query
        if field_id is not None:
            query = f"FIELD_ID=={field_id}"
            with tb.query(query) as sub:
                n_rows = min(sample_rows, sub.nrows())
                if n_rows > 0:
                    ant1 = sub.getcol("ANTENNA1", nrow=n_rows)
                    ant2 = sub.getcol("ANTENNA2", nrow=n_rows)
        else:
            n_rows = min(sample_rows, tb.nrows())
            if n_rows > 0:
                ant1 = tb.getcol("ANTENNA1", nrow=n_rows)
                ant2 = tb.getcol("ANTENNA2", nrow=n_rows)
        
        if n_rows > 0:
            for a1, a2 in zip(ant1, ant2):
                # Filter by valid antennas
                if valid_antennas is not None:
                    if a1 not in valid_antennas or a2 not in valid_antennas:
                        continue
                
                # Exclude auto-correlations if requested
                if exclude_autocorr and a1 == a2:
                    continue
                
                baselines.add((int(a1), int(a2)))
    
    result = sorted(list(baselines))
    
    if logger:
        logger.debug(f"  Found {len(result)} unique baselines")
    
    return result


def parse_selection(
    selection_str: Optional[str],
    valid_values: Optional[List[int]] = None,
    default_all: bool = True
) -> Optional[List[int]]:
    """
    Parse comma-separated selection string.
    
    Args:
        selection_str: String like "0,1,2" or "all" or None
        valid_values: Optional list of valid values to filter against
        default_all: If True, return None (meaning all) for empty/None input
        
    Returns:
        List of selected values, or None meaning "all"
    """
    if selection_str is None or selection_str.strip().lower() in ('', 'all'):
        return None if default_all else valid_values
    
    try:
        values = [int(x.strip()) for x in selection_str.split(',') if x.strip()]
        
        if valid_values is not None:
            values = [v for v in values if v in valid_values]
        
        return values if values else None
        
    except ValueError:
        return None


def parse_baseline_selection(
    baseline_str: Optional[str],
    valid_antennas: Optional[List[int]] = None
) -> Optional[List[Tuple[int, int]]]:
    """
    Parse baseline selection string like "0-1,0-2,1-2".
    
    Args:
        baseline_str: String like "0-1,0-2" or "all"
        valid_antennas: Optional list of valid antenna indices
        
    Returns:
        List of (ant1, ant2) tuples, or None meaning "all"
    """
    if baseline_str is None or baseline_str.strip().lower() in ('', 'all'):
        return None
    
    baselines = []
    try:
        for bl in baseline_str.split(','):
            bl = bl.strip()
            if '-' in bl:
                parts = bl.split('-')
                a1, a2 = int(parts[0]), int(parts[1])
                
                if valid_antennas is not None:
                    if a1 not in valid_antennas or a2 not in valid_antennas:
                        continue
                
                baselines.append((a1, a2))
        
        return baselines if baselines else None
        
    except (ValueError, IndexError):
        return None


def read_data_column(
    ms_file: str,
    datacolumn: str,
    columns: Tuple[str, ...] = ("DATA", "FLAG"),
    taql_where: Optional[str] = None,
    chunks: Optional[Dict[str, int]] = None
) -> List[Any]:
    """
    Read data from MS with support for RESIDUAL_DATA calculation.

    Args:
        ms_file: Path to MS file
        datacolumn: Column name (DATA, CORRECTED_DATA, RESIDUAL_DATA, MODEL_DATA)
        columns: Tuple of columns to read
        taql_where: Optional TAQL query string
        chunks: Optional chunking specification

    Returns:
        List of xarray datasets
    """
    from daskms import xds_from_ms
    
    # Handle RESIDUAL_DATA specially
    if datacolumn.upper() == 'RESIDUAL_DATA':
        cols_list = list(columns)
        if 'DATA' in cols_list:
            cols_list.remove('DATA')
        cols_with_model = tuple(cols_list + ['DATA', 'MODEL_DATA'])

        kwargs = {'columns': cols_with_model}
        if taql_where:
            kwargs['taql_where'] = taql_where
        if chunks:
            kwargs['chunks'] = chunks

        ds_list = xds_from_ms(ms_file, **kwargs)

        result_ds_list = []
        for ds in ds_list:
            if not hasattr(ds, 'MODEL_DATA'):
                raise ValueError(
                    "RESIDUAL_DATA requested but MODEL_DATA not found. "
                    "Run calibration first or use different column."
                )
            
            if ds.DATA.shape != ds.MODEL_DATA.shape:
                raise ValueError(
                    f"DATA shape {ds.DATA.shape} != MODEL_DATA shape "
                    f"{ds.MODEL_DATA.shape}. Cannot compute RESIDUAL_DATA."
                )
            residual_data = ds.DATA - ds.MODEL_DATA
            ds_modified = ds.assign(DATA=residual_data)
            ds_modified = ds_modified.drop_vars('MODEL_DATA')
            result_ds_list.append(ds_modified)

        return result_ds_list

    else:
        # Normal case
        cols_list = list(columns)
        if 'DATA' in cols_list and datacolumn.upper() != 'DATA':
            cols_list[cols_list.index('DATA')] = datacolumn
        cols_final = tuple(cols_list)

        kwargs = {'columns': cols_final}
        if taql_where:
            kwargs['taql_where'] = taql_where
        if chunks:
            kwargs['chunks'] = chunks

        try:
            ds_list = xds_from_ms(ms_file, **kwargs)

            # Rename back to 'DATA' for consistent processing
            result_ds_list = []
            for ds in ds_list:
                if hasattr(ds, datacolumn) and datacolumn != 'DATA':
                    ds_modified = ds.rename({datacolumn: 'DATA'})
                    result_ds_list.append(ds_modified)
                else:
                    result_ds_list.append(ds)

            return result_ds_list

        except Exception as e:
            raise ValueError(f"Failed to read column '{datacolumn}': {e}")


def read_baseline_data(
    ms_file: str,
    field_id: int,
    baselines: List[Tuple[int, int]],
    datacolumn: str = "DATA",
    spw: Optional[int] = None,
    chunk_size: int = 200000,
    logger=None
) -> Dict[Tuple[int, int], Dict[str, np.ndarray]]:
    """
    Read data for multiple baselines efficiently.
    
    Args:
        ms_file: Path to MS file
        field_id: Field ID
        baselines: List of (ant1, ant2) tuples
        datacolumn: Data column to read
        spw: Optional SPW to filter
        chunk_size: Chunk size for reading
        logger: Optional logger
        
    Returns:
        Dictionary mapping baseline to {'data': array, 'flags': array}
    """
    from daskms import xds_from_ms
    
    # Build TAQL query
    bl_clauses = [f"(ANTENNA1={a1} AND ANTENNA2={a2})" for a1, a2 in baselines]
    taql_where = f"FIELD_ID={field_id} AND ({' OR '.join(bl_clauses)})"
    
    if spw is not None:
        taql_where += f" AND DATA_DESC_ID={spw}"
    
    # Read
    ds_list = read_data_column(
        ms_file,
        datacolumn,
        columns=("DATA", "FLAG", "ANTENNA1", "ANTENNA2"),
        taql_where=taql_where,
        chunks={"row": chunk_size},
    )
    
    # Organize by baseline — collect row indices first, then slice once
    baselines_set = set(baselines)
    baseline_rows = defaultdict(list)  # bl -> list of (data_row, chunk_idx)

    all_chunks = []
    for ds in ds_list:
        ant1, ant2, data, flags = dask.compute(
            ds.ANTENNA1.data, ds.ANTENNA2.data, ds.DATA.data, ds.FLAG.data
        )
        chunk_idx = len(all_chunks)
        all_chunks.append((data, flags))

        for i, (a1, a2) in enumerate(zip(ant1, ant2)):
            bl = (int(a1), int(a2))
            if bl in baselines_set:
                baseline_rows[bl].append((i, chunk_idx))

    # Build result arrays with a single np.empty + direct copy
    result = {}
    for bl in baselines:
        if bl not in baseline_rows:
            continue
        rows = baseline_rows[bl]
        n_rows = len(rows)
        if n_rows == 0:
            continue

        # Get shape from first row
        first_i, first_ci = rows[0]
        sample_data = all_chunks[first_ci][0][first_i]
        data_out = np.empty((n_rows, *sample_data.shape), dtype=sample_data.dtype)
        flags_out = np.empty((n_rows, *sample_data.shape), dtype=all_chunks[first_ci][1][first_i].dtype)

        for out_idx, (row_i, ci) in enumerate(rows):
            data_out[out_idx] = all_chunks[ci][0][row_i]
            flags_out[out_idx] = all_chunks[ci][1][row_i]

        result[bl] = {"data": data_out, "flags": flags_out}

    return result


def read_time_chunks(
    ms_file: str,
    field_id: int,
    timebin_sec: float,
    logger=None
) -> List[Tuple[float, float]]:
    """
    Get time chunk boundaries.
    
    Args:
        ms_file: Path to MS file
        field_id: Field ID
        timebin_sec: Time bin size in seconds
        logger: Optional logger
        
    Returns:
        List of (t_start, t_end) tuples in MJD seconds
    """
    from casacore.tables import table
    
    with table(ms_file, ack=False) as tb:
        query = f"FIELD_ID=={field_id}"
        with tb.query(query) as sub:
            if sub.nrows() == 0:
                return []
            times = sub.getcol('TIME')
    
    t_min, t_max = times.min(), times.max()
    
    if timebin_sec <= 0:
        return [(t_min, t_max + 1)]
    
    boundaries = []
    t_start = t_min
    while t_start < t_max:
        t_end = min(t_start + timebin_sec, t_max + 1)
        boundaries.append((t_start, t_end))
        t_start = t_end
    
    if logger:
        logger.debug(f"  Time range: {t_max - t_min:.0f}s in {len(boundaries)} chunks")
    
    return boundaries


def print_ms_summary(info: Dict[str, Any], logger):
    """Print MS summary to logger."""
    logger.info("  MEASUREMENT SET INFO")
    logger.info(f"  Rows: {info['n_rows']:,}")
    logger.info(f"  Channels: {info['n_chan']}")
    logger.info(f"  Correlations: {info['n_corr']} ({', '.join(info['corr_labels'])})")
    logger.info(f"  Fields: {info['n_fields']} ({', '.join(info['field_names'])})")
    logger.info(f"  SPWs: {info['n_spw']} (center freqs: {', '.join(f'{f:.1f} MHz' for f in info['spw_center_mhz'])})")
    logger.info(f"  Antennas: {info['n_antennas']} valid")
