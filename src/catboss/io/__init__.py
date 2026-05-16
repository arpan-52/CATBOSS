"""
I/O utilities for CATBOSS.

Shared MS file reading and writing functionality.
"""

from .ms_reader import (
    get_valid_antennas,
    get_ms_info,
    get_frequencies,
    get_field_ids,
    get_unique_baselines,
    parse_selection,
    parse_corr_selection,
    parse_baseline_selection,
    read_data_column,
    read_baseline_data,
    read_time_chunks,
    print_ms_summary,
)

from .virtual_corr import (
    detect_feed_type,
    get_virtual_corr_constituents,
    compute_virtual_corr,
)

from .ms_writer import (
    apply_flags_to_ms,
    write_flags_batched,
    write_field_flags,
    write_chunked_flags,
)

__all__ = [
    # Reader
    'get_valid_antennas',
    'get_ms_info',
    'get_frequencies',
    'get_field_ids',
    'get_unique_baselines',
    'parse_selection',
    'parse_corr_selection',
    'parse_baseline_selection',
    'read_data_column',
    'read_baseline_data',
    'read_time_chunks',
    'print_ms_summary',
    # Virtual correlations
    'detect_feed_type',
    'get_virtual_corr_constituents',
    'compute_virtual_corr',
    # Writer
    'apply_flags_to_ms',
    'write_flags_batched',
    'write_field_flags',
    'write_chunked_flags',
]
