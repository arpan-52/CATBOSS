"""
CATBOSS Plotting - Modern single-page diagnostic viewer.

Features:
- Single HTML file per field with all baselines
- Clickable baseline navigation with keyboard support
- Dark modern theme
- Left: Normalized dynamic spectra
- Right: With flag overlay (Red=existing, White=new)
"""

from .bokeh_plots import (
    create_field_viewer,
    create_master_index,
)

__all__ = [
    'create_field_viewer',
    'create_master_index',
]
