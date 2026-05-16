"""
Virtual correlation computation for CATBOSS POOH.

Computes derived correlations (Stokes V, linear polarization P) from raw MS
correlation products, with proper Jones matrix algebra for both circular and
linear feed systems.

  V (Stokes V):
    Circular feeds: V = (RR - LL) / 2
    Linear feeds:   V = -1j * (XY - YX) / 2

  P (linear polarization):
    Circular feeds: P = (RL + LR) / 2   [LR = conj(RL) in practice]
    Linear feeds:   P = (XY + YX) / 2

Flag policy: a virtual corr sample is masked wherever ANY constituent raw
correlation is already flagged (union).  When new outliers are found in V or P,
flags are written back to the constituent raw correlations only — not to all
four — unless --propagate-flags is set.

Author: Arpan Pal
Institution: NCRA-TIFR
"""

from typing import List, Tuple, Dict
import numpy as np


_CIRCULAR_CORRS = {'RR', 'RL', 'LR', 'LL'}
_LINEAR_CORRS   = {'XX', 'XY', 'YX', 'YY'}


# ── feed type detection ─────────────────────────────────────────────────────────

def detect_feed_type(corr_labels: List[str]) -> str:
    """
    Identify the feed type from MS correlation labels.

    Returns one of: 'circular', 'linear', 'stokes', 'unknown'.
    """
    label_set = set(corr_labels)
    if label_set & _CIRCULAR_CORRS:
        return 'circular'
    if label_set & _LINEAR_CORRS:
        return 'linear'
    if label_set & {'I', 'Q', 'U', 'V'}:
        return 'stokes'
    return 'unknown'


# ── constituent lookup ──────────────────────────────────────────────────────────

def get_virtual_corr_constituents(
    name: str,
    corr_labels: List[str],
) -> List[int]:
    """
    Return the raw correlation indices that form a virtual correlation.

    Args:
        name:        'V' (Stokes V) or 'P' (linear polarization).
        corr_labels: List of correlation names from the MS, e.g. ['RR','RL','LR','LL'].

    Returns:
        [c0, c1] — two integer indices into the correlation axis.

    Raises:
        ValueError: if the required correlations are absent or feed type is unsupported.
    """
    name = name.upper()
    feed = detect_feed_type(corr_labels)
    lbl2idx: Dict[str, int] = {lbl: i for i, lbl in enumerate(corr_labels)}

    if name == 'V':
        if feed == 'circular':
            _require('V (circular)', ['RR', 'LL'], lbl2idx)
            return [lbl2idx['RR'], lbl2idx['LL']]
        elif feed == 'linear':
            _require('V (linear)', ['XY', 'YX'], lbl2idx)
            return [lbl2idx['XY'], lbl2idx['YX']]
        else:
            raise ValueError(
                f"Cannot form Stokes V: feed type is '{feed}', "
                f"need circular (RR/LL) or linear (XY/YX) correlations."
            )

    elif name == 'P':
        if feed == 'circular':
            _require('P (circular)', ['RL', 'LR'], lbl2idx)
            return [lbl2idx['RL'], lbl2idx['LR']]
        elif feed == 'linear':
            _require('P (linear)', ['XY', 'YX'], lbl2idx)
            return [lbl2idx['XY'], lbl2idx['YX']]
        else:
            raise ValueError(
                f"Cannot form linear pol P: feed type is '{feed}', "
                f"need circular (RL/LR) or linear (XY/YX) correlations."
            )

    else:
        raise ValueError(f"Unknown virtual correlation '{name}'. Supported: 'V', 'P'.")


def _require(label: str, corrs: List[str], mapping: Dict[str, int]) -> None:
    """Raise ValueError listing any missing correlations."""
    missing = [c for c in corrs if c not in mapping]
    if missing:
        raise ValueError(
            f"Virtual corr {label} requires {corrs} but {missing} not present in data."
        )


# ── computation ─────────────────────────────────────────────────────────────────

def compute_virtual_corr(
    data: np.ndarray,
    flags: np.ndarray,
    corr_labels: List[str],
    name: str,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Form a virtual correlation from raw MS visibilities.

    Args:
        data:        (n_rows, n_chan, n_corr)  complex64 or complex128
        flags:       (n_rows, n_chan, n_corr)  bool
        corr_labels: list of corr names matching the last axis of data
        name:        'V' or 'P'

    Returns:
        virt_data   : (n_rows, n_chan) complex128 — virtual visibility
        virt_flags  : (n_rows, n_chan) bool       — union of constituent flags
        constituents: List[int]                   — raw corr indices used
    """
    name = name.upper()
    feed = detect_feed_type(corr_labels)
    constituents = get_virtual_corr_constituents(name, corr_labels)
    c0, c1 = constituents[0], constituents[1]

    d0 = data[:, :, c0].astype(np.complex128)
    d1 = data[:, :, c1].astype(np.complex128)

    if name == 'V':
        if feed == 'circular':
            # Stokes V = (RR - LL) / 2
            virt_data = (d0 - d1) * 0.5
        else:
            # Linear feeds: V = -i(XY - YX) / 2
            virt_data = (-1j * (d0 - d1)) * 0.5

    else:  # P
        # Circular: P = (RL + LR) / 2  [LR ≈ conj(RL)]
        # Linear:   P = (XY + YX) / 2
        virt_data = (d0 + d1) * 0.5

    # A virtual sample is valid only when both constituents are unflagged
    virt_flags = flags[:, :, c0] | flags[:, :, c1]

    return virt_data, virt_flags, constituents
