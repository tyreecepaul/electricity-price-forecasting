"""Fixed NEM topology: regions are nodes, interconnectors are edges."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

DEFAULT_REGIONS: tuple[str, ...] = ("NSW1", "QLD1", "VIC1", "SA1", "TAS1")

# Keys are the ``ic_<id>`` column prefixes produced by preprocessing, so edge
# attributes can later be looked up from the same table.
NEM_INTERCONNECTORS: dict[str, tuple[str, str]] = {
    "ic_nsw1_qld1": ("NSW1", "QLD1"),  # QNI
    "ic_n_q_mnsp1": ("NSW1", "QLD1"),  # Terranora
    "ic_vic1_nsw1": ("VIC1", "NSW1"),  # VNI
    "ic_v_sa": ("VIC1", "SA1"),  # Heywood
    "ic_v_s_mnsp1": ("VIC1", "SA1"),  # Murraylink
    "ic_t_v_mnsp1": ("TAS1", "VIC1"),  # Basslink
}


def adjacency(regions: Sequence[str] = DEFAULT_REGIONS, mode: str = "nem") -> np.ndarray:
    """Return a symmetric 0/1 matrix with self-loops in the given node order.

    ``identity`` keeps only the self-loops. It is the control that shows
    whether cross-region edges, rather than the encoder alone, add value.
    """
    if mode not in {"nem", "identity"}:
        raise ValueError("mode must be 'nem' or 'identity'")
    position = {region: index for index, region in enumerate(regions)}
    if len(position) != len(regions):
        raise ValueError("regions must be distinct")
    matrix = np.eye(len(regions), dtype=np.float32)
    if mode == "nem":
        for first, second in NEM_INTERCONNECTORS.values():
            if first in position and second in position:
                matrix[position[first], position[second]] = 1.0
                matrix[position[second], position[first]] = 1.0
    return matrix
