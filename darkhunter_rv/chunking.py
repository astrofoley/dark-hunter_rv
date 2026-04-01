"""Split orders into wavelength/pixel sub-chunks for robust RV."""
from __future__ import annotations

from typing import Any, Dict, Iterator, List, Tuple

import numpy as np


def iter_order_chunks(
    spec_data: Dict[int, Dict[str, List]],
    bad_orders: List[int],
    subchunks: int,
) -> Iterator[Tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """Yield (chunk_key, wave, flux, eflux). chunk_key is \"{order}\" or \"{order}_{si}\"."""
    for order in sorted(spec_data.keys()):
        if order in bad_orders:
            continue
        data = spec_data[order]
        w = np.array(data["wavelength"], float)
        f = np.array(data["flux"], float)
        e = np.array(data["eflux"], float)
        if len(w) < 5:
            continue
        if subchunks <= 1:
            yield str(order), w, f, e
            continue
        edges = np.linspace(0, len(w), subchunks + 1, dtype=int)
        for si in range(subchunks):
            i0, i1 = int(edges[si]), int(edges[si + 1])
            if i1 - i0 < 5:
                continue
            yield f"{order}_{si}", w[i0:i1], f[i0:i1], e[i0:i1]
