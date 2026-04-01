import numpy as np

from darkhunter_rv.chunking import iter_order_chunks


def test_iter_order_chunks_subchunks():
    spec = {0: {"wavelength": list(range(100)), "flux": [1.0] * 100, "eflux": [1.0] * 100}}
    chunks = list(iter_order_chunks(spec, bad_orders=[], subchunks=4))
    keys = [k for k, *_ in chunks]
    assert keys == ["0_0", "0_1", "0_2", "0_3"]
    assert sum(len(w) for _, w, *_ in chunks) == 100


def test_iter_order_chunks_skip_bad_orders():
    spec = {
        0: {"wavelength": [1, 2, 3, 4, 5], "flux": [1, 1, 1, 1, 1], "eflux": [1, 1, 1, 1, 1]},
        1: {"wavelength": [1, 2, 3, 4, 5], "flux": [1, 1, 1, 1, 1], "eflux": [1, 1, 1, 1, 1]},
    }
    chunks = list(iter_order_chunks(spec, bad_orders=[1], subchunks=1))
    assert [k for k, *_ in chunks] == ["0"]
