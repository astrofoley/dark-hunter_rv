import numpy as np

from darkhunter_rv.chunking import iter_order_chunks, iter_order_chunks_with_edges


from darkhunter_rv import chunking
from darkhunter_rv.chunking import iter_order_chunks, iter_order_chunks_with_edges


def test_parse_chunk_key_merge():
    assert chunking.parse_chunk_key("merge_3") == (None, 3, "merge")
    assert chunking.bias_order_from_chunk_key("merge_3") is None
    assert chunking.parse_chunk_key("15_2")[0] == 15
    assert chunking.chunk_sort_key("15") < chunking.chunk_sort_key("merge_1")


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


def test_iter_order_chunks_with_edges_three_chunks():
    spec = {2: {"wavelength": list(range(30)), "flux": [1.0] * 30, "eflux": [1.0] * 30}}
    chunks = list(
        iter_order_chunks_with_edges(spec, bad_orders=[], pixel_edges=[0.0, 1 / 3, 2 / 3, 1.0])
    )
    assert [k for k, *_ in chunks] == ["2_0", "2_1", "2_2"]
