import pytest
import json
from pathlib import Path

import pandas as pd

from validation import build_bias_set, evaluate_method_consistency, calibrate_error_model


def _make_orders(tmp: Path):
    d = tmp / 'orders'
    d.mkdir()
    (d / 'a_orders.txt').write_text('#\n1 10.0 1.0\n2 11.0 1.0\n')
    (d / 'b_orders.txt').write_text('#\n1 10.4 1.0\n2 11.4 1.0\n')
    return d


@pytest.mark.validation
def test_build_bias_set(tmp_path):
    inp = _make_orders(tmp_path)
    out = tmp_path / 'out'
    s = build_bias_set.build_bias(inp, n_boot=20, out_dir=out)
    assert (out / 'bias_by_chunk.csv').exists()
    assert (out / 'bias_statistics.txt').exists()
    assert s['n_files'] == 2
    assert s['n_chunk_keys'] >= 1


@pytest.mark.validation
def test_method_consistency_and_error_model(tmp_path):
    d = pd.DataFrame([
        {'file':'a','chunk_key':'1','method':'mask_ccf','rv_kms':10.0,'rv_err_kms':1.0,'teff':5000,'telluric_fraction':0.1,'mask_line_count':20},
        {'file':'a','chunk_key':'1','method':'template_fft','rv_kms':10.2,'rv_err_kms':1.0,'teff':5000,'telluric_fraction':0.1,'mask_line_count':20},
        {'file':'b','chunk_key':'1','method':'mask_ccf','rv_kms':11.0,'rv_err_kms':1.0,'teff':5000,'telluric_fraction':0.1,'mask_line_count':20},
        {'file':'b','chunk_key':'1','method':'template_fft','rv_kms':11.1,'rv_err_kms':1.0,'teff':5000,'telluric_fraction':0.1,'mask_line_count':20},
    ])
    fp = tmp_path / 'diag.csv'
    d.to_csv(fp, index=False)

    out1 = tmp_path / 'cons'
    s1 = evaluate_method_consistency.run(d, out1)
    assert s1['n_pairs'] >= 1
    assert (out1 / 'method_pair_offsets.csv').exists()

    out2 = tmp_path / 'err'
    s2 = calibrate_error_model.calibrate(d, out2)
    assert s2['n_methods'] >= 1
    assert (out2 / 'systematic_floors.csv').exists()
