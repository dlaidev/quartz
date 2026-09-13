"""CPU examples for the mlai.blog EggEvolve article.

Run from the EggEvolve checkout with its locked dependencies:
  uv run --frozen --with cloudpickle python /path/to/verify_examples.py

These checks do not compile, validate, or benchmark GPU kernels.
"""

from __future__ import annotations

import itertools
import json
import math
from importlib.metadata import version

import numpy as np

from eggevolve.core.e_graph import (
    EGraphWrapper,
    TileConfig,
    create_egraph_for_cdna4,
    create_egraph_for_rdna4,
)


def config_tuple(config):
    return (
        config.block_m,
        config.block_n,
        config.block_k,
        config.num_warps,
        config.num_stages,
    )


def check_config_spaces():
    result = {}
    cases = [
        ("rdna4", create_egraph_for_rdna4,
         ([32, 64, 128], [32, 64, 128], [16], [4, 8], [2, 3])),
        ("cdna4", create_egraph_for_cdna4,
         ([64, 128, 256], [64, 128, 256], [16, 32], [4, 8, 16], [2, 3, 4])),
    ]
    for name, factory, axes in cases:
        graph = factory(TileConfig(64, 64, 16, 4, 2))
        graph.saturate(iterations=20)
        classes = graph.get_equivalent_configs()
        actual = {config_tuple(c) for ec in classes for c in ec.members}
        expected = set(itertools.product(*axes))
        assert actual == expected, (name, len(actual), len(expected))
        assert len(classes) == 1
        result[name] = {"classes": len(classes), "configs": len(actual)}
    graph = EGraphWrapper()
    graph.register_config(TileConfig(64, 64, 16, 4, 2))
    graph.add_rules([
        "(birewrite (config 64 ?n 16 4 2) (config 128 ?n 16 4 2))",
        "(birewrite (config ?m 64 16 4 2) (config ?m 128 16 4 2))",
    ])
    graph.saturate(iterations=10)
    actual = {config_tuple(c) for c in graph.get_all_configs()}
    expected = set(itertools.product([64, 128], [64, 128], [16], [4], [2]))
    assert actual == expected
    result["two_axis_composition"] = sorted(actual)
    return result


def check_online_softmax():
    # Deliberately independent CPU mathematics, not a GPU-kernel emulator.
    scores = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    values = np.array([[1.0], [2.0], [4.0], [8.0]], dtype=np.float64)
    maximum = -math.inf
    denominator = 0.0
    numerator = np.zeros(1, dtype=np.float64)
    trace = []
    for start in range(0, len(scores), 2):
        s = scores[start:start + 2]
        v = values[start:start + 2]
        new_maximum = max(maximum, float(s.max()))
        alpha = math.exp(maximum - new_maximum)
        weights = np.exp(s - new_maximum)
        denominator = alpha * denominator + weights.sum()
        numerator = alpha * numerator + weights @ v
        maximum = new_maximum
        trace.append({"m": maximum, "l": float(denominator),
                      "u": float(numerator[0])})
    weights = np.exp(scores - scores.max())
    reference = (weights @ values) / weights.sum()
    result = numerator / denominator
    np.testing.assert_allclose(result, reference, rtol=1e-14, atol=1e-14)
    return {"trace": trace, "output": float(result[0]),
            "reference": float(reference[0])}


def check_tail_predicates():
    length = 96
    results = []
    for block in [32, 64, 128]:
        wrapper_even = length % 32 == 0
        candidate_even = length % block == 0
        padded_length = math.ceil(length / block) * block
        results.append({"length": length, "block": block,
                        "wrapper_even": wrapper_even,
                        "candidate_even": candidate_even,
                        "unmasked_tail_positions": padded_length - length})
    assert any(r["wrapper_even"] != r["candidate_even"] for r in results)
    return results


def check_storage_arithmetic():
    batch, heads, queries, keys, dim = 1, 32, 4096, 4096, 128
    bytes_per_element = 2
    score_bytes = batch * heads * queries * keys * bytes_per_element
    output_bytes = batch * heads * queries * dim * bytes_per_element
    blocks = math.ceil(keys / 64)
    # T stores, T-1 reloads, then final normalization read + write.
    output_transfer_bytes = (2 * blocks + 1) * output_bytes
    return {"single_score_matrix_gib": score_bytes / 2**30,
            "output_mib": output_bytes / 2**20,
            "kv_blocks": blocks,
            "logical_output_transfer_mib": output_transfer_bytes / 2**20}


def check_floating_point_associativity():
    a, b, c = (np.float32(x) for x in [1e20, -1e20, 3.14])
    left, right = np.float32(np.float32(a + b) + c), np.float32(a + np.float32(b + c))
    assert left != right
    return {"left": float(left), "right": float(right)}


if __name__ == "__main__":
    report = {
        "scope": "CPU checks; no GPU compilation or timing",
        "versions": {name: version(name) for name in ["egglog", "numpy", "cloudpickle"]},
        "config_spaces": check_config_spaces(),
        "online_softmax": check_online_softmax(),
        "tail_predicates": check_tail_predicates(),
        "storage": check_storage_arithmetic(),
        "float32_associativity": check_floating_point_associativity(),
    }
    print(json.dumps(report, indent=2))
