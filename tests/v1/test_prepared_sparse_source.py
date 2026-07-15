# SPDX-License-Identifier: Apache-2.0

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.gpu_connector.sparse import build_prepared_sparse_source


def test_build_prepared_sparse_source_seals_complete_layers() -> None:
    tensors = [
        [torch.zeros(4), torch.ones(2)],
        [torch.full((4,), 2), torch.full((2,), 3)],
    ]
    pointer_tables = [
        torch.tensor([101, 102], dtype=torch.int64),
        torch.tensor([201, 202], dtype=torch.int64),
    ]

    source = build_prepared_sparse_source(
        tensors,
        pointer_tables,
        num_layers=2,
        total_tokens=6,
    )

    assert source is not None
    assert source.total_tokens == 6
    assert source.layers[0].tensors == tuple(tensors[0])
    assert source.layers[1].chunk_ptrs_npu is pointer_tables[1]
    assert len(source.layout_signature) == 2

    same_layout = build_prepared_sparse_source(
        tensors,
        pointer_tables,
        num_layers=2,
        total_tokens=6,
    )
    assert same_layout is not None
    assert same_layout.layout_key is source.layout_key


def test_build_prepared_sparse_source_waits_for_complete_bootstrap() -> None:
    source = build_prepared_sparse_source(
        [[torch.zeros(4)], []],
        [torch.tensor([101], dtype=torch.int64), None],
        num_layers=2,
        total_tokens=4,
    )

    assert source is None


def test_build_prepared_sparse_source_rejects_partial_pointer_coverage() -> None:
    with pytest.raises(ValueError, match="pointer coverage"):
        build_prepared_sparse_source(
            [[torch.zeros(4), torch.ones(2)]],
            [torch.tensor([101], dtype=torch.int64)],
            num_layers=1,
            total_tokens=6,
        )


def test_build_prepared_sparse_source_rejects_noncontiguous_pointer_table() -> None:
    with pytest.raises(ValueError, match="must be contiguous"):
        build_prepared_sparse_source(
            [[torch.zeros(4), torch.ones(2)]],
            [torch.tensor([101, 0, 102, 0], dtype=torch.int64)[::2]],
            num_layers=1,
            total_tokens=6,
        )
