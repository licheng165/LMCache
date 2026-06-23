# SPDX-License-Identifier: Apache-2.0

# Standard
from types import SimpleNamespace

# Third Party
import torch

# First Party
from lmcache.v1.storage_backend.connector.base_connector import (
    layerwise_connector_meta_shapes,
    resolve_save_chunk_meta,
)


def test_resolve_save_chunk_meta_explicit_false_overrides_layerwise() -> None:
    config = SimpleNamespace(
        extra_config={"save_chunk_meta": False},
        use_layerwise=True,
    )
    assert resolve_save_chunk_meta(config) is False


def test_resolve_save_chunk_meta_layerwise_default_true() -> None:
    config = SimpleNamespace(extra_config={}, use_layerwise=True)
    assert resolve_save_chunk_meta(config) is True


def test_resolve_save_chunk_meta_non_layerwise_default_true() -> None:
    config = SimpleNamespace(extra_config={}, use_layerwise=False)
    assert resolve_save_chunk_meta(config) is True


def test_layerwise_connector_meta_shapes() -> None:
    shapes = [torch.Size([1, 32, 256, 512])]
    adjusted = layerwise_connector_meta_shapes(shapes)
    assert adjusted == [torch.Size([1, 1, 256, 512])]
