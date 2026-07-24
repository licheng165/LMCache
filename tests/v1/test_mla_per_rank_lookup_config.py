# SPDX-License-Identifier: Apache-2.0
"""Regression tests for MLA per-rank prefix cache lookup configuration."""

# Standard
from types import SimpleNamespace

# Third Party
import pytest

# First Party
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.lookup_client.factory import LookupClientFactory


def test_mla_save_only_first_rank_false_uses_all_lookup_servers() -> None:
    config = LMCacheEngineConfig.from_defaults(
        extra_config={"save_only_first_rank": False},
    )
    assert config.get_lookup_server_worker_ids(use_mla=True, world_size=8) == list(
        range(8)
    )


def test_mla_save_only_first_rank_true_uses_rank0_lookup_only() -> None:
    config = LMCacheEngineConfig.from_defaults(
        extra_config={"save_only_first_rank": True},
    )
    assert config.get_lookup_server_worker_ids(use_mla=True, world_size=8) == [0]


def test_mla_save_only_first_rank_default_is_rank0_only() -> None:
    config = LMCacheEngineConfig.from_defaults()
    assert config.get_lookup_server_worker_ids(use_mla=True, world_size=8) == [0]


def test_mla_per_rank_store_rejects_single_rank_external_lookup() -> None:
    config = LMCacheEngineConfig.from_defaults(
        external_lookup_client="mooncakestore://127.0.0.1:50051",
        extra_config={"save_only_first_rank": False},
    )
    metadata = SimpleNamespace(use_mla=True, world_size=8)

    with pytest.raises(ValueError, match="internal all-rank lookup"):
        LookupClientFactory.create_lookup_client(config, metadata)
