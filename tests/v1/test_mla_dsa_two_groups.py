# SPDX-License-Identifier: Apache-2.0
"""Unit and regression tests for the MLA+DSA two-group separate storage design
(NPU shared side).

Covers the NPU-side changes:
- CacheEngineKey / LayerCacheEngineKey kv_group field (hash, eq, to/from_string,
  to_dict/from_dict, with_new_worker_id, split_layers)
- token_database.process_tokens kv_group propagation
- new config knobs (save_full_chunk_in_decode, save_indexer_only_first_rank,
  dsa_two_groups)
- MemoryFormat new tags (KV_MLA_LATENT_FMT, KV_DSA_INDEX_FMT)
- Regression: existing key serialization and config defaults unchanged

Format detection tests (KVCacheFormat) are in the Ascend test file since
KVCacheFormat is defined in the lmcache_ascend package.
"""
# Standard
import os

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey, LayerCacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.token_database import ChunkedTokenDatabase

# Local
from .utils import dumb_metadata, generate_tokens


# ---------------------------------------------------------------------------
# CacheEngineKey / LayerCacheEngineKey kv_group tests
# ---------------------------------------------------------------------------

class TestCacheEngineKeyKvGroup:

    def test_default_kv_group_is_zero(self):
        key = CacheEngineKey(
            model_name="test", world_size=1, worker_id=0,
            chunk_hash=42, dtype=torch.bfloat16,
        )
        assert key.kv_group == 0

    def test_kv_group_set_explicitly(self):
        key = CacheEngineKey(
            model_name="test", world_size=1, worker_id=0,
            chunk_hash=42, dtype=torch.bfloat16, kv_group=1,
        )
        assert key.kv_group == 1

    def test_keys_with_different_kv_group_are_not_equal(self):
        k0 = CacheEngineKey("test", 1, 0, 42, torch.bfloat16, kv_group=0)
        k1 = CacheEngineKey("test", 1, 0, 42, torch.bfloat16, kv_group=1)
        assert k0 != k1
        assert hash(k0) != hash(k1)

    def test_keys_with_same_kv_group_are_equal(self):
        k0a = CacheEngineKey("test", 1, 0, 42, torch.bfloat16, kv_group=0)
        k0b = CacheEngineKey("test", 1, 0, 42, torch.bfloat16, kv_group=0)
        assert k0a == k0b
        assert hash(k0a) == hash(k0b)

    def test_to_string_includes_kv_group(self):
        key = CacheEngineKey("test", 1, 0, 42, torch.bfloat16, kv_group=1)
        s = key.to_string()
        parts = s.split("@")
        # model@ws@wid@hash@dtype@kv_group
        assert parts[5] == "1"

    def test_from_string_parses_kv_group(self):
        key = CacheEngineKey("test", 1, 0, 42, torch.bfloat16, kv_group=1)
        s = key.to_string()
        parsed = CacheEngineKey.from_string(s)
        assert parsed.kv_group == 1
        assert parsed == key

    def test_from_string_backward_compat_no_kv_group(self):
        """Old keys without kv_group should parse with kv_group=0."""
        # Simulate old format: model@ws@wid@hash@dtype (no kv_group)
        old_str = "test@1@0@2a@bfloat16"
        parsed = CacheEngineKey.from_string(old_str)
        assert parsed.kv_group == 0
        assert parsed.model_name == "test"
        assert parsed.world_size == 1

    def test_from_string_backward_compat_with_tags(self):
        """Old keys with tags but no kv_group should parse correctly."""
        old_str = "test@1@0@2a@bfloat16@tag1%val1@tag2%val2"
        parsed = CacheEngineKey.from_string(old_str)
        assert parsed.kv_group == 0
        assert parsed.tags == (("tag1", "val1"), ("tag2", "val2"))

    def test_to_from_string_roundtrip_with_tags_and_kv_group(self):
        key = CacheEngineKey(
            "test", 2, 1, 99, torch.bfloat16, kv_group=1,
            request_configs={"lmcache.tag.foo": "bar"},
        )
        s = key.to_string()
        parsed = CacheEngineKey.from_string(s)
        assert parsed == key
        assert parsed.kv_group == 1
        assert parsed.tags == (("foo", "bar"),)

    def test_to_dict_includes_kv_group(self):
        key = CacheEngineKey("test", 1, 0, 42, torch.bfloat16, kv_group=1)
        d = key.to_dict()
        assert d["kv_group"] == 1

    def test_from_dict_parses_kv_group(self):
        key = CacheEngineKey("test", 1, 0, 42, torch.bfloat16, kv_group=1)
        d = key.to_dict()
        restored = CacheEngineKey.from_dict(d)
        assert restored == key
        assert restored.kv_group == 1

    def test_from_dict_default_kv_group_zero(self):
        """Old dicts without kv_group should default to 0."""
        d = {
            "__type__": "CacheEngineKey",
            "model_name": "test",
            "world_size": 1,
            "worker_id": 0,
            "chunk_hash": 42,
            "dtype": "bfloat16",
        }
        restored = CacheEngineKey.from_dict(d)
        assert restored.kv_group == 0

    def test_with_new_worker_id_preserves_kv_group(self):
        key = CacheEngineKey("test", 1, 0, 42, torch.bfloat16, kv_group=1)
        new_key = key.with_new_worker_id(2)
        assert new_key.kv_group == 1
        assert new_key.worker_id == 2


class TestLayerCacheEngineKeyKvGroup:

    def test_layer_key_kv_group(self):
        key = LayerCacheEngineKey(
            "test", 1, 0, 42, torch.bfloat16, layer_id=5, kv_group=1,
        )
        assert key.kv_group == 1
        assert key.layer_id == 5

    def test_layer_keys_different_kv_group_not_equal(self):
        k0 = LayerCacheEngineKey("test", 1, 0, 42, torch.bfloat16, layer_id=3, kv_group=0)
        k1 = LayerCacheEngineKey("test", 1, 0, 42, torch.bfloat16, layer_id=3, kv_group=1)
        assert k0 != k1
        assert hash(k0) != hash(k1)

    def test_layer_key_to_string_includes_kv_group_and_layer(self):
        key = LayerCacheEngineKey("test", 1, 0, 42, torch.bfloat16, layer_id=3, kv_group=1)
        s = key.to_string()
        parts = s.split("@")
        # model@ws@wid@hash@dtype@kv_group@layer_id
        assert parts[5] == "1"
        assert parts[6] == "3"

    def test_layer_key_from_string_new_format(self):
        key = LayerCacheEngineKey("test", 1, 0, 42, torch.bfloat16, layer_id=3, kv_group=1)
        s = key.to_string()
        parsed = LayerCacheEngineKey.from_string(s)
        assert parsed.kv_group == 1
        assert parsed.layer_id == 3
        assert parsed == key

    def test_layer_key_from_string_old_format_no_kv_group(self):
        """Old format: model@ws@wid@hash@dtype@layer_id (no kv_group)."""
        old_str = "test@1@0@2a@bfloat16@5"
        parsed = LayerCacheEngineKey.from_string(old_str)
        assert parsed.kv_group == 0
        assert parsed.layer_id == 5

    def test_layer_key_from_string_old_format_with_tags(self):
        """Old format: model@ws@wid@hash@dtype@layer_id@tags."""
        old_str = "test@1@0@2a@bfloat16@5@foo%bar"
        parsed = LayerCacheEngineKey.from_string(old_str)
        assert parsed.kv_group == 0
        assert parsed.layer_id == 5
        assert parsed.tags == (("foo", "bar"),)

    def test_split_layers_preserves_kv_group(self):
        key = CacheEngineKey("test", 1, 0, 42, torch.bfloat16, kv_group=1)
        layer_keys = key.split_layers(4)
        assert len(layer_keys) == 4
        for lk in layer_keys:
            assert lk.kv_group == 1
            assert isinstance(lk, LayerCacheEngineKey)


# ---------------------------------------------------------------------------
# Config knob tests
# ---------------------------------------------------------------------------

class TestConfigKnobs:

    def test_save_full_chunk_in_decode_default_false(self):
        config = LMCacheEngineConfig.from_defaults()
        assert config.save_full_chunk_in_decode is False

    def test_save_full_chunk_in_decode_from_env(self):
        os.environ["LMCACHE_SAVE_FULL_CHUNK_IN_DECODE"] = "true"
        try:
            config = LMCacheEngineConfig.from_env()
            assert config.save_full_chunk_in_decode is True
        finally:
            del os.environ["LMCACHE_SAVE_FULL_CHUNK_IN_DECODE"]

    def test_save_indexer_only_first_rank_default_false(self):
        config = LMCacheEngineConfig.from_defaults()
        assert config.save_indexer_only_first_rank is False

    def test_dsa_two_groups_default_false(self):
        config = LMCacheEngineConfig.from_defaults()
        assert config.dsa_two_groups is False

    def test_dsa_two_groups_from_env(self):
        os.environ["LMCACHE_DSA_TWO_GROUPS"] = "1"
        try:
            config = LMCacheEngineConfig.from_env()
            assert config.dsa_two_groups is True
        finally:
            del os.environ["LMCACHE_DSA_TWO_GROUPS"]

    # --- Regression: existing knobs unchanged ---

    def test_regression_save_decode_cache_default(self):
        config = LMCacheEngineConfig.from_defaults()
        assert config.save_decode_cache is False

    def test_regression_save_unfull_chunk_default(self):
        config = LMCacheEngineConfig.from_defaults()
        assert config.save_unfull_chunk is False

    def test_regression_use_layerwise_default(self):
        config = LMCacheEngineConfig.from_defaults()
        assert config.use_layerwise is False


# ---------------------------------------------------------------------------
# Token database kv_group tests
# ---------------------------------------------------------------------------

class TestTokenDatabaseKvGroup:

    def test_process_tokens_default_kv_group_zero(self):
        cfg = LMCacheEngineConfig.from_legacy(chunk_size=64, backend="cpu")
        metadata = dumb_metadata()
        tokens = generate_tokens(128, "cpu")
        db = ChunkedTokenDatabase(cfg, metadata)
        results = list(db.process_tokens(tokens=tokens))
        assert len(results) > 0
        for _, _, key in results:
            assert key.kv_group == 0

    def test_process_tokens_kv_group_one(self):
        cfg = LMCacheEngineConfig.from_legacy(chunk_size=64, backend="cpu")
        metadata = dumb_metadata()
        tokens = generate_tokens(128, "cpu")
        db = ChunkedTokenDatabase(cfg, metadata)
        results = list(db.process_tokens(tokens=tokens, kv_group=1))
        assert len(results) > 0
        for _, _, key in results:
            assert key.kv_group == 1

    def test_process_tokens_different_kv_groups_produce_different_keys(self):
        cfg = LMCacheEngineConfig.from_legacy(chunk_size=64, backend="cpu")
        metadata = dumb_metadata()
        tokens = generate_tokens(128, "cpu")
        db = ChunkedTokenDatabase(cfg, metadata)
        results_0 = list(db.process_tokens(tokens=tokens, kv_group=0))
        results_1 = list(db.process_tokens(tokens=tokens, kv_group=1))
        assert len(results_0) == len(results_1)
        for (_, _, k0), (_, _, k1) in zip(results_0, results_1):
            assert k0 != k1
            assert k0.kv_group == 0
            assert k1.kv_group == 1

    def test_split_layers_with_kv_group(self):
        cfg = LMCacheEngineConfig.from_legacy(chunk_size=64, backend="cpu")
        metadata = dumb_metadata(kv_shape=(4, 2, 64, 8, 128))
        tokens = generate_tokens(64, "cpu")
        db = ChunkedTokenDatabase(cfg, metadata)
        results = list(db.process_tokens(tokens=tokens, kv_group=1))
        assert len(results) == 1
        _, _, key = results[0]
        layer_keys = key.split_layers(4)
        assert len(layer_keys) == 4
        for lk in layer_keys:
            assert lk.kv_group == 1


# ---------------------------------------------------------------------------
# MemoryFormat tests
# ---------------------------------------------------------------------------

class TestMemoryFormat:

    def test_new_formats_exist(self):
        assert hasattr(MemoryFormat, "KV_MLA_LATENT_FMT")
        assert hasattr(MemoryFormat, "KV_DSA_INDEX_FMT")

    def test_mla_latent_is_alias_of_mla_fmt(self):
        assert MemoryFormat.KV_MLA_LATENT_FMT is MemoryFormat.KV_MLA_FMT

    def test_new_formats_have_token_dim(self):
        assert MemoryFormat.KV_MLA_LATENT_FMT.token_dim() == 2
        assert MemoryFormat.KV_DSA_INDEX_FMT.token_dim() == 2

    def test_regression_existing_formats_token_dim(self):
        assert MemoryFormat.KV_2LTD.token_dim() == 2
        assert MemoryFormat.KV_T2D.token_dim() == 1
        assert MemoryFormat.KV_MLA_FMT.token_dim() == 2
