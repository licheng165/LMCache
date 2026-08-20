# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional, Union

# Third Party
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.kv_layer_groups import (
    parse_declared_kv_group_layers,
    resolve_kv_group_num_layers,
)
from lmcache.v1.lookup_client.abstract_client import LookupClientInterface
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.mooncake_key_trace import trace_mooncake_keys
from lmcache.v1.mooncake_layout import (
    mooncake_page_key,
    mooncake_page_layout_enabled,
)
from lmcache.v1.sampled_lookup import (
    find_last_sampled_hit,
    first_last_layer_keys,
)

class MooncakeLookupClient(LookupClientInterface):
    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        master_addr: str,
    ):
        # Third Party
        from mooncake.store import MooncakeDistributedStore

        self.config = config
        self.metadata = metadata
        extra_config = getattr(config, "extra_config", None) or {}
        self.declared_kv_group_layers = parse_declared_kv_group_layers(
            extra_config.get("kv_group_layers", None)
        )
        self.store = MooncakeDistributedStore()
        status = self.store.setup(
            "localhost",
            "P2PHANDSHAKE",
            0,
            0,
            "tcp",
            "",
            master_addr,
        )
        if status not in (None, 0):
            self.store.close()
            raise RuntimeError(f"Mooncake lookup setup failed: status={status}")

        # Initialize token database for processing tokens
        assert isinstance(config, LMCacheEngineConfig), (
            "LMCache v1 configuration is should be passed."
        )

        # First Party
        from lmcache.v1.token_database import ChunkedTokenDatabase

        assert not config.enable_blending, (
            "LMCache v1 blending is not supported in MooncakeLookupClient yet."
        )
        self.token_database = ChunkedTokenDatabase(config, metadata)

    def lookup(
        self,
        token_ids: Union[torch.Tensor, list[int]],
        lookup_id: Optional[str] = None,
        request_configs: Optional[dict] = None,
    ) -> Optional[int]:
        """Return the cached prefix length.

        In experimental sampled mode the result is only a candidate under the
        contiguous-prefix contract. Unlike the production lookup-server path,
        this standalone client does not pin and revalidate intermediate chunks.
        """
        # process token_ids to cacheengine keys
        ends = []
        chunk_keys_by_chunk: list[list[str]] = []
        use_layerwise = bool(
            getattr(getattr(self, "config", None), "use_layerwise", False)
        )
        dsa_two_groups = bool(
            getattr(getattr(self, "config", None), "dsa_two_groups", False)
        )
        num_layers = int(
            getattr(getattr(self, "metadata", None), "kv_shape", (1,))[0]
        )
        registered_groups = getattr(
            getattr(
                getattr(self, "metadata", None),
                "kv_layer_groups_manager",
                None,
            ),
            "kv_layer_groups",
            None,
        ) or []

        def num_layers_for(group_key: CacheEngineKey) -> int:
            return resolve_kv_group_num_layers(
                kv_group=int(getattr(group_key, "kv_group", 0) or 0),
                dsa_two_groups=dsa_two_groups,
                model_num_layers=num_layers,
                registered_groups=registered_groups,
                declared=getattr(self, "declared_kv_group_layers", None),
            )

        sampled_lookup = bool(
            use_layerwise
            and getattr(self.config, "experimental_sampled_layerwise_lookup", False)
        )
        page_first = use_layerwise and mooncake_page_layout_enabled(self.config)

        for start, end, key in self.token_database.process_tokens(
            token_ids, request_configs=request_configs
        ):
            assert isinstance(key, CacheEngineKey)
            group_keys = [key]
            if dsa_two_groups:
                make_key = self.token_database._make_key_by_hash
                index_key = make_key(
                    key.chunk_hash,
                    request_configs,
                    kv_group=1,
                )
                group_keys.append(index_key)

            if page_first and end - start == self.config.chunk_size:
                chunk_keys = [
                    mooncake_page_key(group_key, num_layers_for(group_key))
                    for group_key in group_keys
                ]
            elif sampled_lookup:
                sampled_keys = []
                for group_key in group_keys:
                    sampled_keys.extend(
                        first_last_layer_keys(
                            [group_key], num_layers_for(group_key)
                        )
                    )
                chunk_keys = [
                    key.to_string() for key in sampled_keys
                ]
            elif use_layerwise:
                chunk_keys = [
                    layer_key.to_string()
                    for group_key in group_keys
                    for layer_key in group_key.split_layers(
                        num_layers_for(group_key)
                    )
                ]
            else:
                chunk_keys = [group_key.to_string() for group_key in group_keys]
            chunk_keys_by_chunk.append(chunk_keys)
            ends.append(end)

        if sampled_lookup:
            def batch_exists(keys: list[str]) -> bool:
                if not keys:
                    return False
                results = self.store.batch_is_exist(keys)
                trace_mooncake_keys(
                    "lookup",
                    keys,
                    results,
                    api="MooncakeLookupClient.batch_is_exist",
                    lookup_id=lookup_id,
                )
                return len(results) == len(keys) and all(
                    result == 1 for result in results
                )

            winner = find_last_sampled_hit(
                len(chunk_keys_by_chunk),
                lambda index: batch_exists(chunk_keys_by_chunk[index]),
            )
            return 0 if winner is None else ends[winner]

        # Use batch_is_exist to check all keys at once
        # rets is list of int: 1 = found, 0 = not found, -1 = error
        keys = [key for chunk_keys in chunk_keys_by_chunk for key in chunk_keys]
        rets = self.store.batch_is_exist(keys)
        trace_mooncake_keys(
            "lookup",
            keys,
            rets,
            api="MooncakeLookupClient.batch_is_exist",
            lookup_id=lookup_id,
        )

        # Find the first key that doesn't exist (ret != 1)
        # This follows the same logic as cache engine's lookup method
        offset = 0
        for chunk_idx, chunk_keys in enumerate(chunk_keys_by_chunk):
            key_count = len(chunk_keys)
            chunk_rets = rets[offset : offset + key_count]
            offset += key_count
            if len(chunk_rets) < key_count or any(
                ret != 1 for ret in chunk_rets
            ):
                # Return the end position of the previous chunk
                # If chunk_idx == 0, no chunks were found, return 0
                return ends[chunk_idx - 1] if chunk_idx > 0 else 0

        # All keys were found, return the last end position
        return ends[-1] if ends else 0

    def supports_producer_reuse(self) -> bool:
        """Return True as MooncakeLookupClient supports producer kvcache reuse"""
        return True

    def close(self):
        self.store.close()
