# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional, Union

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.lookup_client.abstract_client import LookupClientInterface
from lmcache.v1.metadata import LMCacheMetadata

logger = init_logger(__name__)


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
        self.store = MooncakeDistributedStore()
        self.store.setup(
            "localhost",
            "P2PHANDSHAKE",
            0,
            16 * 1024 * 1024,
            "tcp",
            "",
            master_addr,
        )

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
        # process token_ids to cacheengine keys
        keys = []
        ends = []
        chunk_key_counts = []
        use_layerwise = bool(
            getattr(getattr(self, "config", None), "use_layerwise", False)
        )
        dsa_two_groups = bool(
            getattr(getattr(self, "config", None), "dsa_two_groups", False)
        )
        num_layers = int(
            getattr(getattr(self, "metadata", None), "kv_shape", (1,))[0]
        )

        for start, end, key in self.token_database.process_tokens(
            token_ids, request_configs=request_configs
        ):
            assert isinstance(key, CacheEngineKey)
            group_keys = [key]
            if dsa_two_groups:
                make_key = getattr(self.token_database, "_make_key_by_hash")
                index_key = make_key(
                    key.chunk_hash,
                    request_configs,
                    kv_group=1,
                )
                group_keys.append(index_key)

            chunk_keys = []
            for group_key in group_keys:
                if use_layerwise:
                    chunk_keys.extend(
                        layer_key.to_string()
                        for layer_key in group_key.split_layers(num_layers)
                    )
                else:
                    chunk_keys.append(group_key.to_string())
            keys.extend(chunk_keys)
            chunk_key_counts.append(len(chunk_keys))
            ends.append(end)

        # Use batch_is_exist to check all keys at once
        # rets is list of int: 1 = found, 0 = not found, -1 = error
        rets = self.store.batch_is_exist(keys)

        # Find the first key that doesn't exist (ret != 1)
        # This follows the same logic as cache engine's lookup method
        offset = 0
        for chunk_idx, key_count in enumerate(chunk_key_counts):
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
        # nothing here
        pass
