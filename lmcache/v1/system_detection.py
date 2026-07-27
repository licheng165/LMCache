# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from typing import Optional
import platform

# Third Party
import psutil
import torch

if torch.cuda.is_available():
    try:
        # First Party
        from lmcache.c_ops import get_gpu_pci_bus_id
    except ImportError:
        # Fallback if c_ops is not available
        get_gpu_pci_bus_id = None

# First Party
from lmcache.logging import init_logger
from lmcache.v1.config import LMCacheEngineConfig

logger = init_logger(__name__)


@dataclass
class NUMAMapping:
    gpu_to_numa_mapping: dict[int, int]


class SystemMemoryDetector:
    @staticmethod
    def get_available_memory_gb() -> float:
        """
        Get system available memory in GB using psutil.
        This method is cross-platform and doesn't require subprocess calls.

        Returns:
            Available memory in GB, or 0.0 if detection fails.
        """
        try:
            # Use psutil to get virtual memory information
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)

            system = platform.system()
            logger.info(f"{system} system available memory: {available_gb:.2f} GB")
            return available_gb

        except Exception as e:
            logger.warning(f"Failed to get system available memory using psutil: {e}")
            return 0.0


class NUMADetector:
    @staticmethod
    def get_shared_cpu_interleave_nodes(
        config: LMCacheEngineConfig,
    ) -> Optional[tuple[int, ...]]:
        """Resolve NUMA nodes for shared CPU cache interleaving.

        Args:
            config: LMCache engine configuration.

        Returns:
            The selected NUMA nodes for ``MPOL_INTERLEAVE``, or ``None`` to
            preserve the existing first-touch allocation policy.

        Raises:
            ValueError: If the policy or configured node list is invalid.
            RuntimeError: If interleaving is requested and the process
                memory-node allowance cannot be detected.
        """
        policy = str(
            config.get_extra_config_value(
                "shared_cpu_cache_numa_policy",
                getattr(config, "shared_cpu_cache_numa_policy", "first_touch"),
            )
        ).strip().lower()
        configured_nodes = config.get_extra_config_value(
            "shared_cpu_cache_numa_nodes",
            getattr(config, "shared_cpu_cache_numa_nodes", None),
        )
        if policy not in {"first_touch", "interleave"}:
            raise ValueError(
                "shared_cpu_cache_numa_policy must be 'first_touch' or "
                f"'interleave', got {policy!r}"
            )
        if policy == "first_touch":
            if configured_nodes is not None:
                raise ValueError(
                    "shared_cpu_cache_numa_nodes requires "
                    "shared_cpu_cache_numa_policy='interleave'"
                )
            return None

        allowed_nodes = NUMADetector._read_allowed_numa_nodes()
        if not allowed_nodes:
            raise RuntimeError(
                "shared CPU cache NUMA interleave policy could not detect "
                "the process Mems_allowed_list"
            )
        if configured_nodes is None:
            return allowed_nodes

        try:
            if isinstance(configured_nodes, str):
                nodes = NUMADetector._parse_numa_node_list(configured_nodes)
            elif isinstance(configured_nodes, int):
                nodes = (configured_nodes,)
            else:
                nodes = tuple(sorted({int(node) for node in configured_nodes}))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "shared_cpu_cache_numa_nodes must contain NUMA node IDs"
            ) from exc
        if not nodes or any(node < 0 for node in nodes):
            raise ValueError(
                "shared_cpu_cache_numa_nodes must contain non-negative NUMA "
                "node IDs"
            )
        if not set(nodes).issubset(allowed_nodes):
            raise ValueError(
                "shared_cpu_cache_numa_nodes contains nodes outside the "
                f"process Mems_allowed_list: nodes={nodes}, "
                f"allowed={allowed_nodes}"
            )
        return nodes

    @staticmethod
    def get_numa_mapping(config: LMCacheEngineConfig) -> Optional[NUMAMapping]:
        """
        Get NUMA mapping.
        """
        assert config.numa_mode in ["manual", "auto", None], (
            "NUMA mode must be either 'auto',  'manual', or None."
            f" Current mode: {config.numa_mode}"
        )

        numa_mapping: Optional[NUMAMapping] = None
        if config.numa_mode == "manual":
            numa_mapping = NUMADetector._read_from_config(config)
        elif config.numa_mode == "auto":
            numa_mapping = NUMADetector._read_from_sys()

        return numa_mapping

    @staticmethod
    def _read_from_config(config) -> NUMAMapping:
        """
        Read NUMA mapping from the LMCache configuration.
        """

        assert config.extra_config is not None, (
            "NUMA mode is set but extra_config is None. "
            "Please ensure the configuration is properly set."
        )

        assert "gpu_to_numa_mapping" in config.extra_config, (
            "NUMA mode is set to `manual` but gpu_to_numa_mapping is None. "
            "Please ensure the configuration is properly set."
        )

        gpu_to_numa_mapping = config.extra_config.get("gpu_to_numa_mapping")

        return NUMAMapping(gpu_to_numa_mapping)

    @staticmethod
    def _read_from_sys() -> Optional[NUMAMapping]:
        """
        Read NUMA mapping from system configuration.
        """

        try:
            device_index = torch.cuda.current_device()
            pci_bus_id = get_gpu_pci_bus_id(device_index).lower()

            numa_node_file = f"/sys/bus/pci/devices/{pci_bus_id}/numa_node"
            with open(numa_node_file) as f:
                numa_node = int(f.read())

            return NUMAMapping(gpu_to_numa_mapping={device_index: numa_node})
        except Exception as e:
            logger.warning(f"Failed to auto read NUMA mapping from system: {e}")
            return None

    @staticmethod
    def _read_allowed_numa_nodes() -> Optional[tuple[int, ...]]:
        try:
            with open("/proc/self/status") as status_file:
                for line in status_file:
                    if line.startswith("Mems_allowed_list:"):
                        return NUMADetector._parse_numa_node_list(
                            line.partition(":")[2]
                        )
        except OSError as exc:
            logger.warning("Failed to read process NUMA allowance: %s", exc)
        return None

    @staticmethod
    def _parse_numa_node_list(value: str) -> tuple[int, ...]:
        nodes: set[int] = set()
        try:
            for item in value.strip().split(","):
                if not item:
                    continue
                if "-" not in item:
                    nodes.add(int(item))
                    continue
                first, last = (int(part) for part in item.split("-", 1))
                if first > last:
                    raise ValueError
                nodes.update(range(first, last + 1))
        except ValueError as exc:
            raise ValueError(f"Invalid NUMA node list: {value!r}") from exc
        return tuple(sorted(nodes))
