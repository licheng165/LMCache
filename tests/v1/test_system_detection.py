# SPDX-License-Identifier: Apache-2.0
# Standard
from unittest.mock import mock_open, patch

# Third Party
import pytest

# First Party
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.system_detection import NUMADetector


def _shared_cpu_config(
    policy: str = "first_touch",
    nodes: str | int | list[int] | None = None,
) -> LMCacheEngineConfig:
    return LMCacheEngineConfig.from_defaults(
        enable_shared_cpu_cache=True,
        shared_cpu_cache_numa_policy=policy,
        shared_cpu_cache_numa_nodes=nodes,
    )


def test_shared_cpu_numa_default_preserves_first_touch() -> None:
    config = _shared_cpu_config()

    assert NUMADetector.get_shared_cpu_interleave_nodes(config) is None


def test_shared_cpu_numa_interleave_uses_allowed_nodes() -> None:
    config = _shared_cpu_config(policy="interleave")

    with patch(
        "builtins.open",
        mock_open(read_data="Name:\tpython\nMems_allowed_list:\t0-2,4\n"),
    ):
        nodes = NUMADetector.get_shared_cpu_interleave_nodes(config)

    assert nodes == (0, 1, 2, 4)


def test_shared_cpu_numa_interleave_requires_process_allowance() -> None:
    config = _shared_cpu_config(policy="interleave")

    with (
        patch("builtins.open", mock_open(read_data="Name:\tpython\n")),
        pytest.raises(RuntimeError, match="Mems_allowed_list"),
    ):
        NUMADetector.get_shared_cpu_interleave_nodes(config)


def test_shared_cpu_numa_interleave_normalizes_explicit_nodes() -> None:
    config = _shared_cpu_config(policy="interleave", nodes=[2, 0, 2])

    with patch(
        "builtins.open",
        mock_open(read_data="Mems_allowed_list:\t0-3\n"),
    ):
        nodes = NUMADetector.get_shared_cpu_interleave_nodes(config)

    assert nodes == (0, 2)


def test_shared_cpu_numa_interleave_parses_explicit_node_ranges() -> None:
    config = _shared_cpu_config(policy="interleave", nodes="0-2,4")

    with patch(
        "builtins.open",
        mock_open(read_data="Mems_allowed_list:\t0-4\n"),
    ):
        nodes = NUMADetector.get_shared_cpu_interleave_nodes(config)

    assert nodes == (0, 1, 2, 4)


def test_shared_cpu_numa_interleave_rejects_malformed_nodes() -> None:
    config = _shared_cpu_config(policy="interleave", nodes="0,bad")

    with (
        patch(
            "builtins.open",
            mock_open(read_data="Mems_allowed_list:\t0-3\n"),
        ),
        pytest.raises(ValueError, match="must contain NUMA node IDs"),
    ):
        NUMADetector.get_shared_cpu_interleave_nodes(config)


def test_shared_cpu_numa_rejects_nodes_outside_process_allowance() -> None:
    config = _shared_cpu_config(policy="interleave", nodes=[0, 4])

    with (
        patch(
            "builtins.open",
            mock_open(read_data="Mems_allowed_list:\t0-3\n"),
        ),
        pytest.raises(ValueError, match="outside the process"),
    ):
        NUMADetector.get_shared_cpu_interleave_nodes(config)


def test_shared_cpu_numa_rejects_nodes_with_first_touch() -> None:
    config = _shared_cpu_config(nodes=[0])

    with pytest.raises(ValueError, match="requires"):
        NUMADetector.get_shared_cpu_interleave_nodes(config)


def test_shared_cpu_numa_rejects_unknown_policy() -> None:
    config = _shared_cpu_config(policy="bind")

    with pytest.raises(ValueError, match="must be"):
        NUMADetector.get_shared_cpu_interleave_nodes(config)
