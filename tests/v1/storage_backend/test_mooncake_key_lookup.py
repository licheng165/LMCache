# SPDX-License-Identifier: Apache-2.0
# Standard
import os
import sys
from types import ModuleType, SimpleNamespace

# First Party
from benchmarks.storage_backend_io.mooncake_key_lookup import _initialize_device


def test_initialize_device_before_mooncake_setup(monkeypatch) -> None:
    observed = []
    module = ModuleType("torch_npu")
    module.npu = SimpleNamespace(set_device=observed.append)
    monkeypatch.setitem(sys.modules, "torch_npu", module)

    _initialize_device("3")

    assert observed == [0]
    assert os.environ["ASCEND_RT_VISIBLE_DEVICES"] == "3"


def test_initialize_device_none_forces_tcp(monkeypatch) -> None:
    monkeypatch.delenv("MC_FORCE_TCP", raising=False)
    _initialize_device("none")
    assert os.environ["MC_FORCE_TCP"] == "1"
