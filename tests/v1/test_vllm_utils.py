# SPDX-License-Identifier: Apache-2.0

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.utils import is_false


@pytest.mark.parametrize(
    ("value", "enabled"),
    [
        ("false", False),
        (" FALSE ", False),
        ("0", False),
        ("off", False),
        ("true", True),
        ("1", True),
    ],
)
def test_force_skip_save_environment_parsing(value: str, enabled: bool) -> None:
    assert (not is_false(value)) is enabled
