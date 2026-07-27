# SPDX-License-Identifier: Apache-2.0
from collections import deque


def publish_delayed_decode_window_commit(
    pending_commits: deque[int],
    completed_end: int,
    delay_windows: int,
    *,
    is_initial_frontier: bool,
) -> int | None:
    """Return the frontier publishable after retaining ``delay_windows``."""
    if delay_windows <= 0 or is_initial_frontier:
        return completed_end

    pending_commits.append(completed_end)
    if len(pending_commits) <= delay_windows:
        return None
    return pending_commits.popleft()
