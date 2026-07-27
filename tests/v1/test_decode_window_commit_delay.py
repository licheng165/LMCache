# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import deque
import unittest

# First Party
from lmcache.integration.vllm.decode_window_commit import (
    publish_delayed_decode_window_commit,
)


class TestDecodeWindowCommitDelay(unittest.TestCase):
    def test_zero_delay_publishes_immediately(self) -> None:
        pending = deque()

        published = publish_delayed_decode_window_commit(
            pending,
            512,
            0,
            is_initial_frontier=False,
        )

        self.assertEqual(published, 512)
        self.assertEqual(list(pending), [])

    def test_initial_frontier_bypasses_delay(self) -> None:
        pending = deque()

        published = publish_delayed_decode_window_commit(
            pending,
            256,
            2,
            is_initial_frontier=True,
        )

        self.assertEqual(published, 256)
        self.assertEqual(list(pending), [])

    def test_delay_one_retains_latest_save(self) -> None:
        pending = deque()

        first = publish_delayed_decode_window_commit(
            pending,
            512,
            1,
            is_initial_frontier=False,
        )
        second = publish_delayed_decode_window_commit(
            pending,
            768,
            1,
            is_initial_frontier=False,
        )

        self.assertIsNone(first)
        self.assertEqual(second, 512)
        self.assertEqual(list(pending), [768])

    def test_delay_two_retains_two_latest_saves(self) -> None:
        pending = deque()
        published = [
            publish_delayed_decode_window_commit(
                pending,
                window_end,
                2,
                is_initial_frontier=False,
            )
            for window_end in (512, 768, 1024)
        ]

        self.assertEqual(published, [None, None, 512])
        self.assertEqual(list(pending), [768, 1024])

    def test_catch_up_range_counts_as_one_save_completion(self) -> None:
        pending = deque()

        first = publish_delayed_decode_window_commit(
            pending,
            1024,
            1,
            is_initial_frontier=False,
        )
        second = publish_delayed_decode_window_commit(
            pending,
            1280,
            1,
            is_initial_frontier=False,
        )

        self.assertIsNone(first)
        self.assertEqual(second, 1024)


if __name__ == "__main__":
    unittest.main()
