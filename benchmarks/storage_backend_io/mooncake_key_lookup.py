# SPDX-License-Identifier: Apache-2.0
import json
import sys
import time
from argparse import ArgumentParser, Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Third Party
import yaml


def _arguments() -> Namespace:
    parser = ArgumentParser(description="Query exact keys in a Mooncake master")
    parser.add_argument("keys", nargs="+", help="exact physical Mooncake keys")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--master")
    parser.add_argument("--local-hostname", default="localhost")
    parser.add_argument("--metadata-server", default="P2PHANDSHAKE")
    return parser.parse_args()


def _extra_config(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return dict(loaded.get("extra_config") or {})


def main() -> int:
    """Open a metadata-only Mooncake client and query exact keys."""
    args = _arguments()
    extra = _extra_config(args.config)
    master = args.master or extra.get("master_server_address")
    if not master:
        raise ValueError("Mooncake master is absent; pass --master")

    # Third Party
    from mooncake.store import MooncakeDistributedStore

    store = MooncakeDistributedStore()
    status = store.setup(
        args.local_hostname,
        args.metadata_server,
        0,
        0,
        "tcp",
        "",
        master,
    )
    if status not in (None, 0):
        store.close()
        raise RuntimeError(f"Mooncake lookup setup failed: status={status}")
    try:
        started = time.perf_counter()
        results = store.batch_is_exist(args.keys)
        elapsed_ms = (time.perf_counter() - started) * 1000
        print(
            json.dumps(
                {
                    "schema": 1,
                    "timestamp": datetime.now(timezone.utc).isoformat(
                        timespec="microseconds"
                    ),
                    "master": master,
                    "elapsed_ms": round(elapsed_ms, 3),
                    "keys": [
                        {
                            "key": key,
                            "result": results[index]
                            if index < len(results)
                            else None,
                        }
                        for index, key in enumerate(args.keys)
                    ],
                },
                indent=2,
            )
        )
        return (
            0
            if len(results) == len(args.keys)
            and all(result == 1 for result in results)
            else 1
        )
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())
