# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import OrderedDict, deque
from typing import Any, Iterable, Optional
import os
import threading
import time


def _env_bool(name: str, default: str = "1") -> bool:
    value = os.environ.get(name, default).lower()
    return value not in ("0", "false", "no", "off")


ENABLED = _env_bool("LMCACHE_MULTILOCATION_DIAG", "1")
MAX_KEYS = int(os.environ.get("LMCACHE_MULTILOCATION_DIAG_MAX_KEYS", "20000"))
MAX_EVENTS_PER_KEY = int(
    os.environ.get("LMCACHE_MULTILOCATION_DIAG_MAX_EVENTS_PER_KEY", "64")
)

_LOCK = threading.Lock()
_EVENTS: OrderedDict[str, deque[tuple[float, str, dict[str, str]]]] = OrderedDict()


def key_to_debug_string(key: Any) -> str:
    try:
        return key.to_string()
    except Exception:
        return repr(key)


def _format_value(value: Any) -> str:
    if value is None or isinstance(value, (bool, int, float, str)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        if len(value) <= 4:
            return "[" + ", ".join(_format_value(item) for item in value) + "]"
        return (
            f"[len={len(value)}, first={_format_value(value[0])}, "
            f"last={_format_value(value[-1])}]"
        )
    if isinstance(value, dict):
        items = list(value.items())
        if len(items) <= 4:
            return "{" + ", ".join(
                f"{_format_value(k)}: {_format_value(v)}" for k, v in items
            ) + "}"
        return f"{{len={len(value)}, keys={[str(k) for k, _ in items[:4]]}}}"
    try:
        if hasattr(value, "to_string"):
            return value.to_string()
    except Exception:
        pass
    return repr(value)


def _safe_debug_call(fn: Any) -> str:
    try:
        return _format_value(fn())
    except Exception as exc:
        return f"error:{type(exc).__name__}:{exc}"


def memory_obj_debug_fields(memory_obj: Any) -> dict[str, str]:
    meta = getattr(memory_obj, "meta", None)
    metadata = getattr(memory_obj, "metadata", None)
    return {
        "memory_obj_type": type(memory_obj).__name__,
        "memory_obj_meta_shape": _format_value(getattr(meta, "shape", None)),
        "memory_obj_meta_dtype": _format_value(getattr(meta, "dtype", None)),
        "memory_obj_metadata_shape": _format_value(getattr(metadata, "shape", None)),
        "memory_obj_can_evict": _format_value(getattr(memory_obj, "can_evict", None)),
        "memory_obj_num_tokens": _safe_debug_call(memory_obj.get_num_tokens),
        "memory_obj_size": _safe_debug_call(memory_obj.get_size),
    }


def record_key_event(event: str, key: Any, **fields: Any) -> None:
    if not ENABLED or key is None:
        return
    key_string = key_to_debug_string(key)
    safe_fields = {name: _format_value(value) for name, value in fields.items()}
    timestamp = time.time()
    with _LOCK:
        history = _EVENTS.get(key_string)
        if history is None:
            if len(_EVENTS) >= MAX_KEYS:
                _EVENTS.popitem(last=False)
            history = deque(maxlen=MAX_EVENTS_PER_KEY)
            _EVENTS[key_string] = history
        else:
            _EVENTS.move_to_end(key_string)
        history.append((timestamp, event, safe_fields))


def record_key_batch_event(
    event: str,
    keys: Iterable[Any],
    *,
    limit: int = 4,
    **fields: Any,
) -> None:
    if not ENABLED:
        return
    for index, key in enumerate(keys):
        if index >= limit:
            break
        record_key_event(event, key, batch_index=index, **fields)


def dump_key_events(
    logger: Any,
    title: str,
    keys: Iterable[Optional[Any]],
    **context: Any,
) -> None:
    if not ENABLED:
        return
    key_strings = []
    seen = set()
    for key in keys:
        if key is None:
            continue
        key_string = key_to_debug_string(key)
        if key_string in seen:
            continue
        seen.add(key_string)
        key_strings.append(key_string)

    context_string = ", ".join(
        f"{name}={_format_value(value)}" for name, value in context.items()
    )
    logger.warning(
        "[MULTILOC_DIAG] %s context={%s} keys=%s",
        title,
        context_string,
        key_strings,
    )
    with _LOCK:
        snapshots = [
            (key_string, list(_EVENTS.get(key_string, [])))
            for key_string in key_strings
        ]
    for key_string, events in snapshots:
        logger.warning(
            "[MULTILOC_DIAG] key=%s recorded_events=%d",
            key_string,
            len(events),
        )
        for timestamp, event, fields in events:
            fields_string = ", ".join(
                f"{name}={value}" for name, value in fields.items()
            )
            logger.warning(
                "[MULTILOC_DIAG] key=%s t=%.6f event=%s %s",
                key_string,
                timestamp,
                event,
                fields_string,
            )
