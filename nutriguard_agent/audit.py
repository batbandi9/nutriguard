"""Append-only audit log ("black box recorder") in JSON Lines format."""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

AUDIT_PATH = Path(os.getenv(
    "NUTRIGUARD_AUDIT_PATH",
    Path(__file__).resolve().parent.parent / "audit" / "audit.jsonl",
))
_lock = threading.Lock()


def _safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def log_event(event: str, **fields: Any) -> None:
    record = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "event": event}
    record.update({k: _safe(v) for k, v in fields.items()})
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _lock, AUDIT_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_events(session_tag: str | None = None, limit: int = 50) -> list[dict]:
    if not AUDIT_PATH.exists():
        return []
    rows = []
    with AUDIT_PATH.open(encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if session_tag is None or row.get("session") == session_tag:
                rows.append(row)
    return rows[-limit:]
