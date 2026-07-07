"""A minimal JSON-lines experiment logger.

Each experiment run owns one file; every call to :meth:`RunLogger.log`
appends a single JSON object on its own line, together with the step number
and the wall-clock time. JSON-lines was chosen over a binary or database
format because it is append-only (crash-safe by construction), diffable,
greppable, and loads into a pandas DataFrame in one line for analysis.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import TracebackType
from typing import Any


class RunLogger:
    """Append-only logger for a single experiment run.

    Parameters
    ----------
    path
        Destination file. Parent directories are created if missing. An
        existing file is appended to, so a run can be resumed.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")
        self._start = time.monotonic()

    def log(self, step: int, **metrics: Any) -> None:
        """Append one record with the given step and metric values."""
        record: dict[str, Any] = {
            "step": step,
            "elapsed_s": round(time.monotonic() - self._start, 3),
            **metrics,
        }
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> RunLogger:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def read_run(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSON-lines run file back into a list of records."""
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
