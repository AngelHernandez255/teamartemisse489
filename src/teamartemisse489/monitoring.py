"""Lightweight system monitoring helpers for training runs."""

from __future__ import annotations

import csv
import os
import threading
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO

import psutil


class ResourceMonitor:
    """Write process and system resource metrics to a CSV file."""

    fieldnames = [
        "timestamp_utc",
        "elapsed_seconds",
        "phase",
        "command",
        "system_cpu_percent",
        "system_memory_percent",
        "system_memory_used_mb",
        "process_cpu_percent",
        "process_memory_rss_mb",
        "process_threads",
    ]

    def __init__(
        self,
        output_path: Path | str,
        *,
        interval_seconds: float = 5.0,
        pid: int | None = None,
        command_label: str = "training",
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be greater than zero")

        self.output_path = Path(output_path)
        self.interval_seconds = interval_seconds
        self.pid = pid if pid is not None else os.getpid()
        self.command_label = command_label
        self._process: psutil.Process | None = None
        self._started_at = 0.0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._file: TextIO | None = None
        self._writer: csv.DictWriter[str] | None = None

    def __enter__(self) -> ResourceMonitor:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    def start(self) -> None:
        """Start monitoring in a background thread."""
        if self._thread is not None:
            return

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.output_path.exists() and self.output_path.stat().st_size > 0
        self._file = self.output_path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
        if not file_exists:
            self._writer.writeheader()

        self._process = psutil.Process(self.pid)
        self._process.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None)
        self._started_at = time.perf_counter()
        self.write_sample("start")

        self._thread = threading.Thread(
            target=self._monitor_loop,
            name="resource-monitor",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop monitoring and write a final sample."""
        if self._thread is None:
            return

        self._stop_event.set()
        self._thread.join(timeout=self.interval_seconds + 1)
        self.write_sample("end")

        if self._file is not None:
            self._file.close()

        self._thread = None
        self._file = None
        self._writer = None

    def write_metric(self, name: str, value: float | int | str) -> None:
        """Write an application-level metric as a CSV row.

        The row keeps the same CSV schema as resource samples and stores the
        metric as ``phase=<name>`` with the value in the command column.
        """
        self._write_row({"phase": name, "command": str(value)})

    def write_sample(self, phase: str = "sample") -> None:
        """Capture one resource sample immediately."""
        try:
            virtual_memory = psutil.virtual_memory()
            row: dict[str, str | float | int] = {
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "elapsed_seconds": round(time.perf_counter() - self._started_at, 3),
                "phase": phase,
                "command": self.command_label,
                "system_cpu_percent": psutil.cpu_percent(interval=None),
                "system_memory_percent": virtual_memory.percent,
                "system_memory_used_mb": round(virtual_memory.used / 1024 / 1024, 2),
                "process_cpu_percent": "",
                "process_memory_rss_mb": "",
                "process_threads": "",
            }

            if self._process is not None and self._process.is_running():
                with self._process.oneshot():
                    row["process_cpu_percent"] = self._process.cpu_percent(
                        interval=None
                    )
                    row["process_memory_rss_mb"] = round(
                        self._process.memory_info().rss / 1024 / 1024,
                        2,
                    )
                    row["process_threads"] = self._process.num_threads()
            self._write_row(row)
        except (psutil.Error, RuntimeError):
            self._write_row(
                {
                    "phase": f"{phase}_unavailable",
                    "command": self.command_label,
                }
            )

    def _monitor_loop(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            self.write_sample()

    def _write_row(self, row: Mapping[str, object]) -> None:
        if self._writer is None or self._file is None:
            return

        complete_row = {field: "" for field in self.fieldnames}
        complete_row.update(row)
        if not complete_row["timestamp_utc"]:
            complete_row["timestamp_utc"] = datetime.now(UTC).isoformat()
        if not complete_row["elapsed_seconds"]:
            complete_row["elapsed_seconds"] = round(
                time.perf_counter() - self._started_at,
                3,
            )

        self._writer.writerow(complete_row)
        self._file.flush()
