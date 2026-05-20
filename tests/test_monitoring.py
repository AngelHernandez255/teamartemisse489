"""Tests for resource monitoring helpers."""

from __future__ import annotations

import csv
import time
from pathlib import Path

from teamartemisse489.monitoring import ResourceMonitor


def test_resource_monitor_writes_csv_samples(tmp_path: Path) -> None:
    output_path = tmp_path / "system_metrics.csv"

    with ResourceMonitor(output_path, interval_seconds=0.05, command_label="test"):
        time.sleep(0.08)

    rows = list(csv.DictReader(output_path.open(encoding="utf-8")))

    assert rows
    assert {"start", "end"}.issubset({row["phase"] for row in rows})
    assert rows[0]["command"] == "test"
    assert "system_cpu_percent" in rows[0]
