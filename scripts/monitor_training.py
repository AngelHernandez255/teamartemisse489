"""Run a command while recording lightweight system metrics to CSV."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from teamartemisse489.logging_config import get_logger, setup_logging  # noqa: E402
from teamartemisse489.monitoring import ResourceMonitor  # noqa: E402

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor CPU and memory usage while a training command runs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs") / "system_metrics.csv",
        help="CSV file that receives monitoring samples.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Seconds between monitoring samples.",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to run. Use -- before the command.",
    )
    return parser.parse_args()


def main() -> int:
    setup_logging()
    args = parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]

    if not command:
        logger.error(
            "No command provided. Example: "
            "python scripts/monitor_training.py -- python models/train.py"
        )
        return 2

    logger.info("Starting monitored command: %s", " ".join(command))
    process = subprocess.Popen(command)
    with ResourceMonitor(
        args.output,
        interval_seconds=args.interval,
        pid=process.pid,
        command_label=" ".join(command),
    ):
        return_code = process.wait()

    if return_code == 0:
        logger.info("Monitoring complete. Metrics written to %s", args.output)
    else:
        logger.error(
            "Command exited with status %d. Metrics written to %s",
            return_code,
            args.output,
        )
    return return_code


if __name__ == "__main__":
    sys.exit(main())
