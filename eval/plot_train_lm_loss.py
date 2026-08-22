#!/usr/bin/env python3
"""Plot Nanotron training ``lm_loss`` curves from one or more log files.

Example:
    python plot_train_lm_loss.py ../nanotron/train_dclm_stack_20m*.log \
      --labels multi single-small single-large --output train_lm_loss.png
"""

from __future__ import annotations

import argparse
import os
import re
from collections import OrderedDict
from pathlib import Path

# The cluster home directory may be read-only from compute nodes. Matplotlib
# needs a writable cache even when it renders with the non-interactive backend.
matplotlib_cache = Path(os.environ.get("TMPDIR", "/tmp")) / "nanotron-matplotlib"
matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(matplotlib_cache))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
LOSS_LINE = re.compile(
    r"\biteration:\s*(?P<step>\d+)\s*/\s*\d+.*?\blm_loss:\s*(?P<loss>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", type=Path, nargs="+", help="Nanotron .log file(s) to plot")
    parser.add_argument("--labels", nargs="+", help="Legend labels, in the same order as the logs")
    parser.add_argument("--output", type=Path, default=Path("train_lm_loss.png"))
    parser.add_argument("--title", default="Training LM loss")
    parser.add_argument("--smooth-window", type=int, default=1, help="Trailing moving-average window; 1 disables smoothing")
    args = parser.parse_args()
    if args.labels is not None and len(args.labels) != len(args.logs):
        parser.error("--labels must provide exactly one label per log")
    if args.smooth_window <= 0:
        parser.error("--smooth-window must be positive")
    for path in args.logs:
        if not path.is_file():
            parser.error(f"Log file does not exist: {path}")
    return args


def read_losses(path: Path) -> tuple[list[int], list[float]]:
    # OrderedDict keeps the last value for a repeated step. This handles a
    # resumed run or duplicated rank output without plotting duplicate points.
    losses: OrderedDict[int, float] = OrderedDict()
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            match = LOSS_LINE.search(ANSI_ESCAPE.sub("", line))
            if match is not None:
                losses[int(match["step"])] = float(match["loss"])
    ordered = sorted(losses.items())
    return [step for step, _ in ordered], [loss for _, loss in ordered]


def moving_average(values: list[float], window: int) -> list[float]:
    if window == 1:
        return values
    totals = [0.0]
    for value in values:
        totals.append(totals[-1] + value)
    return [
        (totals[index + 1] - totals[max(0, index + 1 - window)]) / min(index + 1, window)
        for index in range(len(values))
    ]


def default_label(path: Path) -> str:
    name = path.stem
    return name.removeprefix("train_").replace("_", " ")


def main() -> None:
    args = parse_args()
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)

    for index, path in enumerate(args.logs):
        steps, losses = read_losses(path)
        if not steps:
            raise ValueError(f"No 'iteration: ... | lm_loss: ...' entries found in {path}")
        label = args.labels[index] if args.labels is not None else default_label(path)
        axis.plot(steps, moving_average(losses, args.smooth_window), label=label, linewidth=1.4)
        print(f"{path}: plotted {len(steps):,} steps ({steps[0]} through {steps[-1]})", flush=True)

    axis.set_title(args.title)
    axis.set_xlabel("Training step")
    axis.set_ylabel("LM loss")
    axis.legend()
    figure.savefig(args.output, dpi=180)
    print(f"Wrote plot to {args.output}", flush=True)


if __name__ == "__main__":
    main()
