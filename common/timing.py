"""Timing harness: every run appends a row to results/timing.csv (PLAN principle 4)."""
import csv
import json
import platform
import sys
import time
from contextlib import contextmanager
from datetime import datetime

from . import RESULTS

COLUMNS = ["when", "approach", "stage", "seconds", "peak_vram_gib", "host", "info"]


@contextmanager
def timed(approach: str, stage: str, **info):
    """with timed("a1", "train", epochs=3) as row: ...   — add fields to `row` inside the block."""
    torch = sys.modules.get("torch")   # only measure VRAM if the caller already uses torch
    cuda = torch is not None and torch.cuda.is_available()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    row = dict(info)
    t0 = time.perf_counter()
    try:
        yield row
    finally:
        seconds = time.perf_counter() - t0
        path = RESULTS / "timing.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        new = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")   # csv defaults to CRLF; the repo stores LF
            if new:
                w.writerow(COLUMNS)
            w.writerow([datetime.now().isoformat(timespec="seconds"), approach, stage, f"{seconds:.3f}",
                        f"{torch.cuda.max_memory_allocated() / 2**30:.2f}" if cuda else "",
                        platform.node(), json.dumps(row, default=str)])
