"""Wait until the machine has room for a 7B model in bfloat16 (about 15 GB), then exit.

The 7B jobs swap heavily, and slow every other program, when they start while other
applications hold most of the 24 GB. This polls `vm_stat` once a minute and returns once
free + inactive + speculative + purgeable memory has stayed above the threshold for three
consecutive checks.

Usage: wait_for_memory.py [gigabytes, default 15]
"""
from __future__ import annotations

import re
import subprocess
import sys
import time


def available_gb() -> float:
    text = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(re.search(r"page size of (\d+) bytes", text).group(1))
    count = lambda name: int(re.search(rf"{name}:\s+(\d+)", text).group(1))
    pages = sum(count(name) for name in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"))
    return pages * page / 2**30


def main() -> None:
    need = float(sys.argv[1]) if len(sys.argv) > 1 else 15.0
    streak = 0
    while streak < 3:
        free = available_gb()
        streak = streak + 1 if free >= need else 0
        print(f"{time.strftime('%H:%M:%S')} available {free:.1f} GB (need {need:.0f})", flush=True)
        if streak < 3:
            time.sleep(60)
    print("enough memory; continuing", flush=True)


if __name__ == "__main__":
    main()
