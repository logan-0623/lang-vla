"""Per-task success from upstream run_libero_eval logs (one or more eval output dirs).

  python scripts/per_task.py $RESULTS/eval/oft-*-49255723 $RESULTS/eval/oft-*-49268902
"""
import re
import sys
from collections import Counter
from pathlib import Path

for d in sys.argv[1:]:
    n, k = Counter(), Counter()
    for log in sorted(Path(d, "logs").glob("*.txt")):
        task = None
        for line in log.read_text().splitlines():
            if m := re.match(r"^Task: (.*)", line):
                task = m.group(1).strip()
            elif m := re.match(r"^Success: (True|False)", line):
                n[task] += 1
                k[task] += m.group(1) == "True"
    print(f"== {Path(d).name}: {sum(k.values())}/{sum(n.values())}")
    for t in sorted(n, key=str):
        print(f"  {k[t]:3d}/{n[t]:<3d} {k[t] / n[t]:6.1%}  {t}")
