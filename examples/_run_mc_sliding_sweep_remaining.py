"""Run the 5 remaining cases (conformal N=30,40 + sbm N=20,30,40)."""
import sys
sys.path.insert(0, "examples")
from _run_mc_sliding_sweep import run_one
import time

CASES = [(30, False), (40, False), (20, True), (30, True), (40, True)]

for n, sbm in CASES:
    run_one(n, sbm)
print(f"[{time.strftime('%H:%M:%S')}] all done")
