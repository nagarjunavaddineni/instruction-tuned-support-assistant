import argparse
import json
import statistics
import time
from pathlib import Path

from src.inference.predictor import SupportPredictor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--runs", type=int, default=5)
    a = p.parse_args()
    m = SupportPredictor(a.model)
    q = "My Docker container exits immediately. How do I troubleshoot it?"
    m.generate(q, 32)
    times = []
    for _ in range(a.runs):
        s = time.perf_counter()
        m.generate(q, 128)
        times.append(time.perf_counter() - s)
    r = {
        "model": a.model,
        "runs": a.runs,
        "mean_seconds": statistics.mean(times),
        "median_seconds": statistics.median(times),
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/benchmark.json").write_text(json.dumps(r, indent=2))
    print(r)


if __name__ == "__main__":
    main()
