import argparse
import json
import statistics
import time
from dataclasses import replace
from pathlib import Path

import torch

from src.common.config import InferenceConfig
from src.inference.predictor import SupportPredictor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--new-tokens", type=int, default=128)
    p.add_argument("--use-4bit", action="store_true")
    a = p.parse_args()

    config = replace(InferenceConfig.from_env(), use_4bit=a.use_4bit)
    m = SupportPredictor(a.model, config=config)
    q = "My Docker container exits immediately. How do I troubleshoot it?"
    m.generate(q, 32)  # warmup

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    times = []
    tokens_per_second = []
    for _ in range(a.runs):
        s = time.perf_counter()
        answer = m.generate(q, a.new_tokens)
        elapsed = time.perf_counter() - s
        times.append(elapsed)
        n_tokens = len(m.tok(answer, add_special_tokens=False)["input_ids"])
        tokens_per_second.append(n_tokens / elapsed)

    r = {
        "model": a.model,
        "runs": a.runs,
        "use_4bit": a.use_4bit,
        "mean_seconds": statistics.mean(times),
        "median_seconds": statistics.median(times),
        "mean_tokens_per_second": statistics.mean(tokens_per_second),
    }
    if torch.cuda.is_available():
        r["peak_memory_mb"] = torch.cuda.max_memory_allocated() / (1024**2)

    Path("reports").mkdir(exist_ok=True)
    Path("reports/benchmark.json").write_text(json.dumps(r, indent=2))
    print(r)


if __name__ == "__main__":
    main()
