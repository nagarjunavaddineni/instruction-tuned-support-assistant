import argparse
import json

from src.inference.predictor import SupportPredictor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--limit", type=int, default=20)
    a = p.parse_args()
    m = SupportPredictor(a.model)
    rows = [json.loads(x) for x in open("data/processed/test.jsonl", encoding="utf-8")][
        : a.limit
    ]
    keys = [
        "problem summary",
        "likely causes",
        "troubleshooting",
        "command",
        "prevention",
    ]
    scores = []
    for r in rows:
        out = m.generate(r["input"]).lower()
        scores.append(sum(k in out for k in keys) / len(keys))
    print(
        {
            "examples": len(scores),
            "average_section_coverage": sum(scores) / len(scores) if scores else 0,
        }
    )


if __name__ == "__main__":
    main()
