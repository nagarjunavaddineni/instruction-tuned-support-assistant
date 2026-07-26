import argparse, json, random
from pathlib import Path

CATEGORIES = {
    "docker": ["container exits immediately", "image build is slow", "container cannot reach host"],
    "kubernetes": ["pod is in CrashLoopBackOff", "service is unreachable", "rollout is stuck"],
    "python": ["ModuleNotFoundError occurs", "wrong interpreter is used", "process uses too much memory"],
    "linux": ["disk is full", "systemd service fails", "permission is denied"],
    "git": ["merge conflict occurs", "secret was committed", "branch diverged"],
    "cloud": ["application times out", "object storage is denied", "function times out"],
    "database": ["query is slow", "connection pool is exhausted", "migration is locked"],
    "networking": ["DNS resolves incorrectly", "TCP connection times out", "proxy returns 502"],
    "security": ["API key was exposed", "login traffic is suspicious", "dependency is vulnerable"],
}


def response(category, issue, i):
    return (
        f"Problem summary: {issue}.\n\nLikely causes:\n- configuration error\n- missing dependency or permission\n- resource or network problem\n\n"
        "Troubleshooting steps:\n1. Capture the exact error and timestamp.\n2. Inspect logs and recent changes.\n3. Validate dependencies, permissions, networking, and resources.\n4. Test one hypothesis at a time.\n5. Roll back risky changes when appropriate.\n\n"
        f"Useful command: inspect the relevant {category} logs and status.\n\nPrevention tip: add monitoring and a runbook. Case {i}."
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/raw/support_examples.jsonl")
    p.add_argument("--examples-per-category", type=int, default=30)
    args = p.parse_args()
    rows = []
    for category, issues in CATEGORIES.items():
        for i in range(args.examples_per_category):
            issue = issues[i % len(issues)]
            rows.append({
                "instruction": "Provide a structured technical-support answer with causes, steps, commands, and prevention tips.",
                "input": f"A {category} problem occurs: {issue}. Scenario {i + 1}.",
                "response": response(category, issue, i + 1),
                "category": category,
            })
    random.Random(42).shuffle(rows)
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(x) for x in rows) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} examples to {out}")


if __name__ == "__main__":
    main()
