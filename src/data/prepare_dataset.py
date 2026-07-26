import json
from pathlib import Path
from sklearn.model_selection import train_test_split

REQUIRED = {"instruction", "input", "response", "category"}


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            row = json.loads(line)
            missing = REQUIRED - row.keys()
            if missing:
                raise ValueError(f"line {n} missing {sorted(missing)}")
            rows.append({k: str(row[k]).strip() for k in REQUIRED})
    return rows


def deduplicate(rows):
    return list({r["input"].casefold(): r for r in rows}.values())


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def main():
    rows = deduplicate(read_jsonl("data/raw/support_examples.jsonl"))
    labels = [r["category"] for r in rows]
    train, temp = train_test_split(rows, test_size=.2, random_state=42, stratify=labels)
    validation, test = train_test_split(temp, test_size=.5, random_state=42, stratify=[r["category"] for r in temp])
    out = Path("data/processed")
    write(out / "train.jsonl", train); write(out / "validation.jsonl", validation); write(out / "test.jsonl", test)
    report = {"total": len(rows), "train": len(train), "validation": len(validation), "test": len(test)}
    (out / "dataset_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
