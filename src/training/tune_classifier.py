import argparse
import json

import optuna
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


def read(p):
    return [json.loads(x) for x in open(p, encoding="utf-8")]


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--trials", type=int, default=10)
    n = a.parse_args().trials
    tr, va = read("data/processed/train.jsonl"), read("data/processed/validation.jsonl")

    def obj(t):
        v = TfidfVectorizer(
            max_features=t.suggest_int("max_features", 500, 5000),
            ngram_range=(1, t.suggest_int("ngram_max", 1, 2)),
        )
        x = v.fit_transform([r["input"] for r in tr])
        xv = v.transform([r["input"] for r in va])
        m = LogisticRegression(
            C=t.suggest_float("C", 0.05, 10, log=True), max_iter=1000
        )
        m.fit(x, [r["category"] for r in tr])
        return f1_score([r["category"] for r in va], m.predict(xv), average="weighted")

    s = optuna.create_study(direction="maximize")
    s.optimize(obj, n_trials=n)
    print(s.best_value, s.best_params)


if __name__ == "__main__":
    main()
