import os

import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def main():
    name = os.getenv("CLASSIFIER_MODEL", "distilbert/distilbert-base-uncased")
    ds = load_dataset(
        "json",
        data_files={
            "train": "data/processed/train.jsonl",
            "validation": "data/processed/validation.jsonl",
        },
    )
    labels = sorted(set(ds["train"]["category"]))
    l2i = {x: i for i, x in enumerate(labels)}
    tok = AutoTokenizer.from_pretrained(name)

    def enc(r):
        x = tok(r["input"], truncation=True, max_length=256)
        x["label"] = l2i[r["category"]]
        return x

    ds = ds.map(enc, remove_columns=ds["train"].column_names)
    model = AutoModelForSequenceClassification.from_pretrained(
        name,
        num_labels=len(labels),
        label2id=l2i,
        id2label={v: k for k, v in l2i.items()},
    )

    def metrics(p):
        y = np.argmax(p.predictions, axis=-1)
        return {
            "accuracy": accuracy_score(p.label_ids, y),
            "f1": f1_score(p.label_ids, y, average="weighted"),
        }

    args = TrainingArguments(
        output_dir="outputs/classifier-torch",
        num_train_epochs=3,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to=["mlflow"],
    )
    t = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=DataCollatorWithPadding(tok),
        processing_class=tok,
        compute_metrics=metrics,
    )
    t.train()
    t.save_model("outputs/classifier-torch")
    tok.save_pretrained("outputs/classifier-torch")


if __name__ == "__main__":
    main()
