import mlflow
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from src.data.prompting import format_example


def load_model_tokenizer(c):
    tok = AutoTokenizer.from_pretrained(c.model_name)
    tok.pad_token = tok.pad_token or tok.eos_token
    q = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        if c.use_4bit
        else None
    )
    model = AutoModelForCausalLM.from_pretrained(
        c.model_name, quantization_config=q, device_map="auto" if c.use_4bit else None
    )
    model.config.use_cache = False
    return model, tok


def train(model, tok, c, output):
    ds = load_dataset(
        "json",
        data_files={
            "train": "data/processed/train.jsonl",
            "validation": "data/processed/validation.jsonl",
        },
    )

    def encode(r):
        x = tok(format_example(r, tok), truncation=True, max_length=c.max_seq_length)
        x["labels"] = x["input_ids"].copy()
        return x

    cols = ds["train"].column_names
    ds = ds.map(encode, remove_columns=cols)
    mlflow.set_tracking_uri(c.mlflow_tracking_uri)
    mlflow.set_experiment("support-assistant")
    args = TrainingArguments(
        output_dir=output,
        num_train_epochs=c.num_epochs,
        per_device_train_batch_size=c.train_batch_size,
        per_device_eval_batch_size=c.eval_batch_size,
        gradient_accumulation_steps=c.gradient_accumulation_steps,
        learning_rate=c.learning_rate,
        weight_decay=c.weight_decay,
        warmup_ratio=c.warmup_ratio,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=25,
        save_steps=25,
        logging_steps=5,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to=["mlflow"],
        gradient_checkpointing=True,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
        processing_class=tok,
    )
    trainer.train(resume_from_checkpoint=c.resume_from_checkpoint)
    trainer.save_model(output)
    tok.save_pretrained(output)
