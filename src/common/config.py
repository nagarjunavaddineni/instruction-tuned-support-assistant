from dataclasses import dataclass
import os
from dotenv import load_dotenv


def _bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class TrainingConfig:
    model_name: str
    max_seq_length: int
    num_epochs: float
    train_batch_size: int
    eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    use_4bit: bool
    mlflow_tracking_uri: str
    resume_from_checkpoint: str | None

    @classmethod
    def from_env(cls):
        load_dotenv()
        return cls(
            model_name=os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct"),
            max_seq_length=int(os.getenv("MAX_SEQ_LENGTH", "512")),
            num_epochs=float(os.getenv("NUM_EPOCHS", "2")),
            train_batch_size=int(os.getenv("TRAIN_BATCH_SIZE", "2")),
            eval_batch_size=int(os.getenv("EVAL_BATCH_SIZE", "2")),
            gradient_accumulation_steps=int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "8")),
            learning_rate=float(os.getenv("LEARNING_RATE", "0.0002")),
            weight_decay=float(os.getenv("WEIGHT_DECAY", "0.01")),
            warmup_ratio=float(os.getenv("WARMUP_RATIO", "0.05")),
            lora_rank=int(os.getenv("LORA_RANK", "16")),
            lora_alpha=int(os.getenv("LORA_ALPHA", "32")),
            lora_dropout=float(os.getenv("LORA_DROPOUT", "0.05")),
            use_4bit=_bool(os.getenv("USE_4BIT", "false")),
            mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"),
            resume_from_checkpoint=os.getenv("RESUME_FROM_CHECKPOINT", "").strip() or None,
        )
