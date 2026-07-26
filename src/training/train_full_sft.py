from src.common.config import TrainingConfig
from src.training.lm_common import load_model_tokenizer, train


def main():
    c = TrainingConfig.from_env()
    model, tok = load_model_tokenizer(c)
    train(model, tok, c, "outputs/full-sft")


if __name__ == "__main__":
    main()
