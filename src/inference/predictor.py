import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class SupportPredictor:
    def __init__(self, model_path=None):
        self.path = model_path or os.getenv(
            "INFERENCE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"
        )
        self.tok = AutoTokenizer.from_pretrained(self.path)
        self.tok.pad_token = self.tok.pad_token or self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.path, device_map="auto" if torch.cuda.is_available() else None
        )
        self.model.eval()

    def generate(self, question, max_new_tokens=256, temperature=0.3):
        msgs = [
            {
                "role": "system",
                "content": "Return problem summary, likely causes, troubleshooting steps, commands, and prevention tips.",
            },
            {"role": "user", "content": question},
        ]
        text = (
            self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            if self.tok.chat_template
            else f"System: {msgs[0]['content']}\nUser: {question}\nAssistant:"
        )
        x = self.tok(text, return_tensors="pt")
        d = next(self.model.parameters()).device
        x = {k: v.to(d) for k, v in x.items()}
        with torch.inference_mode():
            y = self.model.generate(
                **x,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                pad_token_id=self.tok.eos_token_id,
            )
        return self.tok.decode(
            y[0][x["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()
