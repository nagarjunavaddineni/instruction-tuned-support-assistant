import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.common.config import InferenceConfig


class SupportPredictor:
    def __init__(self, model_path=None, config=None):
        self.config = config or InferenceConfig.from_env()
        self.path = model_path or self.config.model_path
        self.tok = AutoTokenizer.from_pretrained(self.path)
        self.tok.pad_token = self.tok.pad_token or self.tok.eos_token

        use_cuda = torch.cuda.is_available()
        quant_config = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
            if use_cuda and self.config.use_4bit
            else None
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.path,
            device_map="auto" if use_cuda else None,
            dtype=torch.bfloat16 if use_cuda else None,
            quantization_config=quant_config,
        )
        self.model.eval()
        self._response_cache: dict[tuple, str] = {}

    def generate(self, question, max_new_tokens=256, temperature=0.3, history=None):
        history = history or []
        cache_key = None
        if temperature == 0:
            cache_key = (
                question,
                tuple((h["role"], h["content"]) for h in history),
                max_new_tokens,
            )
            cached = self._response_cache.get(cache_key)
            if cached is not None:
                return cached

        msgs = [
            {
                "role": "system",
                "content": "Return problem summary, likely causes, troubleshooting steps, commands, and prevention tips.",
            },
            *history,
            {"role": "user", "content": question},
        ]
        text = (
            self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            if self.tok.chat_template
            else self._fallback_prompt(msgs)
        )
        x = self.tok(text, return_tensors="pt")
        d = next(self.model.parameters()).device
        x = {k: v.to(d) for k, v in x.items()}
        with torch.inference_mode():
            y = self.model.generate(  # type: ignore[misc]
                **x,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                pad_token_id=self.tok.eos_token_id,
            )
        decoded = self.tok.decode(
            y[0][x["input_ids"].shape[1] :], skip_special_tokens=True
        )
        assert isinstance(decoded, str)
        answer = decoded.strip()

        if cache_key is not None:
            self._response_cache[cache_key] = answer
        return answer

    @staticmethod
    def _fallback_prompt(msgs):
        lines = [f"System: {msgs[0]['content']}"]
        for m in msgs[1:]:
            lines.append(f"{m['role'].capitalize()}: {m['content']}")
        lines.append("Assistant:")
        return "\n".join(lines)
