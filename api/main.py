from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.inference.predictor import SupportPredictor

app = FastAPI(title="Instruction-Tuned Support Assistant")


class Request(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    max_new_tokens: int = Field(default=256, ge=16, le=1024)
    temperature: float = Field(default=0.3, ge=0, le=2)


@lru_cache
def model():
    return SupportPredictor()


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/generate")
def generate(r: Request):
    return {"answer": model().generate(r.question, r.max_new_tokens, r.temperature)}
