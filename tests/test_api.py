from fastapi.testclient import TestClient

import api.main as api_main
from api.main import app


def test_health():
    assert TestClient(app).get("/health").json() == {"status": "healthy"}


class _StubPredictor:
    def generate(self, question, max_new_tokens, temperature):
        return f"stub answer to: {question}"


def test_generate_returns_model_answer(monkeypatch):
    monkeypatch.setattr(api_main, "model", lambda: _StubPredictor())

    response = TestClient(app).post(
        "/generate", json={"question": "Why does my build fail?"}
    )

    assert response.status_code == 200
    assert response.json() == {"answer": "stub answer to: Why does my build fail?"}


def test_generate_rejects_question_too_short(monkeypatch):
    monkeypatch.setattr(api_main, "model", lambda: _StubPredictor())

    response = TestClient(app).post("/generate", json={"question": "hi"})

    assert response.status_code == 422


def test_generate_rejects_temperature_out_of_range(monkeypatch):
    monkeypatch.setattr(api_main, "model", lambda: _StubPredictor())

    response = TestClient(app).post(
        "/generate", json={"question": "Why does my build fail?", "temperature": 5}
    )

    assert response.status_code == 422
