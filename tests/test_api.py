from fastapi.testclient import TestClient

import api.main as api_main
from api.main import app


def test_health():
    assert TestClient(app).get("/health").json() == {"status": "healthy"}


class _StubPredictor:
    def generate(self, question, max_new_tokens, temperature, history=None):
        return f"stub answer to: {question} (history turns: {len(history or [])})"


def test_generate_returns_model_answer(monkeypatch):
    monkeypatch.setattr(api_main, "model", lambda: _StubPredictor())

    response = TestClient(app).post(
        "/generate", json={"question": "Why does my build fail?"}
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "stub answer to: Why does my build fail? (history turns: 0)"
    }


def test_generate_passes_history_to_predictor(monkeypatch):
    monkeypatch.setattr(api_main, "model", lambda: _StubPredictor())

    response = TestClient(app).post(
        "/generate",
        json={
            "question": "And now?",
            "history": [
                {"role": "user", "content": "My build fails."},
                {"role": "assistant", "content": "Check your logs."},
            ],
        },
    )

    assert response.status_code == 200
    assert response.json() == {"answer": "stub answer to: And now? (history turns: 2)"}


def test_generate_rejects_malformed_history(monkeypatch):
    monkeypatch.setattr(api_main, "model", lambda: _StubPredictor())

    response = TestClient(app).post(
        "/generate",
        json={
            "question": "And now?",
            "history": [{"role": "narrator", "content": "not a valid role"}],
        },
    )

    assert response.status_code == 422


def test_generate_rejects_oversized_history(monkeypatch):
    monkeypatch.setattr(api_main, "model", lambda: _StubPredictor())

    history = [{"role": "user", "content": "hi"} for _ in range(21)]
    response = TestClient(app).post(
        "/generate", json={"question": "And now?", "history": history}
    )

    assert response.status_code == 422


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
