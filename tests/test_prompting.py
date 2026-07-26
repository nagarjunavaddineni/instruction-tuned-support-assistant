from src.data.prompting import build_messages


def test_roles():
    r = {
        "instruction": "Explain",
        "input": "problem",
        "response": "solution",
        "category": "linux",
    }
    assert [x["role"] for x in build_messages(r)] == ["system", "user", "assistant"]
