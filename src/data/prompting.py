def build_messages(record):
    return [
        {"role": "system", "content": "You are a careful technical-support assistant."},
        {
            "role": "user",
            "content": f"{record['instruction']}\n\nQuestion:\n{record['input']}",
        },
        {"role": "assistant", "content": record["response"]},
    ]


def format_example(record, tokenizer):
    messages = build_messages(record)
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    return (
        f"### System\n{messages[0]['content']}\n\n"
        f"### User\n{messages[1]['content']}\n\n"
        f"### Assistant\n{messages[2]['content']}"
    )
