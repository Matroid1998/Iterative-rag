"""
count_text_tokens.py

Counts the number of tokens in each `text` field across all path entries in
docs/chemrxiv_qa.json, using the BERT tokenizer (bert-base-uncased).

BASF-AI/ChEmbed is built on BERT and shares the same tokenizer vocabulary,
so this gives the correct token counts without requiring a HuggingFace token.

Run from the src/ directory:
    python scripts/count_text_tokens.py
"""

import json
from pathlib import Path

from transformers import AutoTokenizer

DATA_PATH = Path(__file__).parent.parent / "docs" / "chemrxiv_qa.json"
# ChEmbed uses BERT's tokenizer; bert-base-uncased is public and requires no auth.
TOKENIZER_NAME = "bert-base-uncased"


def main():
    print(f"Loading tokenizer: {TOKENIZER_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    print("Tokenizer loaded.\n")

    print(f"Reading {DATA_PATH} ...")
    with open(DATA_PATH, "r") as f:
        data = json.load(f)

    total_tokens = 0
    count = 0
    min_tokens = float("inf")
    max_tokens = 0

    for item in data:
        for path_entry in item.get("path", []):
            text = path_entry.get("text", "")
            n = len(tokenizer(text, add_special_tokens=False)["input_ids"])
            total_tokens += n
            count += 1
            min_tokens = min(min_tokens, n)
            max_tokens = max(max_tokens, n)

    if count == 0:
        print("No text fields found.")
        return

    avg_tokens = total_tokens / count

    print(f"Total text fields : {count:,}")
    print(f"Total tokens      : {total_tokens:,}")
    print(f"Average tokens    : {avg_tokens:.2f}")
    print(f"Min tokens        : {min_tokens}")
    print(f"Max tokens        : {max_tokens}")


if __name__ == "__main__":
    main()
