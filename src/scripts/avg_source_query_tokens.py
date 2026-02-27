"""
avg_source_query_tokens.py

Computes the average number of BERT (bert-base-uncased) tokens in the
`source_query` field found inside `raw_response.evidence[]` of each JSONL
file in responses_reverified/.

Per question (record), duplicate source_query values are deduplicated before
tokenizing — only unique queries per record are counted.

Also reports average iterations per question, where the number of iterations
for a question equals the maximum `source_step` value seen in its evidence.

Run from the src/ directory:
    python scripts/avg_source_query_tokens.py
"""

import json
from pathlib import Path

from transformers import AutoTokenizer

SRC_DIR       = Path(__file__).parent.parent
RESPONSE_DIR  = SRC_DIR / "responses_reverified"
TOKENIZER_NAME = "bert-base-uncased"


def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def process_file(path: Path, tokenizer) -> tuple[list[int], list[int], int]:
    """
    Returns (token_counts, iter_counts, n_records).
      token_counts : one entry per unique source_query seen in the file
      iter_counts  : one entry per record = max(source_step) for that question
    """
    seen_queries: set[str] = set()
    token_counts: list[int] = []
    iter_counts:  list[int] = []
    n_records = 0

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            n_records += 1
            evidence = rec.get("raw_response", {}).get("evidence", [])

            # Iterations = max source_step in this record
            steps = [ev.get("source_step") for ev in evidence
                     if isinstance(ev.get("source_step"), int)]
            iter_counts.append(max(steps) if steps else 0)

            # Unique source_query values for this record
            record_queries: set[str] = set()
            for ev in evidence:
                sq = (ev.get("source_query") or "").strip()
                if sq:
                    record_queries.add(sq)

            for sq in record_queries:
                if sq not in seen_queries:
                    seen_queries.add(sq)
                    token_counts.append(count_tokens(tokenizer, sq))

    return token_counts, iter_counts, n_records


def main() -> None:
    print(f"Loading tokenizer: {TOKENIZER_NAME} …\n")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    all_token_counts: list[int] = []
    all_iter_counts:  list[int] = []
    files = sorted(RESPONSE_DIR.glob("*.jsonl"))

    if not files:
        print(f"No JSONL files found in {RESPONSE_DIR}")
        return

    col_w = 45
    print(f"{'File':<{col_w}} {'Records':>8} {'Unique queries':>15} "
          f"{'Avg tok':>8} {'Avg iter':>9} {'Max iter':>9}")
    print("-" * (col_w + 55))

    for path in files:
        token_counts, iter_counts, n_records = process_file(path, tokenizer)

        avg_tok  = sum(token_counts) / len(token_counts) if token_counts else 0
        avg_iter = sum(iter_counts)  / len(iter_counts)  if iter_counts  else 0
        max_iter = max(iter_counts)                       if iter_counts  else 0
        all_token_counts.extend(token_counts)
        all_iter_counts.extend(iter_counts)

        print(f"{path.name:<{col_w}} {n_records:>8} {len(token_counts):>15} "
              f"{avg_tok:>8.1f} {avg_iter:>9.2f} {max_iter:>9}")

    print("-" * (col_w + 55))

    print(f"\nOVERALL")
    if all_token_counts:
        print(f"  Total unique source queries : {len(all_token_counts)}")
        print(f"  Average tokens per query    : {sum(all_token_counts)/len(all_token_counts):.2f}")
        print(f"  Min / Max tokens            : {min(all_token_counts)} / {max(all_token_counts)}")
    if all_iter_counts:
        print(f"  Average iterations          : {sum(all_iter_counts)/len(all_iter_counts):.2f}")
        print(f"  Max iterations seen         : {max(all_iter_counts)}")


if __name__ == "__main__":
    main()
