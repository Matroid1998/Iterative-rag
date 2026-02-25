"""
reverify_cot_responses.py

Re-verifies all answer files in responses_cot/ using gpt-5-mini and the
same VERIFY_PROMPT from cot_evaluator.py.

For each JSON-lines file it:
  1. Reads every record
  2. Re-runs the verifier on (expected, answer) pairs
  3. Overwrites the file with updated "is_correct" values
  4. Recalculates accuracy and updates the matching CSV in responses_cot/

Usage:
    python scripts/reverify_cot_responses.py              # all files
    python scripts/reverify_cot_responses.py <file.json>  # single file
"""

import json
import os
import sys
import time
import re
from pathlib import Path

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
import tqdm

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RESPONSES_COT_DIR = Path(__file__).resolve().parents[1] / "responses_cot"
VERIFIER_MODEL = "gpt-5-mini"

VERIFY_PROMPT_TEMPLATE = (
    "Task\n"
    "Decide whether Expected and Candidate name the SAME chemical entity.\n"
    "\n"
    "What counts as the SAME\n"
    "- Aliases, common vs IUPAC names, and formulas refer to the same thing "
    "(e.g., lithium chloride = LiCl; acetic acid = ethanoic acid).\n"
    "- Minor packaging/context words do not change identity: material, compound, "
    "sample, reagent, powder, nanopowder, precursor, solution.\n"
    "- The Candidate may be a long sentence or paragraph with explanations; as long "
    "as it explicitly names the same entity as the answer, count it as the same.\n"
    "\n"
    "What is NOT the same\n"
    "- Different polymorph/crystal structure/phase (wurtzite ZnO vs rocksalt ZnO).\n"
    "- Different charge state or ion vs neutral; cation vs anion (Li vs Li+; "
    "chloride ion vs HCl).\n"
    "- Different oxidation state or stoichiometry (FeCl2 vs FeCl3).\n"
    "- Different hydration/solvation (CuSO4 vs CuSO4*5H2O).\n"
    "- Different stereochemistry or isotopic labeling (L- vs D-; 13C-labeled vs "
    "unlabeled).\n"
    "- Salt vs parent acid/base (acetate vs acetic acid).\n"
    "- Class/family vs specific member (alkali metal chloride vs lithium chloride) "
    "unless the specific Expected entity is explicitly named.\n"
    "- Candidate only mentions Expected to negate or contrast it (uses words like "
    "'not', 'instead of', 'different from', 'vs') while naming a different main "
    "entity.\n"
    "\n"
    "Decision rule\n"
    "- If Candidate explicitly names the same entity as Expected (even inside a "
    "longer explanation), answer: true.\n"
    "- Otherwise, answer: false.\n"
    "\n"
    "Output\n"
    "Answer with exactly: true or false\n"
    "\n"
    "Examples\n"
    "Expected: wurtzite ZnO\n"
    "Candidate: The ZnO polymorph used as the precursor in the synthesis of rsZnO "
    "according to high-pressure nanopowder synthesis methods is wurtzite ZnO (wZnO).\n"
    "Answer: true\n"
    "\n"
    "Expected: wurtzite ZnO\n"
    "Candidate: The product was rocksalt ZnO (rs-ZnO), not wurtzite ZnO.\n"
    "Answer: false\n"
    "\n"
    "Now it is your turn to answer:\n"
    "Expected: {expected}\n"
    "Candidate: {candidate}\n"
    "Answer with true or false."
)

# Pydantic schema for the verifier
class AreSimilar(BaseModel):
    are_the_same: bool = Field(
        ..., description="Whether the two answers name the same chemical entity."
    )

JSON_ENFORCE = (
    "Please return only the raw JSON string that strictly conforms to the following "
    "JSON schema, with no additional text: {json_schema}"
    "Example: {example}"
)


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

class Verifier:
    def __init__(self, model: str = VERIFIER_MODEL):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def _parse_json(self, text: str) -> AreSimilar:
        try:
            parsed = JsonOutputParser().invoke(text)
            return AreSimilar.model_validate(parsed)
        except Exception:
            match = re.search(r"(\{.*\})", text, re.DOTALL)
            if match:
                try:
                    parsed = JsonOutputParser().invoke(match.group(1))
                    return AreSimilar.model_validate(parsed)
                except Exception:
                    pass
        return AreSimilar(are_the_same=False)

    def verify(self, expected: str, candidate: str) -> bool:
        """Return True if candidate matches expected."""
        if candidate.strip().lower() == expected.strip().lower():
            return True

        prompt = VERIFY_PROMPT_TEMPLATE.format(
            expected=expected, candidate=candidate
        )
        schema_note = JSON_ENFORCE.format(
            json_schema=AreSimilar.model_json_schema(),
            example='{"are_the_same": true}',
        )
        full_prompt = f"{prompt}\n{schema_note}"

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=[{"role": "user", "content": [{"type": "text", "text": full_prompt}]}],
                    response_format=AreSimilar,
                    max_completion_tokens=1024,
                )
                parsed = response.choices[0].message.parsed
                return bool(getattr(parsed, "are_the_same", False))
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
                else:
                    print(f"    [verifier error] {e}")
                    return False


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "Model", "Run", "Accuracy (%)", "Avg Input Tokens", "Avg Output Tokens",
    "Total Input Tokens", "Total Output Tokens", "Total Reasoning Tokens",
    "Avg Reasoning Tokens", "Total Samples",
]


def _csv_path_for(json_path: Path) -> Path:
    """Derive the CSV filename from the JSON filename."""
    stem = json_path.stem  # e.g. responses_cot_openai_gpt-4o
    csv_name = stem.replace("responses_cot_", "results_cot_", 1) + ".csv"
    return json_path.parent / csv_name


def _model_name_from(json_path: Path) -> str:
    """Extract a model name like 'openai-gpt-4o' from the filename."""
    stem = json_path.stem  # responses_cot_openai_gpt-4o
    without_prefix = stem.replace("responses_cot_", "", 1)  # openai_gpt-4o
    # provider is the first segment before the first underscore
    parts = without_prefix.split("_", 1)
    provider = parts[0]
    model = parts[1].replace("__", "/") if len(parts) > 1 else ""
    return f"{provider}-{model}"


def _update_csv(csv_path: Path, records: list, model_name: str):
    """Recompute metrics from `records` and write (or update) the CSV."""
    total = len(records)
    if total == 0:
        return

    correct = sum(1 for r in records if r.get("is_correct", False))
    accuracy = (correct / total) * 100

    avg_in = sum(r.get("input_tokens", 0) for r in records) / total
    avg_out = sum(r.get("output_tokens", 0) for r in records) / total
    total_in = sum(r.get("input_tokens", 0) for r in records)
    total_out = sum(r.get("output_tokens", 0) for r in records)
    total_reasoning = sum(r.get("reasoning_tokens", 0) for r in records)
    avg_reasoning = total_reasoning / total

    row = {
        "Model": model_name,
        "Run": "Direct",
        "Accuracy (%)": round(accuracy, 2),
        "Avg Input Tokens": round(avg_in, 2),
        "Avg Output Tokens": round(avg_out, 2),
        "Total Input Tokens": total_in,
        "Total Output Tokens": total_out,
        "Total Reasoning Tokens": total_reasoning,
        "Avg Reasoning Tokens": round(avg_reasoning, 2),
        "Total Samples": total,
    }
    new_df = pd.DataFrame([row])

    if csv_path.exists() and csv_path.stat().st_size > 0:
        try:
            existing = pd.read_csv(csv_path)
            existing = existing[existing["Model"] != model_name]
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined.to_csv(csv_path, index=False)
            return
        except Exception as e:
            print(f"  Warning: could not merge CSV ({e}), overwriting.")
    new_df.to_csv(csv_path, index=False)


# ---------------------------------------------------------------------------
# Per-file reverification
# ---------------------------------------------------------------------------

def reverify_file(json_path: Path, verifier: Verifier):
    print(f"\n{'='*70}")
    print(f"Reverifying: {json_path.name}")
    print(f"{'='*70}")

    # Load all records
    records = []
    with open(json_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  Skipping bad line: {e}")

    if not records:
        print("  No records found, skipping.")
        return

    old_correct = sum(1 for r in records if r.get("is_correct", False))
    changed = 0

    for rec in tqdm.tqdm(records, desc="Verifying"):
        answer = rec.get("answer", "")
        expected = rec.get("expected", "")

        if not answer or not expected:
            rec["is_correct"] = False
            continue

        new_verdict = verifier.verify(expected, answer)
        if new_verdict != rec.get("is_correct"):
            changed += 1
        rec["is_correct"] = new_verdict

    new_correct = sum(1 for r in records if r.get("is_correct", False))
    total = len(records)

    print(f"  Records     : {total}")
    print(f"  Changed     : {changed}")
    print(f"  Old correct : {old_correct} ({old_correct/total*100:.1f}%)")
    print(f"  New correct : {new_correct} ({new_correct/total*100:.1f}%)")

    # Overwrite JSON file
    with open(json_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, default=str) + "\n")

    # Update CSV
    csv_path = _csv_path_for(json_path)
    model_name = _model_name_from(json_path)
    _update_csv(csv_path, records, model_name)
    print(f"  CSV updated : {csv_path.name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    verifier = Verifier()

    if len(sys.argv) > 1:
        # Single file from command line argument
        target = Path(sys.argv[1])
        if not target.is_absolute():
            target = RESPONSES_COT_DIR / target
        if not target.exists():
            print(f"File not found: {target}")
            sys.exit(1)
        reverify_file(target, verifier)
    else:
        # All JSON files in responses_cot/
        json_files = sorted(RESPONSES_COT_DIR.glob("responses_cot_*.json"))
        if not json_files:
            print(f"No response files found in {RESPONSES_COT_DIR}")
            sys.exit(0)
        print(f"Found {len(json_files)} file(s) to reverify.")
        for json_path in json_files:
            reverify_file(json_path, verifier)

    print("\nDone.")


if __name__ == "__main__":
    main()
