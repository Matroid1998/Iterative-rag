"""Reverify JSONL response files with a custom equivalence prompt.

For each *.jsonl input file, a new sibling file named <original>_reverified.jsonl
is written containing the updated verification results.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, Tuple

from tqdm import tqdm

try:
    from src.benchmark.evaluator import StructuredLLM, Provider, AreSimilar
except ImportError:  # pragma: no cover - fallback when executed from repo root
    from benchmark.evaluator import StructuredLLM, Provider, AreSimilar


PROMPT_TEMPLATE = """Task
Decide whether Expected and Candidate name the SAME chemical entity.

What counts as the SAME
- Aliases, common vs IUPAC names, and formulas refer to the same thing (e.g., lithium chloride = LiCl; acetic acid = ethanoic acid).
- Minor packaging/context words don’t change identity: material, compound, sample, reagent, powder, nanopowder, precursor, solution.
- The Candidate may be a long sentence or paragraph with explanations; as long as it explicitly names the same entity anywhere, count it as the same.
- Subset acceptance: If Expected names a class/category and Candidate names a specific member/subclass that clearly falls under Expected, answer true (e.g., “chiral spirocycles” vs “Germanium-centered chiral spirocycles” → true).

What is NOT the same
- Different polymorph/crystal structure/phase (wurtzite ZnO vs rocksalt ZnO).
- Different charge state or ion vs neutral; cation vs anion (Li vs Li⁺; chloride ion vs HCl).
- Different oxidation state or stoichiometry (FeCl₂ vs FeCl₃).
- Different hydration/solvation (CuSO₄ vs CuSO₄·5H₂O).
- Different stereochemistry or isotopic labeling (L- vs D-; ¹³C-labeled vs unlabeled).
- Salt vs parent acid/base (acetate vs acetic acid).
- Class/family vs specific member in the opposite direction: If Expected is specific and Candidate only names a broader class (without explicitly naming the Expected entity), answer false (e.g., Expected “lithium chloride”; Candidate “alkali metal chlorides” → false).
- Candidate only mentions Expected to negate/contrast it (“not”, “instead of”, “different from”, “vs”) while naming a different main entity.

Tie-breakers (multi-entity Candidates)
- If Candidate lists multiple entities and includes the Expected entity among them, answer true unless Expected is mentioned only as a contrast/alternative and a different entity is asserted as the subject.

Decision rule
- If Candidate explicitly names the same entity as Expected (or a specific member of an Expected class), even inside a much longer explanation, answer: true. The goal is to detect whether the LLM found the correct answer.
- Otherwise, answer: false.
- The subset rule NEVER overrides the “What is NOT the same” chemistry-changing distinctions above.

Output
Answer with exactly: true or false

Examples
Expected: chiral spirocycles
Candidate: We investigated Germanium-centered chiral spirocycles because of their configurational stability.
Answer: true


Expected: wurtzite ZnO
Candidate: The ZnO polymorph used as the precursor in the synthesis of rs-ZnO was wurtzite ZnO (w-ZnO).
Answer: true

Expected: wurtzite ZnO
Candidate: The product was rocksalt ZnO (rs-ZnO), not wurtzite ZnO.
Answer: false

Now it is your turn to answer:
Expected: {expected}
Candidate: {candidate}
Answer with true or false."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_folder",
        type=Path,
        help="Folder containing JSONL response files to reverify",
    )
    parser.add_argument(
        "--verifier-provider",
        default="openai",
        choices=[p.value for p in Provider],
        help="LLM provider to use for the verifier",
    )
    parser.add_argument(
        "--verifier-model",
        default="gpt-5-mini",
        help="Model identifier to use for the verifier",
    )
    parser.add_argument(
        "--accuracy-csv",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "results"
        / "reverify_accuracies.csv",
        help="Path to the CSV file where per-model accuracies are appended",
    )
    return parser.parse_args()


def build_verifier(provider_name: str, model_id: str) -> StructuredLLM:
    provider = Provider(provider_name)
    return StructuredLLM(
        provider=provider,
        model_id=model_id,
        output_format=AreSimilar,
    )


def verify_pair(verifier: StructuredLLM, expected: str, candidate: str) -> bool:
    prompt = PROMPT_TEMPLATE.format(expected=expected, candidate=candidate)
    result = verifier(prompt)
    parsed = result.get("parsed_output") if isinstance(result, dict) else None
    if parsed is None:
        return False
    return bool(getattr(parsed, "are_the_same", False))


def reverify_file(path: Path, verifier: StructuredLLM) -> Tuple[int, int, Path]:
    total = 0
    correct = 0
    output_path = path.with_name(f"{path.stem}_reverified{path.suffix}")

    with path.open("r", encoding="utf-8") as src, output_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for raw_line in src:
            stripped = raw_line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            total += 1

            raw = record.get("raw") or {}
            expected = raw.get("expected") or ""
            candidate = record.get("candidate") or ""

            if expected and candidate:
                is_correct = verify_pair(verifier, expected, candidate)
            else:
                is_correct = False

            record["is_correct"] = bool(is_correct)
            if is_correct:
                correct += 1

            dst.write(json.dumps(record, ensure_ascii=False) + "\n")
            dst.flush()

    return total, correct, output_path


def iter_jsonl_files(folder: Path) -> Iterable[Path]:
    yield from sorted(folder.glob("*.jsonl"))


def append_accuracy_rows(csv_path: Path, rows: Iterable[dict]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp",
        "folder",
        "file_name",
        "total_questions",
        "correct_answers",
        "accuracy",
        "verifier_provider",
        "verifier_model",
    ]
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    if not args.input_folder.is_dir():
        raise FileNotFoundError(f"Input folder not found: {args.input_folder}")

    verifier = build_verifier(args.verifier_provider, args.verifier_model)

    accuracy_rows = []
    timestamp = datetime.now().isoformat(timespec="seconds")

    jsonl_files = list(iter_jsonl_files(args.input_folder))
    progress = tqdm(jsonl_files, dynamic_ncols=True)
    progress.set_postfix({"verifier": args.verifier_model})
    for jsonl_path in progress:
        model_label = jsonl_path.stem.replace("responses_", "", 1)
        progress.set_description(f"Reverifying {model_label}")
        total, correct, output_path = reverify_file(jsonl_path, verifier)
        accuracy = (correct / total) if total else 0.0
        accuracy_rows.append(
            {
                "timestamp": timestamp,
                "folder": str(args.input_folder),
                "file_name": output_path.name,
                "total_questions": total,
                "correct_answers": correct,
                "accuracy": f"{accuracy:.4f}",
                "verifier_provider": args.verifier_provider,
                "verifier_model": args.verifier_model,
            }
        )

    append_accuracy_rows(args.accuracy_csv, accuracy_rows)


if __name__ == "__main__":
    main()
