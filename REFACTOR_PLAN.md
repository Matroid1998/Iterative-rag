# Iterative-rag Refactor Plan (Repository / Service / Presentation)

Date: 2026-02-02

## Summary
This plan organizes the codebase into three layers (Repository, Service, Presentation), consolidates analysis/plots, and unifies data outputs. The goal is cleaner separation, reproducible runs, and simpler onboarding.

## Observed Problems (from current `src/`)
- **Code + data mixed**: results, plots, response JSONL, and corpus data live inside `src/`.
- **Analysis split across multiple systems**: `src/analyzing` and `src/rag_analysis/*_plots` each have their own runners and path conventions.
- **Layer leakage**: `service/structured_llm_adapter.py` imports from `benchmark/evaluator.py` (evaluation layer), creating tight coupling.
- **Monolithic evaluator**: `src/benchmark/evaluator.py` contains provider SDKs, schemas, I/O, eval logic, CLI, and plotting in one file.
- **Naming inconsistencies**: e.g., `embeddeing_models.py` typo, `judgement/judgment` variations.
- **Hardcoded absolute paths**: multiple READMEs and some scripts embed `/media/torontoai/...` paths.

## Target Architecture (Draft)
```
root/
├── src/
│   ├── repository/          # embeddings, index, retrievers, utils
│   ├── service/             # orchestrator, rag service, llm adapters, evaluation services
│   ├── presentation/        # CLI tools, analysis runners, plotting entrypoints
│   ├── protocols/           # stable interfaces and data contracts
│   ├── config/              # settings + path configuration
│   └── benchmark/           # (optional) evaluation modules if kept separate
├── data/
│   ├── corpus/              # docs (original input data)
│   ├── responses/           # response JSONL (with/without context, reverified)
│   ├── analysis/            # rag_analysis/output and intermediate artifacts
│   ├── results/             # CSV summaries, metrics
│   └── plots/               # generated figures
└── scripts/                 # optional: thin launchers for presentation layer
```

## Plan (Sequential)
### 1) Inventory + Migration Map
- Enumerate all folders/files under `src/` and map them to Repository/Service/Presentation.
- Draft a migration table: **old path → new path**.
- Decide canonical data folders and finalize naming conventions.

### 2) Centralize Paths + Data Relocation
- Create a single path/config module (e.g., `src/config/paths.py`).
- Move:
  - `src/docs` → `data/corpus`
  - `src/response-jsonl-*`, `src/responses_reverified` → `data/responses/`
  - `src/rag_analysis/output` → `data/analysis/`
  - `src/results` → `data/results/`
  - `src/plots` → `data/plots/`
- Update analysis scripts and service code to use centralized paths.

### 3) Repository Layer Cleanup
- Move `src/repo/*` to `src/repository/*` (or keep `src/repo` with shims).
- Fix naming issues (`embeddeing_models.py` → `embedding_models.py`).
- Keep indexing utilities clean and layer-appropriate.

### 4) Service Layer Cleanup
- Split `src/benchmark/evaluator.py` into:
  - `service/llm_clients/*`
  - `service/evaluation/*`
  - `service/usage_tracking/*`
- Remove service → benchmark circular dependencies.
- Make orchestrator and rag services depend only on Repository + Protocols.

### 5) Presentation Layer Consolidation
- Move `src/scripts` into `src/presentation/cli/`.
- Merge `src/analyzing` + `src/rag_analysis/*_plots` into `src/presentation/analysis/` with clear categories.
- Add one unified runner with category selection (e.g., `python -m presentation.analysis.run_all --category hallucination`).

### 6) Docs + Validation
- Update READMEs with new relative paths.
- Add smoke checks for:
  - indexing flow
  - benchmark/eval run
  - analysis/plot generation

## Folder-by-Folder Execution Order (Suggested)
1. **Data/results consolidation** (move data out of `src` + path config)
2. **Repository layer** (`repo/` → `repository/`, fix naming)
3. **Benchmark/evaluator refactor** (split modules + remove coupling)
4. **Analysis/plots consolidation** (single runner + categories)

## Open Decisions (Confirm)
- **Data location**: `data/` at repo root vs `src/data/`?
- **Rename**: `src/repo` → `src/repository` or keep and add shims?
- **Analysis taxonomy**: keep existing `rag_analysis` categories and fold `analyzing` into them, or redefine categories?

## Next Step
Pick the first folder to fix:
1) Data/results consolidation
2) Repository layer (indexing + Chroma)
3) Benchmark/evaluator refactor
4) Analysis/plots consolidation
