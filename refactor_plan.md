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

## Current Code → Layer Mapping (excluding protocols + data)

**Excluded (not mapped):**
- Protocols: `src/protocols/*`
- Data/output folders: `src/docs`, `src/results`, `src/plots`, `src/response-jsonl-*`, `src/responses_reverified`, `src/rag_analysis/output`

**Repository Layer (data access + indexing + retrieval primitives)**
- `src/repo/index/*` → repository/index
- `src/repo/embeddings/*` → repository/embeddings
- `src/repo/retrievers/*` → repository/retrievers
- `src/repo/utils/*` → repository/utils

**Service Layer (business logic, orchestration, evaluation, LLM adapters)**
- `src/service/*` → service/*
- `src/config/*` → service/config (centralized configuration)
- `src/repo/planning/planner_iface.py` → service/planning (planning interfaces + JSON planner)
- `src/benchmark/evaluator.py` → service/evaluation (to split into smaller modules)
- `src/analyzing/*` → service/plot_codes/general/*
- `src/rag_analysis/*_plots/*` → service/plot_codes/rag/<category>/*

**Presentation Layer (CLI, runners only)**
- `src/scripts/*` → presentation/cli/*
- `src/reverify/*` → presentation/cli/reverify/*
- `presentation/analysis/*/run_all.py` → runner-only entrypoints (no plot code here)

## Presentation Layer: Required Entry Points

You requested **three top-level presentation commands**:
1) **Index**: runs ingestion + indexing pipeline  
2) **Analyze/Plot All**: runs all analyses + plots end-to-end  
3) **Benchmark Evaluate**: runs evaluation/benchmarking (current `evaluator.py`)

In addition, **each analysis category needs its own “run-only-this-category” command**.

### Proposed Presentation Structure (Runners Only)
```
src/presentation/
├── cli/
│   ├── index.py               # single entrypoint for indexing
│   ├── analyze_all.py         # run all analyses/plots (global)
│   └── benchmark_eval.py      # wrapper for evaluation/benchmark runs
└── analysis/
    ├── general/
    │   └── run_all.py
    └── rag/
        ├── hallucination/
        │   └── run_all.py
        ├── quality/
        │   └── run_all.py
        ├── coverage_gap/
        │   └── run_all.py
        ├── cross_system/
        │   └── run_all.py
        └── advanced/
            └── run_all.py
```

### Plot Code Location (Not in Presentation)
Plot/analysis scripts will live under **Service**, not Presentation:
```
src/service/plot_codes/
├── general/        # former src/analyzing/*.py
└── rag/
    ├── hallucination/
    ├── quality/
    ├── coverage_gap/
    ├── cross_system/
    └── advanced/
```

### Entry Point Behavior (Concrete)
- `presentation/cli/index.py`  
  - calls repository + service indexing flows (wraps current `scripts/index_data.py`)
  - accepts corpus path + collection + embedding config

- `presentation/cli/analyze_all.py`  
  - runs all category runners + general analysis in a consistent order
  - aggregates status + exit codes

- `presentation/cli/benchmark_eval.py`  
  - invokes evaluation pipeline (current `benchmark/evaluator.py`, later refactored)
  - supports model/provider selection + output directory

### Category-Only Runners (Required)
Each category will have a `run_all.py` that only runs that category’s scripts.
This is already partially present in `rag_analysis/*/run_all_plots.py` and will be unified to:
```
python -m presentation.analysis.rag.hallucination.run_all
python -m presentation.analysis.rag.quality.run_all
python -m presentation.analysis.rag.coverage_gap.run_all
python -m presentation.analysis.rag.cross_system.run_all
python -m presentation.analysis.rag.advanced.run_all
python -m presentation.analysis.general.run_all
```

## Redundant Code Cleanup (for analysis/plots)

You called out redundancy across `src/analyzing`, `src/plots`, and `src/rag_analysis`.  
This will be addressed by consolidating common logic into shared utilities.

### Planned de-duplication work
- **Shared loading utilities**: one module for JSONL + CSV loading, model name normalization, and response discovery.
- **Shared plotting utilities**: consistent styles, colors, labeling, and saving behavior.
- **Shared path config**: all scripts use one centralized paths module (no absolute paths).
- **Shared CLI runner**: one runner pattern across all categories (consistent logging + error handling).

### Target Utility Modules (Draft)
```
src/service/plot_codes/_core/
├── paths.py          # centralized paths
├── data_loaders.py   # JSONL + CSV loaders
├── model_names.py    # normalization + display mapping
├── plot_style.py     # matplotlib defaults + shared colors
└── runner.py         # reusable run_all driver
```

## Plot Output Folder (New Requirement)
All generated plots will be saved under a **single root folder** with per-category subfolders:
```
data/plots/
├── general/
├── rag/
│   ├── hallucination/
│   ├── quality/
│   ├── coverage_gap/
│   ├── cross_system/
│   └── advanced/
```
If new analysis categories are added, a corresponding subfolder will be created automatically.

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
- Keep `src/presentation/analysis/*` as runner-only; move all plot scripts into `src/service/plot_codes/*`.
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
