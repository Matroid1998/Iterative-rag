# Iterative RAG for Scientific Multi-hop QA

Companion code for **"When Iterative RAG Beats Ideal Evidence: A Diagnostic Study in
Scientific Multi-hop Question Answering"** (TMLR) —
paper: <https://openreview.net/pdf?id=pa5TnBdyDP>.

A training-free **iterative retrieval–reasoning controller** that alternates targeted
retrieval with partial-answer updates and evidence-aware stopping, evaluated on the
chemistry multi-hop benchmark **ChemKGMultiHopQA**. The repo is organized as a single
installable package, `iterative_rag`, exposing **five command-line endpoints**.

```
chunk + index  →  ask (one question)  →  benchmark (whole dataset)
                                      ↘  diagnose (LLM-judge) →  figures (paper plots)
```

## Repository layout

```
iterative_rag/
  config.py            # central paths + LLM/planner settings (env-overridable)
  chunking/            # document normalization, chunking, corpus building
  indexing/            # embeddings + Chroma vector index + ingestion
  system/              # the iterative RAG system (planner, retriever, orchestrator, service, LLM clients)
  benchmark/           # dataset loader + iterative-RAG evaluator (LLM-as-judge verifier)
  diagnostics/         # batch LLM-as-judge auditors (coverage / faithfulness / query quality)
  figures/             # one module per paper figure + shared loaders (common.py)
  endpoints/           # the 5 runnable entry points
data/docs/             # corpus (chemrxiv/pubchem/wikipedia) + chemrxiv_qa.json + graph JSON
responses/             # model response JSONL (iterative / with-context / without-context / cot)
diagnostics_output/    # precomputed LLM-judge diagnostic JSONL (inputs to failure-mode figures)
results/               # aggregate metric CSVs
paper_figures/         # output of `irag-figures`
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate

# Full stack (chunk/index, ask, benchmark, diagnose, figures):
pip install -e .

# OR, just to regenerate the paper figures from shipped data (no torch/chromadb):
pip install -e ".[figures]"
```

API keys are read from the environment, e.g. `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, or
AWS credentials for Bedrock. Provider/model and all filesystem paths can be overridden with
`IRAG_*` environment variables (see [`iterative_rag/config.py`](iterative_rag/config.py)).

## The five endpoints

Each is available as a console script (after install) and as `python -m iterative_rag.endpoints.<name>`.

### 1. `irag-index` — chunk documents & build the vector store
```bash
irag-index                                  # index the shipped corpus into ./chroma_store
irag-index --from-graph                     # rebuild corpus text files from the graph JSON, then index
irag-index --docs-root data/docs/chemrxiv_graph_v2_texts/wikipedia \
           --collection smoke --limit 50    # quick subset
```

### 2. `irag-ask` — run the iterative RAG system on one question
```bash
irag-ask -q "Which (1,4)-linked unit is the building block of cyclodextrins?"
irag-ask -q "What is the pKa of formic acid?" --collection chemrxiv_graph --json
```
Uses the planner/composer LLM from `config` (needs an API key); falls back to a rule-based
planner when no LLM is available.

### 3. `irag-benchmark` — run iterative RAG over the whole dataset
```bash
irag-benchmark --provider openai --model gpt-4o
irag-benchmark --provider openai --model gpt-4o --limit 50 --workers 4
```
Writes `responses/responses_<provider>_<model>.jsonl` and a metrics CSV in `results/`.
(Only the Iterative-RAG regime is run here.)

### 4. `irag-figures` — regenerate the paper figures (offline)
```bash
irag-figures                       # all figures -> paper_figures/
irag-figures --only fig07,figS03   # a subset
```
Reads `responses/`, `diagnostics_output/`, and the QA dataset only — **no API calls** — so it
reproduces every paper figure from the shipped data. The figure set is exactly the paper's:
Figures 2–15 and appendix Figures S1, S2, S3, S7, S13, S14. (Figure 1 is a hand-drawn
architecture diagram and is not script-generated.)

### 5. `irag-diagnose` — regenerate the LLM-judge diagnostics
```bash
irag-diagnose --responses responses/responses_reverified/responses_openai_gpt-4o_reverified.jsonl
irag-diagnose --responses <file>.jsonl --kind coverage --limit 2   # smoke test
```
Runs the auditors (paper Figures S8/S9/S10) over a response file and writes the judgment
JSONL into `diagnostics_output/` that the failure-mode figures consume.

## Reproduce the paper figures (quickstart)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[figures]"
irag-figures
ls paper_figures/
```

## Reproduce a benchmark run end-to-end

```bash
pip install -e .
export OPENAI_API_KEY=...                 # or the relevant provider key
irag-index                                # build chroma_store (one-time, slow)
irag-benchmark --provider openai --model gpt-4o --limit 50
irag-diagnose  --responses responses/responses_openai_gpt-4o.jsonl
irag-figures
```

## Method summary

The controller (`iterative_rag/system`) runs up to a fixed budget of retrieval steps. Each
step the **planner** (an LLM returning a single JSON action) decides to retrieve a targeted
sub-query or to finalize; the **retriever** queries a Chroma index over chemistry-embedded
chunks (`BASF-AI/ChEmbed`); the **orchestrator** accumulates curated evidence and a running
partial answer; a conservative **composer** produces the final cited answer. Correctness is
judged by an LLM entity-equivalence verifier. The diagnostic suite audits retrieval coverage
gaps, anchor carry-drop, sufficiency/coverage, confidence miscalibration, composition
failures, distractor latch, and query quality.
