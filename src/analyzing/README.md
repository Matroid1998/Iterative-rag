# Analysis Scripts - README

## Overview

This directory contains analysis scripts that generate plots from JSONL response files. The scripts have been updated to be **scalable** and **non-hardcoded**, automatically discovering and processing all available model response files.

## Key Features

✅ **Auto-discovery**: Scripts automatically find all JSONL files in the responses directory  
✅ **Dynamic model recognition**: New models are automatically detected and labeled  
✅ **Centralized configuration**: All settings in one place (`config.py`)  
✅ **Flexible paths**: Searches multiple response directories  
✅ **Runner script**: Execute all analysis scripts with one command  

## Quick Start

### Run All Analysis Scripts

```bash
# From the analyzing directory
cd /media/torontoai/Iterative-rag/src/analyzing
source ../../venv/bin/activate
python run_all_analysis.py
```

This will:
- Discover all analysis scripts
- Run them sequentially
- Report success/failure for each
- Generate all plots in `/media/torontoai/Iterative-rag/src/plots`

### Run Individual Scripts

```bash
# Example: Generate average output tokens plot
python average_output_tokens.py

# Example: Generate reasoning tokens analysis
python average_reasoning_tokens.py
```

## Configuration

### Response Directories

The system searches for JSONL files in these directories (in order):
1. `src/responses_reverified/` (primary)
2. `src/responses/`
3. `src/response-jsonl-with-context/`
4. `src/response-jsonl-without-context/`

To add new directories, edit `RESPONSE_DIRS` in `config.py`.

### Model Name Mapping

Model names are automatically normalized and displayed in human-readable format. To customize:

Edit `MODEL_NAME_PATTERNS` in `config.py`:

```python
MODEL_NAME_PATTERNS = {
    "mistral-large": "Mistral Large",
    "gpt-4o": "GPT-4o",
    # Add new models here
}
```

### Reasoning Model Detection

Reasoning models are automatically detected based on patterns in their filenames:

```python
REASONING_INDICATORS = [
    "reasoning",
    "o1-",
    "o3-",
    "qwq",
    "deepseek-r1",
]
```

## File Structure

```
analyzing/
├── config.py                          # Centralized configuration
├── run_all_analysis.py               # Runner script
├── average_output_tokens.py          # Output tokens analysis
├── average_reasoning_tokens.py       # Reasoning tokens analysis
├── output_tokens_per_hop.py          # Per-hop output analysis
├── reasoning_tokens_per_hop.py       # Per-hop reasoning analysis
├── wrong_answers_per_hop.py          # Wrong answers by hop
├── wrong_tokens_heatmap.py           # Token usage heatmaps
├── plot_*.py                          # Various plotting scripts
└── README.md                          # This file
```

## Adding New Models

No code changes needed! Just add new JSONL files to the responses directory:

```bash
# Example: New model results
cp new_model_responses.jsonl src/responses_reverified/
```

The scripts will automatically:
- Detect the new file
- Parse the model name
- Generate appropriate display names
- Include it in all relevant plots

## JSONL File Format

Expected format for response files:

```json
{
  "question": "What is...",
  "is_correct": true,
  "output_tokens": 1234,
  "reasoning_tokens": 567,  // Optional, for reasoning models
  "number_of_hops": 2,
  "raw": { ... }
}
```

Alternative format with `usage` object:

```json
{
  "question": "What is...",
  "is_correct": true,
  "usage": {
    "output_tokens": 1234,
    "reasoning_tokens": 567
  },
  "number_of_hops": 2
}
```

Both formats are supported with automatic fallback.

## Generated Plots

All plots are saved to `/media/torontoai/Iterative-rag/src/plots/`:

- `average_output_tokens.png` - Average output tokens by correctness
- `average_output_tokens_reasoning.png` - Reasoning models only
- `average_output_tokens_non_reasoning.png` - Non-reasoning models only
- `average_reasoning_tokens.png` - Reasoning token comparison
- `output_tokens_per_hop.png` - Token usage by hop count
- `reasoning_tokens_per_hop.png` - Reasoning tokens by hop
- `wrong_answers_per_hop.png` - Error distribution
- `wrong_output_tokens_heatmap.png` - Token vs hop heatmap
- `wrong_reasoning_tokens_heatmap.png` - Reasoning token heatmap
- ... and many more

## Troubleshooting

### No JSONL files found

Check that response files exist in one of the configured directories:

```bash
ls src/responses_reverified/*.jsonl
```

### Missing dependencies

Activate the virtual environment:

```bash
source venv/bin/activate
pip install matplotlib numpy
```

### Script fails for specific model

Check the JSONL file format:

```bash
head -n 1 src/responses_reverified/your_model.jsonl | python -m json.tool
```

### Model name not displaying correctly

Add a custom mapping in `config.py`:

```python
MODEL_NAME_PATTERNS = {
    "your-model-key": "Your Model Display Name",
}
```

## Script Details

### Core Analysis Scripts

| Script | Description | Output |
|--------|-------------|--------|
| `average_output_tokens.py` | Compares output tokens for correct vs wrong answers | 3 plots (all, reasoning, non-reasoning) |
| `average_reasoning_tokens.py` | Analyzes reasoning token usage | 1 plot |
| `output_tokens_per_hop.py` | Token usage by hop count | 1 multi-panel plot |
| `reasoning_tokens_per_hop.py` | Reasoning tokens by hop | 1 multi-panel plot |
| `wrong_answers_per_hop.py` | Wrong answer distribution | 1 multi-panel plot |
| `wrong_tokens_heatmap.py` | Token usage heatmaps | 2 plots |

### Utility Functions

All scripts import from `config.py`:

- `get_responses_dir()` - Find the responses directory
- `discover_jsonl_files()` - List all JSONL files
- `discover_reasoning_jsonl_files()` - List reasoning model files
- `get_display_name()` - Get human-readable model name
- `is_reasoning_model()` - Check if model is reasoning-capable
- `normalize_model_key()` - Normalize filename to model key

## Development

### Adding a New Analysis Script

1. Create your script in the `analyzing/` directory
2. Import from `config.py`:

```python
from config import (
    get_responses_dir,
    PLOTS_DIR,
    get_display_name,
    discover_jsonl_files,
)
```

3. Use the discovery functions:

```python
def main():
    responses_dir = get_responses_dir()
    jsonl_files = discover_jsonl_files(responses_dir)
    
    for path in jsonl_files:
        display_name = get_display_name(path.stem)
        # Your analysis here
        
    # Save plot
    plt.savefig(PLOTS_DIR / "my_plot.png")
```

4. Run it via the runner script:

```bash
python run_all_analysis.py
```

### Testing

Test individual scripts:

```bash
python average_output_tokens.py
```

Run all with error handling:

```bash
python run_all_analysis.py
```

## Future Enhancements

Potential improvements:

- [ ] Parallel execution of independent scripts
- [ ] Configuration file (YAML/JSON) for model mappings
- [ ] Interactive plots with Plotly
- [ ] Automatic model grouping by family
- [ ] Statistical significance testing
- [ ] Export results to CSV/Excel

## License

Part of the Iterative-rag project.

## Contact

For issues or questions, please refer to the main repository.
