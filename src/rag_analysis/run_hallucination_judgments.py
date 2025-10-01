"""Convenient script to run hallucination judgments on Claude 3.7 Sonnet Reasoning responses."""

import subprocess
import sys
from pathlib import Path

def main():
    """Run hallucination judgments with predefined settings."""
    script_dir = Path(__file__).parent
    hallucination_script = script_dir / "hallucination_judgment.py"
    
    # Default paths
    input_jsonl = (
        script_dir.parent 
        / "responses_reverified" 
        / "responses_bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0-reasoning_reverified.jsonl"
    )
    output_jsonl = script_dir / "hallucination_judgments_claude37_reasoning.jsonl"
    
    # Build command
    cmd = [
        sys.executable, str(hallucination_script),
        "--jsonl", str(input_jsonl),
        "--output", str(output_jsonl),
        "--model", "gpt-4o-mini",  # Using gpt-4o-mini as it's more available
        "--num-workers", "6"
    ]
    
    # Add any command line arguments passed to this script
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])
    
    print(f"Running: {' '.join(cmd)}")
    print(f"Input: {input_jsonl}")
    print(f"Output: {output_jsonl}")
    print("=" * 60)
    
    # Run the command
    try:
        subprocess.run(cmd, check=True)
        print(f"\n✅ Hallucination judgments completed! Results saved to: {output_jsonl}")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running hallucination judgments: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n🛑 Interrupted by user")
        sys.exit(130)

if __name__ == "__main__":
    main()