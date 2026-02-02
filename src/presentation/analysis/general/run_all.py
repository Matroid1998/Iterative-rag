#!/usr/bin/env python3
"""Run all analysis scripts to generate plots."""

import sys
import subprocess
from pathlib import Path
from typing import List, Tuple


def discover_analysis_scripts() -> List[Path]:
    """Discover all analysis Python scripts in the general plot codes folder."""
    script_dir = Path(__file__).resolve().parent
    repo_root = _find_repo_root(script_dir)
    codes_dir = repo_root / "src" / "service" / "plot_codes" / "general"
    
    # Exclude the runner script itself and config
    exclude = {"config.py", "__init__.py"}
    
    scripts = []
    for script_path in sorted(codes_dir.glob("*.py")):
        if script_path.name not in exclude:
            scripts.append(script_path)
    
    return scripts


def _find_repo_root(start: Path) -> Path:
    """Find repo root by locating requirements.txt."""
    for p in [start] + list(start.parents):
        if (p / "requirements.txt").exists():
            return p
    return start.parents[-1]


def run_script(script_path: Path) -> Tuple[bool, str]:
    """Run a single analysis script."""
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            check=False,
        )
        
        if result.returncode == 0:
            return True, result.stdout
        else:
            return False, f"Exit code {result.returncode}\n{result.stderr}"
            
    except subprocess.TimeoutExpired:
        return False, "Script timed out after 5 minutes"
    except Exception as e:
        return False, str(e)


def main() -> None:
    print("="*80)
    print("RUNNING GENERAL ANALYSIS SCRIPTS")
    print("="*80)
    print()
    
    scripts = discover_analysis_scripts()
    
    if not scripts:
        print("No analysis scripts found!")
        return
    
    print(f"Found {len(scripts)} analysis script(s):\n")
    for script in scripts:
        print(f"  • {script.name}")
    print()
    
    results = []
    
    for i, script_path in enumerate(scripts, 1):
        print(f"[{i}/{len(scripts)}] Running {script_path.name}...", end=" ", flush=True)
        
        success, output = run_script(script_path)
        results.append((script_path.name, success, output))
        
        if success:
            print("✓ SUCCESS")
            # Print any output from the script
            if output.strip():
                for line in output.strip().split('\n'):
                    if line.strip():
                        print(f"    {line}")
        else:
            print("✗ FAILED")
    
    # Summary
    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print()
    
    successful = sum(1 for _, success, _ in results if success)
    failed = len(results) - successful
    
    print(f"Total: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print()
    
    if failed > 0:
        print("Failed scripts:")
        for name, success, output in results:
            if not success:
                print(f"\n  ✗ {name}")
                print(f"     Error: {output[:200]}...")
    
    print()
    print("="*80)
    
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
