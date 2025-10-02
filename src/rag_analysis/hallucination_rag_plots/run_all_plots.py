"""
Run all hallucination analysis plots.
"""
import subprocess
import sys
from pathlib import Path


def run_plot(script_name: str):
    """Run a single plot script."""
    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print('='*60)
    
    script_path = Path(__file__).parent / script_name
    result = subprocess.run([sys.executable, str(script_path)], 
                          capture_output=False)
    
    if result.returncode != 0:
        print(f"WARNING: {script_name} failed with code {result.returncode}")
    else:
        print(f"✓ {script_name} completed successfully")
    
    return result.returncode


def main():
    """Run all hallucination plots in sequence."""
    plots = [
        '1_miscalibration_by_hop.py',
        '2_sufficiency_vs_coverage.py',
        '3_unsupported_claims_distribution.py',
        '4_composition_failure_root_causes.py',
        '5_composition_failure_rate.py',
        '6_sufficiency_distribution.py',
        '7_miscalibration_mix.py',
        '8_coverage_vs_confidence.py',
    ]
    
    print("="*60)
    print("HALLUCINATION ANALYSIS PLOT GENERATION")
    print("="*60)
    
    failed = []
    
    for plot in plots:
        code = run_plot(plot)
        if code != 0:
            failed.append(plot)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total plots: {len(plots)}")
    print(f"Successful: {len(plots) - len(failed)}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print("\nFailed plots:")
        for plot in failed:
            print(f"  - {plot}")
        sys.exit(1)
    else:
        print("\n✓ All plots generated successfully!")


if __name__ == '__main__':
    main()
