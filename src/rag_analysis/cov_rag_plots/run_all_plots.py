"""
Master script to generate all coverage gap analysis plots.
Run this to create all visualizations at once.
"""
import subprocess
import sys
from pathlib import Path

def run_plot_script(script_path):
    """Run a single plot script and report status."""
    script_name = script_path.name
    print(f"\n{'='*80}")
    print(f"Running: {script_name}")
    print(f"{'='*80}")
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=False,
            text=True,
            check=True
        )
        print(f"✓ {script_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {script_name} failed with error code {e.returncode}")
        return False
    except Exception as e:
        print(f"✗ {script_name} failed: {e}")
        return False


def main():
    # Get script directory
    script_dir = Path(__file__).resolve().parent
    
    # Find all numbered plot scripts
    plot_scripts = sorted(script_dir.glob("[0-9]_*.py"))
    
    if not plot_scripts:
        print("No plot scripts found!")
        return
    
    print("="*80)
    print("COVERAGE GAP ANALYSIS - PLOT GENERATION")
    print("="*80)
    print(f"Found {len(plot_scripts)} plot scripts to run\n")
    
    # Run each script
    results = {}
    for script in plot_scripts:
        success = run_plot_script(script)
        results[script.name] = success
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    successful = sum(results.values())
    total = len(results)
    
    print(f"\nCompleted: {successful}/{total} plots generated successfully\n")
    
    for script_name, success in results.items():
        status = "✓" if success else "✗"
        print(f"{status} {script_name}")
    
    if successful == total:
        print("\n🎉 All plots generated successfully!")
        print(f"📊 Check the plots in: {script_dir}")
    else:
        print(f"\n⚠️  {total - successful} plot(s) failed. Check the output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
