"""
Run all advanced analysis plots.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def run_all_plots():
    """Execute all plot scripts."""
    
    scripts = [
        '1_step_error_evolution.py',
        '2_model_comparison_radar.py',
        '3_hop_count_effects.py',
        '4_steps_per_run.py',
    ]
    
    results = []
    
    print("="*60)
    print("RUNNING ALL ADVANCED ANALYSIS PLOTS")
    print("="*60)
    
    for script in scripts:
        print(f"\n{'='*60}")
        print(f"Running: {script}")
        print("="*60)
        
        try:
            script_path = Path(__file__).parent / script
            with open(script_path, 'r') as f:
                code = f.read()
            
            # Execute the script
            exec(code, {'__name__': '__main__', '__file__': str(script_path)})
            results.append((script, True, None))
            print(f"✓ {script} completed successfully")
            
        except Exception as e:
            results.append((script, False, str(e)))
            print(f"✗ {script} failed: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    successful = sum(1 for _, success, _ in results if success)
    failed = len(results) - successful
    
    print(f"Total plots: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print("\nFailed plots:")
        for script, success, error in results:
            if not success:
                print(f"  ✗ {script}: {error}")
    else:
        print("\n✓ All plots generated successfully!")


if __name__ == '__main__':
    run_all_plots()
