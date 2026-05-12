"""
Orchestrator: runs all 4 algorithms × 3 seeds = 12 experiments sequentially.
Each run is a subprocess for crash isolation.

Usage:  python run_all.py
"""
import subprocess
import sys
import time
import os

from training.config import CONFIG


def run_one(algo: str, seed: int):
    """Run a single experiment as a subprocess."""
    cmd = [sys.executable, "run_experiment.py", "--algo", algo, "--seed", str(seed)]
    print(f"\n{'═'*60}")
    print(f"  STARTING: {algo.upper()} | Seed {seed}")
    print(f"{'═'*60}")

    start = time.time()
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    elapsed = time.time() - start

    status = "✓ SUCCESS" if result.returncode == 0 else f"✗ FAILED (exit {result.returncode})"
    print(f"  {status}: {algo} seed={seed} in "
          f"{time.strftime('%H:%M:%S', time.gmtime(elapsed))}")

    return result.returncode == 0


def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  RL COMPARISON FRAMEWORK — FULL EXPERIMENT SUITE        ║")
    print(f"║  Algorithms: {', '.join(CONFIG.algorithms):<42} ║")
    print(f"║  Seeds: {CONFIG.seeds!s:<47} ║")
    print(f"║  Total runs: {len(CONFIG.algorithms) * len(CONFIG.seeds):<43} ║")
    print("╚══════════════════════════════════════════════════════════╝")

    total_start = time.time()
    results = {}

    for algo in CONFIG.algorithms:
        for seed in CONFIG.seeds:
            key = f"{algo}_seed{seed}"
            success = run_one(algo, seed)
            results[key] = success

    total_elapsed = time.time() - total_start

    # ── Summary ──
    print(f"\n{'═'*60}")
    print("  EXPERIMENT SUITE COMPLETE")
    print(f"  Total time: {time.strftime('%H:%M:%S', time.gmtime(total_elapsed))}")
    print(f"{'═'*60}")
    for key, ok in results.items():
        marker = "✓" if ok else "✗"
        print(f"  {marker} {key}")

    failed = [k for k, v in results.items() if not v]
    if failed:
        print(f"\n  ⚠ {len(failed)} run(s) failed: {failed}")
    else:
        print(f"\n  All {len(results)} runs completed successfully!")
        print("  Run 'python plots/generate_plots.py' to generate comparison figures.")


if __name__ == "__main__":
    main()
