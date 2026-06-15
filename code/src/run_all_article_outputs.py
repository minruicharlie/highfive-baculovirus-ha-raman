from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "code" / "src"

MODEL_STEPS = [
    "run_plsr_baselines.py",
    "run_via_bounded_logit_process_selectsvr.py",
    "run_dim_logit_range_process_selectsvr.py",
    "run_ha_ordinal_tolerance_ablation.py",
    "run_via_coarse_weighting_supplement.py",
]

ARTICLE_STEPS = [
    "compile_article_tables.py",
    "make_final_figures.py",
]


def run_script(script_name: str) -> None:
    script = SRC / script_name
    if not script.exists():
        raise FileNotFoundError(f"Missing article script: {script}")
    start = time.time()
    print(f"\n[run] {script_name}")
    subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)
    print(f"[ok] {script_name} ({time.time() - start:.1f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the final article result pipeline.")
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Only rebuild manuscript tables and figures from existing model outputs.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the planned script order without running anything.",
    )
    args = parser.parse_args()

    steps = ([] if args.skip_models else MODEL_STEPS) + ARTICLE_STEPS
    if args.list:
        for step in steps:
            print(step)
        return

    print(f"Article package root: {ROOT}")
    for step in steps:
        run_script(step)
    print("\nArticle outputs regenerated.")


if __name__ == "__main__":
    main()
