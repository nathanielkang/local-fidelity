"""Baseline design, 200 repetitions.

200 outer repetitions, alpha 0.10. Generator seeds are nested inside the
repetition. Writes results/ablation_method1.json and results/status.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulate import run_study  # noqa: E402


def main() -> int:
    result = run_study(n_rep=200, alpha=0.10, seed=20261015)
    dest = ROOT / "results" / "ablation_method1.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2), encoding="utf-8")
    status = {
        "paper_table": False,
        "smoke_exit_code": 0,
        "ablation": "results/ablation_method1.json",
        "n_rep": result["n_rep"],
        "alpha": result["alpha"],
        "coverage_rate": result["coverage_rate"],
        "coverage_clopper_pearson_95": result["coverage_clopper_pearson_95"],
        "session_error_rate": result["session_error_rate"],
        "session_error_clopper_pearson_95": result["session_error_clopper_pearson_95"],
        "coverage_pass": result["validity"]["coverage_pass"],
        "session_pass": result["validity"]["session_pass"],
        "rank_pass": result["rank_pass"],
        "pass": result["pass"],
    }
    (ROOT / "results" / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
