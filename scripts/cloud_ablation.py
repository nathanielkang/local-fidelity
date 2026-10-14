"""Single-regime run at 2000 repetitions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulate import run_study  # noqa: E402


def main() -> int:
    result = run_study(n_rep=2000, alpha=0.10, seed=20261014)
    dest = ROOT / "results" / "cloud_ablation_n2000.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "paper_table": False,
        "n_rep": result["n_rep"],
        "coverage_rate": result["coverage_rate"],
        "coverage_clopper_pearson_95": result["coverage_clopper_pearson_95"],
        "session_error_rate": result["session_error_rate"],
        "session_error_clopper_pearson_95": result["session_error_clopper_pearson_95"],
        "pass": result["pass"],
        "per_method_session_error": result["per_method_session_error"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
