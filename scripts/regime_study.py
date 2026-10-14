"""One-factor design regimes.

200 outer repetitions in each regime, alpha 0.10. Writes results/regime_study.json.
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulate import regime_grid, run_study  # noqa: E402


def run_named(payload: tuple[str, int, int]) -> dict:
    name, n_rep, seed = payload
    regime = next(item for item in regime_grid() if item.name == name)
    out = run_study(n_rep=n_rep, alpha=0.10, seed=seed, regime=regime)
    print(name, "coverage", round(out["coverage_rate"], 3), "session", round(out["session_error_rate"], 3), flush=True)
    return out


def main() -> int:
    n_rep = 200
    payloads = [(regime.name, n_rep, 20261015 + i) for i, regime in enumerate(regime_grid())]
    with ProcessPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(run_named, payloads))
    ranks = [item["ranks"]["replicate_maximum"] for item in results]
    mean_rank = float(sum(ranks) / len(ranks))
    failed = [item["regime_name"] for item in results if not item["pass"]]
    summary = {
        "paper_table": False,
        "n_rep": n_rep,
        "alpha": 0.10,
        "n_regimes": len(results),
        "mean_rank": mean_rank,
        "rank_bar": mean_rank <= 2.5,
        "failed_regimes": failed,
        "pass": len(failed) == 0 and mean_rank <= 2.5,
        "regimes": results,
    }
    dest = ROOT / "results" / "regime_study.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"pass": summary["pass"], "mean_rank": mean_rank, "failed_regimes": failed}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
