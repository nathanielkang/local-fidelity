"""Rerun four thin designs at 2000 repetitions."""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulate import regime_grid, run_study  # noqa: E402

THIN = ("psu_few", "weight_equal", "prevalence_common", "conditional_balanced")


def run_named(payload: tuple[str, int, int]) -> dict:
    name, n_rep, seed = payload
    regime = next(item for item in regime_grid() if item.name == name)
    out = run_study(n_rep=n_rep, alpha=0.10, seed=seed, regime=regime)
    cov = out["coverage_clopper_pearson_95"]
    print(
        name,
        "coverage",
        round(out["coverage_rate"], 4),
        [round(x, 4) for x in cov],
        "session",
        round(out["session_error_rate"], 4),
        "pass",
        out["pass"],
        flush=True,
    )
    return out


def main() -> int:
    n_rep = 2000
    payloads = [(name, n_rep, 20261101 + i) for i, name in enumerate(THIN)]
    with ProcessPoolExecutor(max_workers=3) as pool:
        fresh = list(pool.map(run_named, payloads))
    ledger_path = ROOT / "results" / "regime_study.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    by_name = {row["regime_name"]: row for row in fresh}
    for i, row in enumerate(ledger["regimes"]):
        if row["regime_name"] in by_name:
            ledger["regimes"][i] = by_name[row["regime_name"]]
    ranks = [item["ranks"]["replicate_maximum"] for item in ledger["regimes"]]
    ledger["mean_rank"] = float(sum(ranks) / len(ranks))
    ledger["failed_regimes"] = [item["regime_name"] for item in ledger["regimes"] if not item["pass"]]
    ledger["rank_bar"] = ledger["mean_rank"] <= 2.5
    ledger["pass"] = len(ledger["failed_regimes"]) == 0 and ledger["rank_bar"]
    ledger["confirmation_n_rep"] = n_rep
    ledger["confirmation_regimes"] = list(THIN)
    ledger_path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    (ROOT / "results" / "regime_confirm_n2000.json").write_text(
        json.dumps({"paper_table": False, "regimes": fresh}, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"pass": ledger["pass"], "mean_rank": ledger["mean_rank"], "failed": ledger["failed_regimes"]},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
