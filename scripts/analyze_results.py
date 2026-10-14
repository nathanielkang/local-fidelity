"""Print coverage, session error, and rank from a saved study. ASCII only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _bar(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def main() -> int:
    path = ROOT / "results" / "ablation_method1.json"
    if not path.is_file():
        print("missing results/ablation_method1.json")
        return 1
    payload = json.loads(path.read_text(encoding="utf-8"))
    cov_lo, cov_hi = payload["coverage_clopper_pearson_95"]
    ses_lo, ses_hi = payload["session_error_clopper_pearson_95"]
    validity = payload["validity"]
    print("Local-fidelity study")
    print(
        "coverage rate: %.4f  Clopper-Pearson 95%% [%.4f, %.4f]  %s"
        % (payload["coverage_rate"], cov_lo, cov_hi, _bar(validity["coverage_pass"]))
    )
    print(
        "session-error rate: %.4f  Clopper-Pearson 95%% [%.4f, %.4f]  %s"
        % (payload["session_error_rate"], ses_lo, ses_hi, _bar(validity["session_pass"]))
    )
    power = payload["power_reversal"]
    if isinstance(power, dict):
        print(
            "power against reversal: "
            + "  ".join("%s %.4f" % (name, power[name]) for name in ("pointwise", "bonferroni", "replicate_maximum"))
        )
    else:
        print("power against reversal: %.4f" % power)
    print(
        "inconclusive fraction: %.4f" % payload["inconclusive_fraction"]
    )
    print(
        "mean retained size: %.4f" % payload["mean_retained_size"]
    )
    scores = payload["per_method_session_error"]
    ranks = payload["ranks"]
    for name in ("pointwise", "bonferroni", "replicate_maximum"):
        print(
            "%s session-error: %.4f  rank: %.2f" % (name, scores[name], ranks[name])
        )
    if "marginal_band_session_error" in payload:
        print(
            "marginal-band session-error: %.4f" % payload["marginal_band_session_error"]
        )
    print("rank bar (replicate-maximum in the top 2): %s" % _bar(payload["rank_pass"]))
    print("validity and rank: %s" % _bar(payload["pass"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
