"""Tiny synthetic smoke for the local-fidelity certificate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import certificate as cert  # noqa: E402
from simulate import run_repetition, make_ablation_population  # noqa: E402


def fail(message: str) -> None:
    raise SystemExit(message)


def main() -> int:
    rng = np.random.default_rng(7)
    raw = np.array(
        [
            [0, 0],
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 0],
            [1, 1],
        ],
        dtype=int,
    )
    edit = {(1, 1)}
    kept_key = cert.screen_release(raw, edit, 2, [cert.key_projection()])
    if kept_key.shape[0] > raw.shape[0]:
        fail("screen increased the release")
    if any((int(a), int(b)) == (1, 1) for a, b in kept_key):
        fail("screen left a declared edit record")
    if kept_key.shape[0] != 5:
        fail("key-only screen dropped a cell that already cleared the floor")
    kept_both = cert.screen_release(
        raw,
        edit,
        2,
        [cert.key_projection(), cert.outcome_projection()],
    )
    if kept_both.shape[0] > kept_key.shape[0]:
        fail("screen increased the release")
    if kept_both.shape[0] == 0 or np.any(kept_both[:, 1] == 1):
        # The outcome-1 cell has count 1 after the edit drop, so it must go.
        if np.any(kept_both[:, 1] == 1):
            fail("registered outcome projection did not drop a count below q")
    cascade = np.array([[0, 0], [0, 1], [1, 0]], dtype=int)
    cascaded = cert.screen_release(
        cascade,
        set(),
        2,
        [cert.key_projection(), cert.outcome_projection()],
    )
    if cascaded.shape[0] > cascade.shape[0]:
        fail("screen increased the release")
    if cascaded.shape[0] != 0:
        fail("screen did not recompute projections after a deletion")

    cells = np.full(cert.N_KEY, 0.4 / (cert.N_KEY - 1))
    cells[0] = 0.6
    joints = np.zeros(cert.N_KEY * cert.N_OUT)
    joints[0] = 0.45
    joints[1] = 0.15
    for key in range(1, cert.N_KEY):
        joints[key * cert.N_OUT] = cells[key]
    center = np.concatenate([cells, joints])
    replicates = np.tile(center, (30, 1))
    noise_cols = center.shape[0] - 1
    replicates[:, :noise_cols] += rng.normal(scale=0.01, size=(30, noise_cols))
    band = cert.band_from_replicates(center, replicates, 0.10)
    if band.se[-1] != 0.0 or band.half[-1] != 0.0:
        fail("a coordinate with standard error 0 does not have half-width 0")
    if band.half[-1] >= 1e-8 and band.se[-1] == 0.0:
        fail("zero standard error was replaced by a 1e-8 floor")

    point = center.copy()
    point_half = np.zeros(center.shape[0])
    point_ok = cert._box_feasible(point, point_half, False)
    point_region = cert.Region(point, point_half, point_half, 0.0, point_ok, False)
    if not point_ok:
        fail("point region on the simplex was marked empty")
    if cert.fidelity_reply(point_region, 0, np.array([0.75, 0.25]), 0.25) != "certify":
        fail("a positive point inside the tolerance was not certified")
    if cert.fidelity_reply(point_region, 0, np.array([0.25, 0.75]), 0.25) != "reject":
        fail("a positive point outside the tolerance was not rejected")

    wide_joints = np.full(cert.N_KEY * cert.N_OUT, 1.0 / (cert.N_KEY * cert.N_OUT))
    wide_cells = np.full(cert.N_KEY, 1.0 / cert.N_KEY)
    wide_center = np.concatenate([wide_cells, wide_joints])
    wide_half = np.full(wide_center.shape[0], 0.5)
    wide_ok = cert._box_feasible(wide_center, wide_half, False)
    wide = cert.Region(wide_center, wide_half, wide_half, 1.0, wide_ok, False)
    if not wide_ok:
        fail("wide region was empty")
    wide_reply = cert.fidelity_reply(wide, 0, np.array([0.5, 0.5]), 0.25)
    if wide_reply == "certify":
        fail("a fidelity clause whose region includes cell mass 0 was certified")

    empty_cells = cells.copy()
    empty_cells[0] = 0.9
    empty_center = np.concatenate([empty_cells, joints])
    empty_half = np.zeros(empty_center.shape[0])
    empty_ok = cert._box_feasible(empty_center, empty_half, False)
    if empty_ok:
        fail("inconsistent box was treated as nonempty")
    empty = cert.Region(empty_center, empty_half, empty_half, 0.0, False, False)
    if cert.fidelity_reply(empty, 0, np.array([0.5, 0.5]), 0.25) != "inconclusive":
        fail("an empty region was not inconclusive")
    if cert.support_reply(empty, 0, 0.05) != "inconclusive":
        fail("an empty region was not inconclusive")

    silent = np.array([[1, 0], [1, 0], [1, 0]], dtype=int)
    if cert.clause_replies(point_region, silent, [0], 0.05, 0.25):
        fail("a clause was issued for a cell the release does not populate")

    pop = make_ablation_population()
    rep_rng = np.random.default_rng(11)
    gen_rng = np.random.default_rng(int(rep_rng.integers(0, 2**31 - 1)))
    out = run_repetition(pop, rep_rng, gen_rng, alpha=0.10, n_boot=49)
    if out["n_dev"] == 0 or out["n_aud"] == 0:
        fail("PSU split left one side empty")
    if out["n_dev_psu"] == 0 or out["n_aud_psu"] == 0:
        fail("PSU split left one side empty")
    if out["screen_grew"]:
        fail("screen increased the release")
    if not out["zero_se_point_band"]:
        fail("a coordinate with standard error 0 does not have half-width 0")

    payload = {
        "smoke": True,
        "paper_table": False,
        "n_dev": out["n_dev"],
        "n_aud": out["n_aud"],
        "n_release_raw": out["n_raw"],
        "n_release_kept": out["n_kept"],
        "screen_grew": out["screen_grew"],
        "zero_se_half_width": 0.0,
        "zero_mass_fidelity_reply": wide_reply,
        "empty_region_reply": "inconclusive",
        "utility_ok": out["utility_ok"],
        "covered": out["covered"],
        "certified": wide_reply == "certify",
        "support_reply": cert.support_reply(point_region, 0, 0.05),
        "fidelity_reply": wide_reply,
    }
    dest = ROOT / "results" / "smoke_method1.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
