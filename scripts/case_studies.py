"""Case-study diagnostics. Not a coverage claim.

Fifty repetitions on each public table. The certificate replies, the screen,
and the runtime are recorded. The files are not treated as known populations
for coverage or session error.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import certificate as cert  # noqa: E402
from simulate import _psu_table, bootstrap_replicates  # noqa: E402

BANK = ROOT / "data" / "bank" / "bank" / "bank-full.csv"
CREDIT = ROOT / "data" / "credit_default" / "credit_default.csv"
ACS = ROOT / "data" / "acs2024" / "psam_p06.csv"

N_STRATA = 4
N_PER = 25
N_PSU = 40
N_DRAW = 24
N_RELEASE = 80
N_BOOT = 199
N_REP = 50
ALPHA = 0.10
TAU = 0.05
ETA = 0.25
MULTIPLICITY = 2


def _rank_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    bins = np.empty(values.shape[0], dtype=int)
    for b, idx in enumerate(np.array_split(order, n_bins)):
        bins[idx] = b
    return bins


def _collapse_labels(labels: np.ndarray, n_keep: int = 5) -> tuple[np.ndarray, dict[str, int]]:
    """Most common labels keep their own keys. The rest share the last key."""
    values, counts = np.unique(labels.astype(str), return_counts=True)
    order = np.argsort(-counts, kind="mergesort")
    kept = [str(values[i]) for i in order[:n_keep]]
    mapping = {name: i for i, name in enumerate(kept)}
    other = n_keep
    keys = np.array([mapping.get(str(v), other) for v in labels], dtype=int)
    mapping["other"] = other
    return keys, mapping


def _frame(stratum: np.ndarray, key: np.ndarray, outcome: np.ndarray, n_psu: int, rng: np.random.Generator) -> dict:
    """Equal primary-sampling-unit counts in every age stratum."""
    chosen_s: list[np.ndarray] = []
    chosen_k: list[np.ndarray] = []
    chosen_y: list[np.ndarray] = []
    chosen_u: list[np.ndarray] = []
    for s in range(N_STRATA):
        idx = np.flatnonzero(stratum == s)
        take = n_psu * N_PER
        if idx.size < take:
            raise RuntimeError(f"stratum {s} has {idx.size} rows, need {take}")
        pick = rng.choice(idx, size=take, replace=False)
        chosen_s.append(np.full(take, s, dtype=int))
        chosen_k.append(key[pick])
        chosen_y.append(outcome[pick])
        psu = np.repeat(np.arange(s * n_psu, (s + 1) * n_psu), N_PER)
        chosen_u.append(psu)
    return {
        "stratum": np.concatenate(chosen_s),
        "psu": np.concatenate(chosen_u),
        "key": np.concatenate(chosen_k),
        "outcome": np.concatenate(chosen_y),
        "n_psu": n_psu,
    }


def _edit_pair(key: np.ndarray, outcome: np.ndarray) -> tuple[int, int] | None:
    for cell in range(cert.N_KEY):
        rows = key == cell
        if not np.any(rows):
            continue
        if not np.any(outcome[rows] == 1):
            return cell, 1
    return None


def load_bank(rng: np.random.Generator, n_psu: int) -> dict:
    table = pd.read_csv(BANK, sep=";")
    keys, mapping = _collapse_labels(table["job"].to_numpy(), n_keep=5)
    outcome = (table["y"].astype(str).to_numpy() == "yes").astype(int)
    stratum = _rank_bins(table["age"].to_numpy(), N_STRATA)
    frame = _frame(stratum, keys, outcome, n_psu, rng)
    frame["base"] = np.ones(frame["key"].shape[0], dtype=float)
    frame.update(
        {
            "name": "bank_marketing",
            "source": str(BANK),
            "n_file": int(len(table)),
            "key_rule": "five most common jobs, remaining jobs as one key",
            "key_map": mapping,
            "outcome_rule": "term deposit subscription",
            "weight_rule": "no survey weight; base weight 1 on an imposed age-stratum design",
            "note": "UCI bank-full.csv, 45211 contacts.",
        }
    )
    return frame


def load_credit(rng: np.random.Generator, n_psu: int) -> dict:
    table = pd.read_csv(CREDIT)
    keys = _rank_bins(table["AGE"].to_numpy(), cert.N_KEY)
    outcome = table["default payment next month"].to_numpy().astype(int)
    stratum = _rank_bins(table["LIMIT_BAL"].to_numpy(), N_STRATA)
    frame = _frame(stratum, keys, outcome, n_psu, rng)
    frame["base"] = np.ones(frame["key"].shape[0], dtype=float)
    frame.update(
        {
            "name": "credit_default",
            "source": str(CREDIT),
            "n_file": int(len(table)),
            "key_rule": "six age bins of equal count",
            "outcome_rule": "default next month",
            "weight_rule": "no survey weight; base weight 1; strata are credit-limit quartiles",
            "note": "Imbalanced default indicator. Not a complex survey.",
        }
    )
    return frame


def _schl_key(schl: np.ndarray) -> np.ndarray:
    """Six schooling bins from the 2024 PUMS schooling code."""
    keys = np.full(schl.shape[0], -1, dtype=int)
    keys[(schl >= 1) & (schl <= 15)] = 0
    keys[(schl >= 16) & (schl <= 17)] = 1
    keys[(schl >= 18) & (schl <= 19)] = 2
    keys[schl == 20] = 3
    keys[schl == 21] = 4
    keys[(schl >= 22) & (schl <= 24)] = 5
    return keys


def load_acs(rng: np.random.Generator, n_psu: int) -> dict:
    table = pd.read_csv(ACS, usecols=["AGEP", "ESR", "SCHL", "PWGTP", "PUMA", "PINCP"])
    n_file = int(len(table))
    keep = (
        (table["AGEP"] >= 18)
        & table["SCHL"].between(1, 24)
        & table["ESR"].between(1, 6)
        & (table["PWGTP"] > 0)
    )
    table = table.loc[keep].copy()
    table["key"] = _schl_key(table["SCHL"].to_numpy())
    table = table.loc[table["key"] >= 0]
    table["outcome"] = table["ESR"].isin([1, 2, 4, 5]).astype(int)
    puma_income = table.groupby("PUMA")["PINCP"].median()
    puma_stratum = pd.Series(_rank_bins(puma_income.to_numpy(), N_STRATA), index=puma_income.index)
    counts = puma_stratum.value_counts()
    n_keep = int(min(n_psu, int(counts.min())))
    chosen = []
    for s in range(N_STRATA):
        members = puma_stratum.index[puma_stratum == s].to_numpy()
        pick = rng.choice(members, size=n_keep, replace=False)
        chosen.extend(int(u) for u in pick)
    full_in_puma = table.loc[table["PUMA"].isin(chosen)].copy()
    puma_n = full_in_puma.groupby("PUMA").size()
    parts = []
    for puma, group in full_in_puma.groupby("PUMA"):
        n_here = int(puma_n.loc[puma])
        if n_here > 40:
            group = group.sample(40, random_state=int(rng.integers(0, 2**31 - 1))).copy()
            group["PWGTP"] = group["PWGTP"] * (n_here / 40.0)
        parts.append(group)
    table = pd.concat(parts, ignore_index=True)
    table["stratum"] = table["PUMA"].map(puma_stratum).astype(int)
    frame = {
        "stratum": table["stratum"].to_numpy(dtype=int),
        "psu": table["PUMA"].to_numpy(dtype=int),
        "key": table["key"].to_numpy(dtype=int),
        "outcome": table["outcome"].to_numpy(dtype=int),
        "base": table["PWGTP"].to_numpy(dtype=float),
        "n_psu": n_keep,
        "name": "acs_pums_2024_ca",
        "source": str(ACS),
        "n_file": n_file,
        "key_rule": "six schooling bins from the 2024 person-level schooling code",
        "outcome_rule": "employed, including armed forces, versus not employed",
        "weight_rule": "person weight PWGTP divided by the sampling and audit probabilities",
        "note": "2024 ACS 1-year PUMS, California persons. Primary sampling units are PUMAs, stratified by PUMA median income. At most 40 persons are kept in each PUMA and the person weight is scaled by that fraction.",
    }
    return frame


def _regions(center: np.ndarray, replicates: np.ndarray) -> dict:
    full = np.arange(cert.N_COORD)
    cells = np.arange(cert.N_KEY)
    return {
        "replicate_maximum": cert.band_from_replicates(center, replicates, ALPHA, full, marginal=False),
        "pointwise": cert.marginal_band_from_replicates(center, replicates, ALPHA, "pointwise"),
        "bonferroni": cert.marginal_band_from_replicates(center, replicates, ALPHA, "bonferroni"),
        "marginal_band": cert.band_from_replicates(center, replicates, ALPHA, cells, marginal=True),
    }


def _tally(replies: dict) -> dict[str, int]:
    counts = {"certify": 0, "reject": 0, "inconclusive": 0}
    for reply in replies.values():
        if reply in counts:
            counts[reply] += 1
    return counts


def one_repetition(frame: dict, rng: np.random.Generator, gen_rng: np.random.Generator) -> dict:
    started = time.perf_counter()
    n_psu = int(frame["n_psu"])
    stratum_of = {}
    for u in np.unique(frame["psu"]):
        stratum_of[int(u)] = int(frame["stratum"][frame["psu"] == u][0])
    sampled: list[int] = []
    for s in range(N_STRATA):
        units = np.unique(frame["psu"][frame["stratum"] == s])
        sampled.extend(int(u) for u in rng.choice(units, size=N_DRAW, replace=False))
    dev: list[int] = []
    aud: list[int] = []
    for s in range(N_STRATA):
        group = np.array([u for u in sampled if stratum_of[u] == s], dtype=int)
        order = rng.permutation(group)
        n_audit = N_DRAW // 2
        dev.extend(int(u) for u in order[:-n_audit])
        aud.extend(int(u) for u in order[-n_audit:])
    dev_rows = np.isin(frame["psu"], dev)
    aud_rows = np.isin(frame["psu"], aud)
    base = frame["base"][aud_rows]
    weight = base / ((N_DRAW / n_psu) * 0.5)
    ids, ind_sums, w_sums = _psu_table(frame["key"][aud_rows], frame["outcome"][aud_rows], weight, frame["psu"][aud_rows])
    stratum_id = np.array([stratum_of[int(u)] for u in ids], dtype=int)
    center, replicates = bootstrap_replicates(ind_sums, w_sums, stratum_id, N_BOOT, rng)
    regions = _regions(center, replicates)
    dev_key = frame["key"][dev_rows]
    dev_out = frame["outcome"][dev_rows]
    pick = gen_rng.integers(0, dev_key.shape[0], size=N_RELEASE)
    raw = np.column_stack([dev_key[pick], dev_out[pick]]).astype(int)
    edit = frame["edit"]
    projections = [cert.key_projection(), cert.pair_projection()]
    screened = cert.screen_release(raw, edit, MULTIPLICITY, projections)
    methods = ("pointwise", "bonferroni", "replicate_maximum", "marginal_band")
    replies = {}
    failures = 0
    for name in methods:
        try:
            replies[name] = _tally(
                cert.clause_replies(regions[name], screened, list(range(cert.N_KEY)), TAU, ETA)
            )
        except Exception:
            failures += 1
            replies[name] = {"certify": 0, "reject": 0, "inconclusive": 0}
    shift_reject = 0
    shift_active = 0
    cell_rows = np.flatnonzero(raw[:, 0] == 0)
    if cell_rows.size > 0:
        conditional = cert.released_conditional(raw, 0)
        if conditional is not None and float(np.max(conditional)) > 0.34:
            target = conditional.copy()
            hi = int(np.argmax(target))
            target[hi] -= 0.33
            target[1 - hi] += 0.33
            shifted = cert.flip_protected_conditional(raw, 0, target, gen_rng)
            shifted = cert.screen_release(shifted, edit, MULTIPLICITY, projections)
            try:
                got = cert.clause_replies(
                    regions["replicate_maximum"], shifted, [0], TAU, ETA
                )
                if ("fidelity", 0) in got:
                    shift_active = 1
                    shift_reject = int(got[("fidelity", 0)] == "reject")
            except Exception:
                failures += 1
    leak_removed = None
    if frame["edit_pair"] is not None:
        key, outcome = frame["edit_pair"]
        leak = np.vstack([raw, np.tile([key, outcome], (4, 1))])
        cleaned = cert.screen_release(leak, edit, MULTIPLICITY, projections)
        leak_removed = int(not np.any((cleaned[:, 0] == key) & (cleaned[:, 1] == outcome))) if cleaned.size else 1
    elapsed = time.perf_counter() - started
    return {
        "retained": int(screened.shape[0]),
        "raw": int(raw.shape[0]),
        "replies": replies,
        "nonempty": {name: int(regions[name].nonempty) for name in methods},
        "shift_active": shift_active,
        "shift_reject": shift_reject,
        "leak_removed": leak_removed,
        "solver_failures": failures,
        "seconds": elapsed,
    }


def run_case(loader, n_psu: int, n_rep: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    frame = loader(rng, n_psu)
    edit_pair = _edit_pair(frame["key"], frame["outcome"])
    frame["edit_pair"] = edit_pair
    frame["edit"] = set() if edit_pair is None else {edit_pair}
    parent = np.random.SeedSequence(seed + 17)
    children = parent.spawn(n_rep)
    rows = []
    for child in children:
        rep_rng = np.random.default_rng(child)
        gen_rng = np.random.default_rng(int(rep_rng.integers(0, 2**31 - 1)))
        rows.append(one_repetition(frame, rep_rng, gen_rng))
    methods = ("pointwise", "bonferroni", "replicate_maximum", "marginal_band")
    reply_totals = {name: {"certify": 0, "reject": 0, "inconclusive": 0} for name in methods}
    for row in rows:
        for name in methods:
            for kind in reply_totals[name]:
                reply_totals[name][kind] += int(row["replies"][name][kind])
    reply_rates = {}
    for name in methods:
        total = sum(reply_totals[name].values())
        reply_rates[name] = {
            kind: (reply_totals[name][kind] / total if total else None) for kind in reply_totals[name]
        }
        reply_rates[name]["n_replies"] = total
    shift_n = sum(int(row["shift_active"]) for row in rows)
    leak_rows = [row["leak_removed"] for row in rows if row["leak_removed"] is not None]
    return {
        "paper_table": False,
        "coverage_claim": False,
        "name": frame["name"],
        "source": frame["source"],
        "n_file": frame["n_file"],
        "n_rows_used": int(frame["key"].shape[0]),
        "n_psu_per_stratum": n_psu,
        "n_rep": n_rep,
        "key_rule": frame["key_rule"],
        "key_map": frame.get("key_map"),
        "outcome_rule": frame["outcome_rule"],
        "weight_rule": frame["weight_rule"],
        "note": frame["note"],
        "edit_pair": None if edit_pair is None else [int(edit_pair[0]), int(edit_pair[1])],
        "mean_retained_size": float(np.mean([row["retained"] for row in rows])),
        "mean_seconds": float(np.mean([row["seconds"] for row in rows])),
        "nonempty_rate": {
            name: float(np.mean([row["nonempty"][name] for row in rows])) for name in methods
        },
        "reply_rates": reply_rates,
        "shift_reject_rate": (sum(int(row["shift_reject"]) for row in rows) / shift_n) if shift_n else None,
        "shift_active_reps": shift_n,
        "leak_removed_rate": float(np.mean(leak_rows)) if leak_rows else None,
        "solver_failures": int(sum(row["solver_failures"] for row in rows)),
    }


def main() -> int:
    specs = (
        (load_bank, 40, N_REP, 20261031),
        (load_credit, 40, N_REP, 20261032),
        (load_acs, 40, N_REP, 20261033),
    )
    cases = [run_case(*spec) for spec in specs]
    scale_rng = np.random.default_rng(7)
    started = time.perf_counter()
    scale = run_case(load_credit, 160, 1, 20261034)
    scale["wall_seconds"] = time.perf_counter() - started
    payload = {
        "paper_table": False,
        "coverage_claim": False,
        "n_rep": N_REP,
        "cases": cases,
        "credit_scale_one_rep": scale,
    }
    dest = ROOT / "results" / "case_studies.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for case in cases:
        cert_rates = case["reply_rates"]["replicate_maximum"]
        print(
            case["name"],
            "retained",
            round(case["mean_retained_size"], 1),
            "sec",
            round(case["mean_seconds"], 3),
            "certify",
            cert_rates["certify"],
            "reject",
            cert_rates["reject"],
            "inconclusive",
            cert_rates["inconclusive"],
            "shift_reject",
            case["shift_reject_rate"],
            flush=True,
        )
    print("scale_rows", scale["n_rows_used"], "scale_sec", round(scale["mean_seconds"], 3), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
