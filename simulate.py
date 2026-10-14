"""Finite-population experiment for the protected local-fidelity certificate.

The baseline regime is four strata, forty primary sampling units, 25 persons,
twenty-four units sampled per stratum, then a half split. Other regimes change
one design factor at a time. The ideal release draws from the finite-population
law. A second copy shifts the protected conditional by total variation 0.33.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import beta

import certificate as cert

N_STRATA = 4
N_PER = 25
MULTIPLICITY = 2
TAU = 0.05
ETA = 0.25
ALPHA = 0.10
UTILITY_FLOOR = 0.5
N_BOOT = 199
PROTECTED = 0
EDIT = {(cert.N_KEY - 1, 1)}
# Kept so a caller can name the baseline shift. The population builder derives
# the same shift from the realized conditional: move 0.33 off the larger outcome.
REVERSAL_CONDITIONAL = np.array([0.58, 0.42], dtype=float)


@dataclass(frozen=True)
class Regime:
    """One controlled design. Unspecified fields stay at the baseline."""

    name: str
    n_psu: int = 40
    n_draw: int = 24
    n_release: int = 80
    swing: int = 2
    weight_high: float = 2.0
    base_counts: tuple[int, ...] = (8, 6, 4, 3, 2, 2)
    n_protected: int = 6
    n_projections: int = 1
    conditional: str = "baseline"
    perturbations: bool = False


def regime_grid() -> list[Regime]:
    """Baseline plus one change for each factor named in the evaluation section."""
    return [
        Regime("baseline", perturbations=True),
        Regime("psu_few", n_psu=20, n_draw=12),
        Regime("psu_many", n_psu=60, n_draw=36),
        Regime("icc_low", swing=0, conditional="low_icc"),
        Regime("icc_high", swing=4),
        Regime("weight_equal", weight_high=1.0),
        Regime("weight_high", weight_high=8.0),
        Regime("prevalence_rare", base_counts=(4, 7, 6, 4, 2, 2)),
        Regime("prevalence_common", base_counts=(12, 4, 3, 2, 2, 2)),
        Regime("projections_two", n_projections=2, perturbations=True),
        Regime("protected_two", n_protected=2),
        Regime("conditional_balanced", conditional="balanced"),
        Regime("release_40", n_release=40),
        Regime("release_160", n_release=160),
    ]


def _unit_key_counts(unit: int, base: tuple[int, ...], swing: int) -> np.ndarray:
    """Twenty-five persons. A transfer across the first four keys keeps the mean."""
    counts = np.array(base, dtype=int).copy()
    if int(counts.sum()) != N_PER:
        raise RuntimeError("base key counts do not sum to the unit size")
    if swing > 0:
        donor = int(unit % 4)
        receiver = int((unit + 1) % 4)
        move = min(int(swing), int(counts[donor]) - 2)
        if move > 0:
            counts[donor] -= move
            counts[receiver] += move
    if np.any(counts < 1) or int(counts.sum()) != N_PER:
        raise RuntimeError("PSU key counts are not a partition of the unit")
    return counts


def _outcome_share(unit: int, key: int, mode: str) -> float:
    """Protected key stays at 0.75 except in the balanced regime."""
    if key == 0:
        return 0.55 if mode == "balanced" else 0.75
    if mode == "low_icc":
        return 0.40
    return 0.15 + 0.70 * (((unit + 2 * key) % 5) / 4.0)


def _reversal_target(conditional: np.ndarray) -> np.ndarray:
    """Move 0.33 probability off the larger outcome. Total variation is 0.33."""
    target = np.asarray(conditional, dtype=float).copy()
    hi = int(np.argmax(target))
    lo = 1 - hi
    target[hi] -= 0.33
    target[lo] += 0.33
    if np.any(target <= 0.0) or np.any(target >= 1.0):
        raise RuntimeError("reversal target left the probability simplex")
    return target


@dataclass(frozen=True)
class Population:
    stratum: np.ndarray
    psu: np.ndarray
    key: np.ndarray
    outcome: np.ndarray
    base: np.ndarray
    n_psu: int
    n_draw: int
    n_release: int
    n_protected: int
    n_projections: int
    perturbations: bool
    reversal: np.ndarray
    regime_name: str


def make_population(regime: Regime) -> Population:
    """Finite population for one regime. Composition does not depend on the stratum."""
    stratum_list: list[int] = []
    psu_list: list[int] = []
    key_list: list[int] = []
    outcome_list: list[int] = []
    base_list: list[float] = []
    for s in range(N_STRATA):
        stratum_base = 1.0 if s == 0 else float(regime.weight_high)
        for u in range(regime.n_psu):
            psu = s * regime.n_psu + u
            counts = _unit_key_counts(u, regime.base_counts, regime.swing)
            if int(counts[0]) < 2:
                raise RuntimeError("PSU composition is not a feasible protected cell")
            keys_list: list[int] = []
            outcomes_list: list[int] = []
            for key, count in enumerate(counts.tolist()):
                count = int(count)
                if key == cert.N_KEY - 1:
                    y1 = 0
                else:
                    share = _outcome_share(u, key, regime.conditional)
                    y1 = int(round(count * share))
                    y1 = min(max(y1, 0), count if key else count - 1)
                    if key == 0:
                        y1 = min(max(y1, 1), count - 1)
                y0 = count - y1
                keys_list.extend([key] * count)
                outcomes_list.extend([1] * y1 + [0] * y0)
            stratum_list.extend([s] * N_PER)
            psu_list.extend([psu] * N_PER)
            key_list.extend(keys_list)
            outcome_list.extend(outcomes_list)
            base_list.extend([stratum_base] * N_PER)
    pop = Population(
        stratum=np.asarray(stratum_list, dtype=int),
        psu=np.asarray(psu_list, dtype=int),
        key=np.asarray(key_list, dtype=int),
        outcome=np.asarray(outcome_list, dtype=int),
        base=np.asarray(base_list, dtype=float),
        n_psu=int(regime.n_psu),
        n_draw=int(regime.n_draw),
        n_release=int(regime.n_release),
        n_protected=int(regime.n_protected),
        n_projections=int(regime.n_projections),
        perturbations=bool(regime.perturbations),
        reversal=np.zeros(2, dtype=float),
        regime_name=regime.name,
    )
    reversal = _reversal_for(pop)
    pop = Population(
        stratum=pop.stratum,
        psu=pop.psu,
        key=pop.key,
        outcome=pop.outcome,
        base=pop.base,
        n_psu=pop.n_psu,
        n_draw=pop.n_draw,
        n_release=pop.n_release,
        n_protected=pop.n_protected,
        n_projections=pop.n_projections,
        perturbations=pop.perturbations,
        reversal=reversal,
        regime_name=pop.regime_name,
    )
    _assert_population(pop)
    return pop


def make_ablation_population() -> Population:
    """Baseline regime used by the smoke test."""
    return make_population(Regime("baseline", perturbations=True))


def _reversal_for(pop: Population) -> np.ndarray:
    theta = cert.population_vector(pop.key, pop.outcome)
    mass = float(theta[PROTECTED])
    joints = theta[cert.N_KEY + PROTECTED * cert.N_OUT : cert.N_KEY + (PROTECTED + 1) * cert.N_OUT]
    return _reversal_target(joints / mass)


def _assert_population(pop: Population) -> None:
    if np.any((pop.key == cert.N_KEY - 1) & (pop.outcome == 1)):
        raise RuntimeError("structural-zero pair is populated")
    theta = cert.population_vector(pop.key, pop.outcome)
    weighted = cert.population_vector(pop.key, pop.outcome, pop.base)
    if np.max(np.abs(theta - weighted)) > 1e-12:
        raise RuntimeError("stratum base weights moved the finite-population vector")
    if theta[PROTECTED] < TAU:
        raise RuntimeError("protected cell is below the support threshold")
    mass = float(theta[PROTECTED])
    joints = theta[cert.N_KEY + PROTECTED * cert.N_OUT : cert.N_KEY + (PROTECTED + 1) * cert.N_OUT]
    conditional = joints / mass
    shift = cert.total_variation(conditional, pop.reversal)
    if shift <= ETA or shift > ETA + 0.15:
        raise RuntimeError("reversal shift is not a moderate violation of the tolerance")


def sample_psus(pop: Population, rng: np.random.Generator) -> np.ndarray:
    """Stratified sample of primary sampling units. Every person in a drawn unit enters."""
    chosen: list[int] = []
    for s in range(N_STRATA):
        units = np.unique(pop.psu[pop.stratum == s])
        pick = rng.choice(units, size=pop.n_draw, replace=False)
        chosen.extend(int(u) for u in pick)
    return np.asarray(chosen, dtype=int)


def split_psus(
    units: np.ndarray,
    stratum_of: dict[int, int],
    rng: np.random.Generator,
    n_draw: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Randomize whole sampled units to development or audit inside each stratum.

    Half of the sampled units go to the audit. The audit-assignment
    probability is one half, and both sides stay nonempty.
    """
    dev: list[int] = []
    aud: list[int] = []
    n_audit = n_draw // 2
    for s in range(N_STRATA):
        group = np.array([int(u) for u in units if stratum_of[int(u)] == s], dtype=int)
        order = rng.permutation(group)
        n_dev = len(order) - n_audit
        if n_dev <= 0 or n_audit <= 0:
            raise RuntimeError("split would empty one side")
        dev.extend(int(u) for u in order[:n_dev])
        aud.extend(int(u) for u in order[n_dev:])
    return np.asarray(dev, dtype=int), np.asarray(aud, dtype=int)


def draw_ideal(pop: Population, n_release: int, rng: np.random.Generator) -> np.ndarray:
    """Independent draws from the finite-population law on feasible pairs.

    The structural-zero pair has population count 0, so these draws miss the
    edit set. The generator does not read the audit sample.
    """
    pick = rng.integers(0, pop.key.shape[0], size=n_release)
    return np.column_stack([pop.key[pick], pop.outcome[pick]]).astype(int)


def _psu_table(
    key: np.ndarray,
    outcome: np.ndarray,
    weight: np.ndarray,
    psu: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Weighted indicator totals for each audit primary sampling unit, grouped by stratum."""
    ind = cert.cell_joint_indicators(key, outcome)
    order = np.argsort(psu, kind="mergesort")
    psu_s = psu[order]
    ind_s = ind[order]
    w_s = weight[order]
    breaks = np.flatnonzero(np.diff(psu_s)) + 1
    groups = np.split(np.arange(psu_s.shape[0]), breaks)
    ids = []
    ind_sums = []
    w_sums = []
    for idx in groups:
        ids.append(int(psu_s[idx[0]]))
        w_sums.append(float(w_s[idx].sum()))
        ind_sums.append((w_s[idx, None] * ind_s[idx]).sum(axis=0))
    return (
        np.asarray(ids, dtype=int),
        np.vstack(ind_sums),
        np.asarray(w_sums, dtype=float),
    )


def bootstrap_replicates(
    ind_sums: np.ndarray,
    w_sums: np.ndarray,
    stratum_id: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample audit primary sampling units with replacement inside each stratum.

    The point estimate stays the Hajek total. Each replicate reweights those
    units by 1 - a_h + a_h m_i, with a_h = sqrt(n_h / (n_h - 1)) and m_i the
    number of times the unit was drawn. That factor matches the with-replacement
    resample to the primary-sampling-unit variance. The manuscript standard
    error and quantile are applied to these replicates. There is no parametric
    draw and no extra scale chosen to lift coverage.
    """
    n_units, n_coord = ind_sums.shape
    center = ind_sums.sum(axis=0) / float(w_sums.sum())
    means = ind_sums / np.maximum(w_sums, 1e-15)[:, None]
    replicates = np.zeros((n_boot, n_coord), dtype=float)
    groups = [np.flatnonzero(stratum_id == s) for s in np.unique(stratum_id)]
    for b in range(n_boot):
        w_star = np.zeros(n_units, dtype=float)
        for cols in groups:
            n_h = int(cols.size)
            if n_h < 2:
                w_star[cols] = w_sums[cols]
                continue
            draw = rng.choice(n_h, size=n_h, replace=True)
            multiplicity = np.bincount(draw, minlength=n_h).astype(float)
            scale = float(np.sqrt(n_h / (n_h - 1.0)))
            w_star[cols] = w_sums[cols] * (1.0 - scale + scale * multiplicity)
        total = float(w_star.sum())
        if total == 0.0:
            replicates[b] = center
            continue
        replicates[b] = (w_star @ means) / total
    return center, replicates


def _force_singleton(release: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, tuple[int, int]]:
    """Leave one record on a single non-edit key-outcome cell."""
    out = np.array(release, dtype=int, copy=True)
    candidates: list[tuple[int, int, np.ndarray]] = []
    for key in range(cert.N_KEY - 1):
        rows = np.flatnonzero(out[:, 0] == key)
        if rows.size < 4:
            continue
        for y in (0, 1):
            yrows = rows[out[rows, 1] == y]
            if yrows.size >= 2:
                candidates.append((key, y, yrows))
    if not candidates:
        return out, (-1, -1)
    pick = candidates[int(rng.integers(0, len(candidates)))]
    keep = np.ones(out.shape[0], dtype=bool)
    keep[pick[2][1:]] = False
    return out[keep], (pick[0], pick[1])


def _add_leak(release: np.ndarray, n: int = 4) -> np.ndarray:
    """Append records on the declared structural-zero pair."""
    leak = np.column_stack(
        [np.full(n, cert.N_KEY - 1, dtype=int), np.ones(n, dtype=int)]
    )
    if release.size == 0:
        return leak
    return np.vstack([release, leak])


def _has_pair(release: np.ndarray, key: int, outcome: int) -> bool:
    if release.size == 0 or key < 0:
        return False
    return bool(np.any((release[:, 0] == key) & (release[:, 1] == outcome)))


def _incorrect(reply: str, inside: bool) -> bool:
    if reply == "certify" and not inside:
        return True
    if reply == "reject" and inside:
        return True
    return False


def run_repetition(
    pop: Population,
    rng: np.random.Generator,
    gen_rng: np.random.Generator,
    alpha: float = ALPHA,
    n_boot: int = N_BOOT,
) -> dict:
    """One outer repetition: sample, split, two releases, four regions."""
    stratum_of = {int(u): int(pop.stratum[pop.psu == u][0]) for u in np.unique(pop.psu)}
    sampled = sample_psus(pop, rng)
    dev_psu, aud_psu = split_psus(sampled, stratum_of, rng, pop.n_draw)
    dev = np.isin(pop.psu, dev_psu)
    aud = np.isin(pop.psu, aud_psu)
    if not np.any(dev) or not np.any(aud):
        raise RuntimeError("split left a side empty")
    # The ideal generator uses the finite population, not the audit rows.
    _ = dev

    pi_sample = pop.n_draw / pop.n_psu
    pi_audit = 0.5
    weight = cert.audit_weight(pop.base[aud], pi_sample, pi_audit)
    ids, ind_sums, w_sums = _psu_table(pop.key[aud], pop.outcome[aud], weight, pop.psu[aud])
    stratum_id = np.array([stratum_of[int(u)] for u in ids], dtype=int)
    center, replicates = bootstrap_replicates(ind_sums, w_sums, stratum_id, n_boot, rng)

    full_columns = np.arange(cert.N_COORD)
    cell_columns = np.arange(cert.N_KEY)
    regions = {
        "replicate_maximum": cert.band_from_replicates(center, replicates, alpha, full_columns, marginal=False),
        "pointwise": cert.marginal_band_from_replicates(center, replicates, alpha, "pointwise"),
        "bonferroni": cert.marginal_band_from_replicates(center, replicates, alpha, "bonferroni"),
        "marginal_band": cert.band_from_replicates(center, replicates, alpha, cell_columns, marginal=True),
    }
    theta = cert.population_vector(pop.key, pop.outcome)
    covered = cert.in_box(theta, regions["replicate_maximum"])

    raw = draw_ideal(pop, pop.n_release, gen_rng)
    projections = [cert.key_projection()]
    if pop.n_projections >= 2:
        projections.append(cert.pair_projection())
    ideal = cert.screen_release(raw, EDIT, MULTIPLICITY, projections)
    raw_rev = cert.flip_protected_conditional(raw, PROTECTED, pop.reversal, gen_rng)
    reversal = cert.screen_release(raw_rev, EDIT, MULTIPLICITY, projections)

    # Even cells are registered at the true proportion, so the clause is true
    # and only a low miss rejects it. Odd cells are registered just above the
    # truth, so the clause is false and only a high miss certifies it.
    # Thresholds are fixed from the finite population before the audit opens.
    protected_cells = list(range(pop.n_protected))
    taus = {}
    for cell in protected_cells:
        mass = float(theta[cell])
        taus[cell] = mass if cell % 2 == 0 else mass + 1e-4
    methods = ("pointwise", "bonferroni", "replicate_maximum", "marginal_band")
    session = {name: 0 for name in methods}
    solver_failures = 0
    proposed_active = 0
    proposed_inconclusive = 0
    power_reject = {name: 0 for name in methods}
    nonempty = {name: int(regions[name].nonempty) for name in methods}
    leak_removed = None
    singleton_removed = None
    if pop.perturbations:
        raw_unique, singleton = _force_singleton(raw, gen_rng)
        unique = cert.screen_release(raw_unique, EDIT, MULTIPLICITY, projections)
        raw_leak = _add_leak(raw)
        leaked = cert.screen_release(raw_leak, EDIT, MULTIPLICITY, projections)
        raw_joint, _singleton_joint = _force_singleton(raw_rev, gen_rng)
        raw_joint = _add_leak(raw_joint)
        joint = cert.screen_release(raw_joint, EDIT, MULTIPLICITY, projections)
        leak_removed = int(not _has_pair(leaked, cert.N_KEY - 1, 1))
        singleton_removed = int(not _has_pair(unique, singleton[0], singleton[1]))
        arms = (
            ("ideal", ideal, raw.shape[0]),
            ("reversal", reversal, raw_rev.shape[0]),
            ("uniqueness", unique, raw_unique.shape[0]),
            ("leakage", leaked, raw_leak.shape[0]),
            ("joint", joint, raw_joint.shape[0]),
        )
    else:
        arms = (("ideal", ideal, raw.shape[0]), ("reversal", reversal, raw_rev.shape[0]))
    for arm_name, screened, n_raw in arms:
        kept_ok = cert.utility_ok(n_raw, screened.shape[0], UTILITY_FLOOR)
        if not kept_ok or screened.shape[0] == 0:
            continue
        for name in methods:
            order = sorted(protected_cells, key=lambda cell: float(regions[name].half[cell]))
            try:
                replies = cert.clause_replies(regions[name], screened, order, taus, ETA)
            except Exception:
                solver_failures += 1
                continue
            for cell in order:
                if ("support", cell) not in replies:
                    continue
                conditional = cert.released_conditional(screened, cell)
                if conditional is None:
                    continue
                pair = (
                    (replies[("support", cell)], cert.truth_in_support(theta, cell, taus[cell])),
                    (replies[("fidelity", cell)], cert.truth_in_fidelity(theta, cell, conditional, ETA)),
                )
                if any(_incorrect(reply, inside) for reply, inside in pair):
                    session[name] = 1
                if name == "replicate_maximum":
                    for reply, _inside in pair:
                        proposed_active += 1
                        if reply == "inconclusive":
                            proposed_inconclusive += 1
                        if reply not in ("certify", "reject", "inconclusive"):
                            solver_failures += 1
                    if (
                        arm_name == "reversal"
                        and cell == PROTECTED
                        and replies.get(("fidelity", PROTECTED)) == "reject"
                    ):
                        power_reject[name] = 1
    zero_se = np.where(regions["replicate_maximum"].se == 0.0)[0]
    zero_half = (
        bool(np.all(regions["replicate_maximum"].half[zero_se] == 0.0)) if zero_se.size else True
    )
    return {
        "n_dev": int(dev.sum()),
        "n_aud": int(aud.sum()),
        "n_dev_psu": int(dev_psu.size),
        "n_aud_psu": int(aud_psu.size),
        "covered": int(covered),
        "session": session,
        "power_reject": {name: int(power_reject[name]) for name in methods},
        "nonempty": nonempty,
        "cell_half": {
            name: float(regions[name].half[PROTECTED]) for name in ("pointwise", "bonferroni", "replicate_maximum")
        },
        "proposed_active": int(proposed_active),
        "proposed_inconclusive": int(proposed_inconclusive),
        "n_raw": int(raw.shape[0]),
        "n_kept": int(ideal.shape[0]),
        "screen_grew": bool(ideal.shape[0] > raw.shape[0]),
        "solver_failures": int(solver_failures),
        "zero_se_point_band": zero_half,
        "utility_ok": bool(cert.utility_ok(raw.shape[0], ideal.shape[0], UTILITY_FLOOR)),
        "leak_removed": None if leak_removed is None else int(leak_removed),
        "singleton_removed": None if singleton_removed is None else int(singleton_removed),
    }


def clopper_pearson(hits: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Clopper-Pearson interval. Empty and zero counts stay in the bounds."""
    if n <= 0:
        return float("nan"), float("nan")
    alpha = 1.0 - level
    lower = 0.0 if hits <= 0 else float(beta.ppf(alpha / 2.0, hits, n - hits + 1))
    upper = 1.0 if hits >= n else float(beta.ppf(1.0 - alpha / 2.0, hits + 1, n - hits))
    return lower, upper


def _ranks(scores: dict[str, float]) -> dict[str, float]:
    names = list(scores)
    values = np.array([scores[name] for name in names], dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(names), dtype=float)
    start = 0
    while start < len(names):
        stop = start + 1
        while stop < len(names) and values[order[stop]] == values[order[start]]:
            stop += 1
        # Average rank, 1-based, lower session error is better.
        avg = 0.5 * ((start + 1) + stop)
        for pos in range(start, stop):
            ranks[order[pos]] = avg
        start = stop
    return {name: float(ranks[i]) for i, name in enumerate(names)}


def run_study(
    n_rep: int = 200,
    alpha: float = ALPHA,
    seed: int = 20261015,
    n_boot: int = N_BOOT,
    regime: Regime | None = None,
) -> dict:
    """Outer repetitions. The generator seed is nested inside the repetition seed."""
    if regime is None:
        regime = Regime("baseline", perturbations=True)
    pop = make_population(regime)
    parent = np.random.SeedSequence(seed)
    children = parent.spawn(n_rep)
    methods = ("pointwise", "bonferroni", "replicate_maximum", "marginal_band")
    session_hits = {name: 0 for name in methods}
    power_hits = {name: 0 for name in methods}
    nonempty_hits = {name: 0 for name in methods}
    half_sums = {name: 0.0 for name in ("pointwise", "bonferroni", "replicate_maximum")}
    covered = 0
    active = 0
    inconclusive = 0
    kept = 0.0
    solver_failures = 0
    empty_split = 0
    leak_hits = 0
    singleton_hits = 0
    screen_n = 0
    for r in range(n_rep):
        rng = np.random.default_rng(children[r])
        gen_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        out = run_repetition(pop, rng, gen_rng, alpha=alpha, n_boot=n_boot)
        covered += int(out["covered"])
        active += int(out["proposed_active"])
        for name in methods:
            power_hits[name] += int(out["power_reject"][name])
            nonempty_hits[name] += int(out["nonempty"][name])
        for name in half_sums:
            half_sums[name] += float(out["cell_half"][name])
        inconclusive += int(out["proposed_inconclusive"])
        kept += float(out["n_kept"])
        solver_failures += int(out["solver_failures"])
        if out["leak_removed"] is not None:
            screen_n += 1
            leak_hits += int(out["leak_removed"])
            singleton_hits += int(out["singleton_removed"])
        if out["n_dev"] == 0 or out["n_aud"] == 0:
            empty_split += 1
        for name in methods:
            session_hits[name] += int(out["session"][name])
        if (r + 1) % 20 == 0:
            print(f"repetition {r + 1} / {n_rep}", flush=True)
    coverage_rate = covered / n_rep
    session_rate = session_hits["replicate_maximum"] / n_rep
    cov_lo, cov_hi = clopper_pearson(covered, n_rep)
    ses_lo, ses_hi = clopper_pearson(session_hits["replicate_maximum"], n_rep)
    ranked_names = ("pointwise", "bonferroni", "replicate_maximum")
    ranked_scores = {name: session_hits[name] / n_rep for name in ranked_names}
    ranks = _ranks(ranked_scores)
    coverage_fail = cov_hi < 0.90
    session_fail = ses_lo > 0.10
    rank_pass = ranks["replicate_maximum"] <= 2.5
    inconclusive_fraction = (inconclusive / active) if active else 0.0
    return {
        "paper_table": False,
        "n_rep": int(n_rep),
        "alpha": float(alpha),
        "seed": int(seed),
        "n_boot": int(n_boot),
        "replicate_scheme": "within-stratum resample of audit primary sampling units, Rao-Wu factor sqrt(n_h/(n_h-1))",
        "lambda": "(B-1)/B",
        "coverage_rate": coverage_rate,
        "coverage_counts": {"hits": int(covered), "n": int(n_rep)},
        "coverage_clopper_pearson_95": [cov_lo, cov_hi],
        "session_error_rate": session_rate,
        "session_error_counts": {"hits": int(session_hits["replicate_maximum"]), "n": int(n_rep)},
        "session_error_clopper_pearson_95": [ses_lo, ses_hi],
        "power_reversal": {name: power_hits[name] / n_rep for name in ranked_names},
        "nonempty_rate": {name: nonempty_hits[name] / n_rep for name in methods},
        "mean_protected_cell_half": {name: half_sums[name] / n_rep for name in half_sums},
        "inconclusive_fraction": inconclusive_fraction,
        "mean_retained_size": kept / n_rep,
        "per_method_session_error": ranked_scores,
        "marginal_band_session_error": session_hits["marginal_band"] / n_rep,
        "ranks": ranks,
        "validity": {
            "coverage_falsifier": bool(coverage_fail),
            "session_falsifier": bool(session_fail),
            "coverage_pass": bool(not coverage_fail),
            "session_pass": bool(not session_fail),
        },
        "rank_pass": bool(rank_pass),
        "pass": bool((not coverage_fail) and (not session_fail) and rank_pass),
        "solver_failures": int(solver_failures),
        "empty_split": int(empty_split),
        "leak_removed_rate": None if screen_n == 0 else leak_hits / screen_n,
        "singleton_removed_rate": None if screen_n == 0 else singleton_hits / screen_n,
        "regime_name": regime.name,
        "regime": {
            "name": regime.name,
            "n_strata": N_STRATA,
            "n_psu": regime.n_psu,
            "n_per": N_PER,
            "n_draw": regime.n_draw,
            "n_audit": regime.n_draw // 2,
            "n_release": regime.n_release,
            "swing": regime.swing,
            "weight_high": regime.weight_high,
            "base_counts": list(regime.base_counts),
            "n_protected": regime.n_protected,
            "n_projections": regime.n_projections,
            "conditional": regime.conditional,
            "perturbations": regime.perturbations,
            "multiplicity": MULTIPLICITY,
            "tau": TAU,
            "support_thresholds": "even cells at the finite-population share; odd cells 0.0001 above that share",
            "eta": ETA,
            "utility_floor": UTILITY_FLOOR,
            "protected_cell": PROTECTED,
            "edit_pair": [cert.N_KEY - 1, 1],
            "reversal_conditional": [float(x) for x in pop.reversal],
            "projection": "key" if regime.n_projections == 1 else "key+pair",
        },
    }
