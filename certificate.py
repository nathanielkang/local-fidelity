"""Protected local-fidelity certificate.

The release rule matches the manuscript. Primary sampling units of the sample
are split before fitting. Support and multiplicity are a fixed-release screen.
One simultaneous region is built from audit weights, and each active population
clause gets a certify, reject, or inconclusive reply. The true finite-population
vector is not an input to the critical value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

# Numerical tolerance for linear programs. This is not a floor on a standard error.
LP_TOL = 1e-8
N_KEY = 6
N_OUT = 2
N_COORD = N_KEY + N_KEY * N_OUT  # cells, then joints


@dataclass(frozen=True)
class Projection:
    """A registered key projection. `columns` index the release record."""

    name: str
    columns: tuple[int, ...]


@dataclass(frozen=True)
class Region:
    """Coordinate box intersected with the simplex and the projection identities."""

    center: np.ndarray
    half: np.ndarray
    se: np.ndarray
    critical: float
    nonempty: bool
    marginal: bool


def key_projection() -> Projection:
    return Projection("key", (0,))


def outcome_projection() -> Projection:
    return Projection("outcome", (1,))


def pair_projection() -> Projection:
    """Key and outcome together. A count of one on this projection is a singleton cell."""
    return Projection("pair", (0, 1))


def cell_joint_indicators(key: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    """Columns: P(K=0), P(K=1), then P(K=a, Y=y) in row-major order."""
    key = np.asarray(key, dtype=int)
    outcome = np.asarray(outcome, dtype=int)
    ind = np.zeros((key.shape[0], N_COORD), dtype=float)
    if key.size == 0:
        return ind
    rows = np.arange(key.shape[0])
    ind[rows, key] = 1.0
    ind[rows, N_KEY + key * N_OUT + outcome] = 1.0
    return ind


def population_vector(key: np.ndarray, outcome: np.ndarray, weight: np.ndarray | None = None) -> np.ndarray:
    """Hajek proportions for the registered cell and joint coordinates."""
    key = np.asarray(key)
    outcome = np.asarray(outcome)
    if weight is None:
        weight = np.ones(key.shape[0], dtype=float)
    else:
        weight = np.asarray(weight, dtype=float)
    total = float(np.sum(weight))
    if total <= 0.0:
        raise ValueError("weights sum to zero")
    ind = cell_joint_indicators(key, outcome)
    return (weight[:, None] * ind).sum(axis=0) / total


def audit_weight(base: np.ndarray, pi_sample: float, pi_audit: float) -> np.ndarray:
    """Inverse of (sampling inclusion probability times audit-assignment probability), scaled by the stratum base.

    The cluster factor is 1/pi_sample. The stratum base is 1 or 2 as registered
    for the ablation. When every stratum has the same cell totals, the base
    does not move the finite-population proportion.
    """
    if pi_sample <= 0.0 or pi_audit <= 0.0:
        raise ValueError("inclusion probabilities must be positive")
    return np.asarray(base, dtype=float) / (pi_sample * pi_audit)


def screen_release(
    release: np.ndarray,
    edit_pairs: set[tuple[int, int]],
    multiplicity: int,
    projections: list[Projection],
) -> np.ndarray:
    """Drop the declared edit set, then delete low-count cells of every registered projection.

    A cell whose count lies in {1, ..., q-1} loses all of its records, and every
    projection is recomputed. The screen does not read an audit sample and it
    never adds a record.
    """
    release = np.asarray(release, dtype=int)
    if release.size == 0:
        return np.zeros((0, 2), dtype=int)
    if release.ndim != 2 or release.shape[1] < 2:
        raise ValueError("release must be an n by 2 array of key, outcome")
    if multiplicity < 2:
        raise ValueError("multiplicity floor q must be at least 2")
    keep = np.array([(int(row[0]), int(row[1])) not in edit_pairs for row in release], dtype=bool)
    release = release[keep]
    changed = True
    while changed and release.shape[0] > 0:
        changed = False
        drop = np.zeros(release.shape[0], dtype=bool)
        for proj in projections:
            codes = _projection_codes(release, proj)
            values, counts = np.unique(codes, return_counts=True)
            bad = set(int(v) for v, c in zip(values, counts) if 1 <= int(c) <= multiplicity - 1)
            if bad:
                drop |= np.isin(codes, list(bad))
                changed = True
        if changed:
            release = release[~drop]
    return release


def _projection_codes(release: np.ndarray, proj: Projection) -> np.ndarray:
    codes = np.zeros(release.shape[0], dtype=int)
    for col in proj.columns:
        codes = codes * 2 + release[:, col].astype(int)
    return codes


def released_cell_count(release: np.ndarray, cell: int) -> int:
    if release.size == 0:
        return 0
    return int(np.sum(release[:, 0] == cell))


def released_conditional(release: np.ndarray, cell: int) -> np.ndarray | None:
    """Empirical conditional on a protected cell. None when the cell is empty."""
    if release.size == 0:
        return None
    rows = release[release[:, 0] == cell]
    if rows.shape[0] == 0:
        return None
    probs = np.zeros(N_OUT, dtype=float)
    for y in range(N_OUT):
        probs[y] = np.sum(rows[:, 1] == y) / rows.shape[0]
    return probs


def utility_ok(n_raw: int, n_kept: int, floor: float = 0.5) -> bool:
    """Retained fraction of the fixed release. Does not read the audit sample."""
    if n_raw <= 0:
        return False
    return (n_kept / n_raw) >= floor


def replicate_se(deviations: np.ndarray) -> np.ndarray:
    """sqrt(sum_b lambda_b (theta^b - theta)^2) with lambda_b = (B-1)/B.

    A coordinate with every replicate equal to the point estimate has standard
    error 0. There is no positive floor.
    """
    deviations = np.asarray(deviations, dtype=float)
    n_rep = deviations.shape[0]
    if n_rep < 2:
        return np.zeros(deviations.shape[1], dtype=float)
    lam = (n_rep - 1.0) / n_rep
    return np.sqrt(lam * np.sum(deviations * deviations, axis=0))


def weighted_quantile(values: np.ndarray, weights: np.ndarray, prob: float) -> float:
    """Smallest value whose weighted empirical distribution reaches `prob`."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.size == 0:
        return 0.0
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    cumulative = np.cumsum(weights[order])
    total = float(cumulative[-1])
    if total <= 0.0:
        return float(ordered[-1])
    cumulative /= total
    index = int(np.searchsorted(cumulative, prob, side="left"))
    index = min(index, ordered.shape[0] - 1)
    return float(ordered[index])


def studentized_max(deviations: np.ndarray, se: np.ndarray) -> np.ndarray:
    """M_b = max_j |theta_j^b - theta_j| / se_j. A zero standard error contributes 0."""
    deviations = np.asarray(deviations, dtype=float)
    se = np.asarray(se, dtype=float)
    ratio = np.zeros_like(deviations)
    active = se > 0.0
    if np.any(active):
        ratio[:, active] = np.abs(deviations[:, active]) / se[active]
    if ratio.shape[1] == 0:
        return np.zeros(deviations.shape[0], dtype=float)
    return ratio.max(axis=1)


def critical_value(deviations: np.ndarray, se: np.ndarray, alpha: float) -> float:
    """Empirical (1-alpha) quantile of M_b with weights lambda_b / sum lambda.

    Lambda is constant, so the weights are uniform. The true population vector
    is not used.
    """
    n_rep = deviations.shape[0]
    if n_rep == 0:
        return 0.0
    lam = np.full(n_rep, (n_rep - 1.0) / n_rep if n_rep > 1 else 1.0)
    return weighted_quantile(studentized_max(deviations, se), lam, 1.0 - alpha)


def half_widths(se: np.ndarray, critical: float) -> np.ndarray:
    """Coordinate half-widths. A zero standard error is a point band, not a floor."""
    se = np.asarray(se, dtype=float)
    half = critical * se
    half = np.where(se == 0.0, 0.0, half)
    return half


def band_from_replicates(
    center: np.ndarray,
    replicates: np.ndarray,
    alpha: float,
    columns: np.ndarray | None = None,
    marginal: bool = False,
) -> Region:
    """Build one coordinate box. `columns` selects the coordinates that enter M_b."""
    center = np.asarray(center, dtype=float)
    replicates = np.asarray(replicates, dtype=float)
    if columns is None:
        columns = np.arange(center.shape[0])
    deviations = replicates[:, columns] - center[columns]
    se_sub = replicate_se(deviations)
    crit = critical_value(deviations, se_sub, alpha)
    half = np.zeros(center.shape[0], dtype=float)
    se = np.zeros(center.shape[0], dtype=float)
    se[columns] = se_sub
    half[columns] = half_widths(se_sub, crit)
    nonempty = _box_feasible(center, half, marginal)
    return Region(center=center, half=half, se=se, critical=float(crit), nonempty=nonempty, marginal=marginal)


def marginal_band_from_replicates(
    center: np.ndarray,
    replicates: np.ndarray,
    alpha: float,
    mode: str,
) -> Region:
    """Pointwise or Bonferroni box from the same replicates as the maximum statistic.

    Pointwise uses the (1-alpha) quantile of each absolute studentized replicate.
    Bonferroni uses the (1-alpha/d) quantile. Neither draws a parametric critical value.
    """
    center = np.asarray(center, dtype=float)
    replicates = np.asarray(replicates, dtype=float)
    deviations = replicates - center
    se = replicate_se(deviations)
    n_rep = deviations.shape[0]
    lam = np.full(n_rep, (n_rep - 1.0) / n_rep if n_rep > 1 else 1.0)
    d = max(int(np.sum(se > 0.0)), 1)
    if mode == "pointwise":
        prob = 1.0 - alpha
    elif mode == "bonferroni":
        prob = 1.0 - alpha / d
    else:
        raise ValueError("mode must be pointwise or bonferroni")
    half = np.zeros(center.shape[0], dtype=float)
    crits = []
    for j in range(center.shape[0]):
        if se[j] == 0.0:
            continue
        ratio = np.abs(deviations[:, j]) / se[j]
        crit_j = weighted_quantile(ratio, lam, prob)
        half[j] = crit_j * se[j]
        crits.append(crit_j)
    nonempty = _box_feasible(center, half, marginal=False)
    crit = float(np.max(crits)) if crits else 0.0
    return Region(center=center, half=half, se=se, critical=crit, nonempty=nonempty, marginal=False)


def in_box(truth: np.ndarray, region: Region, tol: float = LP_TOL) -> bool:
    """True when the finite-population vector lies in the coordinate box."""
    if not region.nonempty:
        return False
    truth = np.asarray(truth, dtype=float)
    if region.marginal:
        cols = np.arange(N_KEY)
        return bool(np.all(np.abs(truth[cols] - region.center[cols]) <= region.half[cols] + tol))
    return bool(np.all(np.abs(truth - region.center) <= region.half + tol))


def _joint_bounds() -> list[tuple[float, float | None]]:
    return [(0.0, None)] * (N_KEY * N_OUT)


def _n_joint() -> int:
    return N_KEY * N_OUT


def _constraint_matrix(region: Region) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Inequalities for the joint simplex, cell identities, and the coordinate box.

    Variables are the joint probabilities. Each cell margin is the sum of its joints.
    """
    center = region.center
    half = region.half
    n_joint = _n_joint()
    rows: list[np.ndarray] = []
    rhs: list[float] = []

    def add(coeff: np.ndarray, limit: float) -> None:
        rows.append(coeff)
        rhs.append(limit)

    eye = np.eye(n_joint)
    if not region.marginal:
        for k in range(n_joint):
            lo = center[N_KEY + k] - half[N_KEY + k]
            hi = center[N_KEY + k] + half[N_KEY + k]
            add(eye[k], hi)
            add(-eye[k], -lo)
    for a in range(N_KEY):
        coeff = _cell_coeff(a)
        lo = center[a] - half[a]
        hi = center[a] + half[a]
        add(coeff, hi)
        add(-coeff, -lo)
    a_ub = np.vstack(rows) if rows else np.zeros((0, n_joint))
    b_ub = np.asarray(rhs, dtype=float)
    a_eq = np.ones((1, n_joint))
    b_eq = np.ones(1)
    return a_ub, b_ub, a_eq, b_eq


def _solve(
    objective: np.ndarray,
    a_ub: np.ndarray,
    b_ub: np.ndarray,
    a_eq: np.ndarray,
    b_eq: np.ndarray,
) -> tuple[bool, float]:
    result = linprog(
        np.asarray(objective, dtype=float),
        A_ub=a_ub if a_ub.size else None,
        b_ub=b_ub if b_ub.size else None,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=_joint_bounds(),
        method="highs",
    )
    if not result.success:
        return False, float("nan")
    return True, float(result.fun)


def _box_feasible(center: np.ndarray, half: np.ndarray, marginal: bool) -> bool:
    probe = Region(center=center, half=half, se=np.zeros_like(center), critical=0.0, nonempty=True, marginal=marginal)
    a_ub, b_ub, a_eq, b_eq = _constraint_matrix(probe)
    ok, _ = _solve(np.zeros(_n_joint()), a_ub, b_ub, a_eq, b_eq)
    return bool(ok)


def _cell_coeff(cell: int) -> np.ndarray:
    if cell < 0 or cell >= N_KEY:
        raise ValueError("cell is outside the registered key")
    coeff = np.zeros(_n_joint(), dtype=float)
    coeff[cell * N_OUT : (cell + 1) * N_OUT] = 1.0
    return coeff


def _gap_coeff(cell: int, conditional: np.ndarray, eta: float, sign: np.ndarray) -> np.ndarray:
    """Gradient of (1/2) sum_y sign_y (p_y v_cell - v_cell,y) - eta v_cell."""
    conditional = np.asarray(conditional, dtype=float)
    sign = np.asarray(sign, dtype=float)
    coeff = np.zeros(_n_joint(), dtype=float)
    base = cell * N_OUT
    shared = 0.5 * float(np.dot(sign, conditional)) - eta
    coeff[base : base + N_OUT] += shared
    for y in range(N_OUT):
        coeff[base + y] -= 0.5 * sign[y]
    return coeff


def support_reply(region: Region, cell: int, tau: float) -> str:
    """Three-way reply for the population-support clause v_cell >= tau."""
    if not region.nonempty:
        return "inconclusive"
    a_ub, b_ub, a_eq, b_eq = _constraint_matrix(region)
    coeff = _cell_coeff(cell)
    ok_min, min_mass = _solve(coeff, a_ub, b_ub, a_eq, b_eq)
    ok_max, neg_max = _solve(-coeff, a_ub, b_ub, a_eq, b_eq)
    if not ok_min or not ok_max:
        return "inconclusive"
    max_mass = -neg_max
    if min_mass >= tau - LP_TOL:
        return "certify"
    if max_mass < tau - LP_TOL:
        return "reject"
    return "inconclusive"


def fidelity_reply(region: Region, cell: int, conditional: np.ndarray, eta: float) -> str:
    """Three-way reply for the total-variation clause.

    Certify only when the minimum cell mass on the region is strictly positive
    and every vector meets the tolerance. The zero-mass face clears the
    homogeneous inequality and is not a certification. Reject when the region
    is nonempty and no vector in it has positive cell mass and total variation
    at most eta. A region that meets the clause and its complement is inconclusive.
    """
    if not region.nonempty:
        return "inconclusive"
    conditional = np.asarray(conditional, dtype=float)
    if conditional.shape != (N_OUT,) or abs(float(conditional.sum()) - 1.0) > 1e-6:
        return "inconclusive"
    a_ub, b_ub, a_eq, b_eq = _constraint_matrix(region)
    cell_c = _cell_coeff(cell)
    ok_min, min_mass = _solve(cell_c, a_ub, b_ub, a_eq, b_eq)
    if not ok_min:
        return "inconclusive"
    signs = [np.array([s0, s1], dtype=float) for s0 in (-1.0, 1.0) for s1 in (-1.0, 1.0)]
    gaps_ok = True
    for sign in signs:
        grad = _gap_coeff(cell, conditional, eta, sign)
        ok, neg_gap = _solve(-grad, a_ub, b_ub, a_eq, b_eq)
        if not ok:
            return "inconclusive"
        if -neg_gap > LP_TOL:
            gaps_ok = False
    # Feasibility of positive mass AND the tolerance, not of the cleared ratio alone.
    extra = []
    extra_rhs = []
    for sign in signs:
        grad = _gap_coeff(cell, conditional, eta, sign)
        extra.append(grad)
        extra_rhs.append(0.0)
    a_tv = np.vstack([a_ub, np.vstack(extra)]) if a_ub.size else np.vstack(extra)
    b_tv = np.concatenate([b_ub, np.asarray(extra_rhs)]) if b_ub.size else np.asarray(extra_rhs, dtype=float)
    ok_pos, neg_pos = _solve(-cell_c, a_tv, b_tv, a_eq, b_eq)
    positive_feasible = bool(ok_pos) and (-neg_pos) > LP_TOL
    if min_mass > LP_TOL and gaps_ok:
        return "certify"
    if not positive_feasible:
        return "reject"
    return "inconclusive"


def clause_replies(
    region: Region,
    release: np.ndarray,
    protected_cells: list[int],
    tau: float,
    eta: float,
) -> dict[tuple[str, int], str]:
    """Replies for protected cells the screened release actually populates.

    A cell with released count 0 is inactive and receives no reply.
    """
    replies: dict[tuple[str, int], str] = {}
    tau_of = tau if isinstance(tau, dict) else {}
    for cell in protected_cells:
        if released_cell_count(release, cell) <= 0:
            continue
        conditional = released_conditional(release, cell)
        if conditional is None:
            continue
        cell_tau = float(tau_of[cell]) if tau_of else float(tau)
        replies[("support", cell)] = support_reply(region, cell, cell_tau)
        replies[("fidelity", cell)] = fidelity_reply(region, cell, conditional, eta)
    return replies


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return 0.5 * float(np.sum(np.abs(p - q)))


def truth_in_support(theta: np.ndarray, cell: int, tau: float) -> bool:
    return bool(theta[cell] >= tau)


def truth_in_fidelity(theta: np.ndarray, cell: int, conditional: np.ndarray, eta: float) -> bool:
    """Population cell mass is positive and total variation is at most eta."""
    mass = float(theta[cell])
    if mass <= 0.0:
        return False
    joints = theta[N_KEY + cell * N_OUT : N_KEY + (cell + 1) * N_OUT]
    population_conditional = joints / mass
    return total_variation(conditional, population_conditional) <= eta + 1e-12


def counts_from_probs(probs: np.ndarray, n: int) -> np.ndarray:
    """Integer counts with the given probabilities and exact sum n."""
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    raw = probs * n
    counts = np.floor(raw).astype(int)
    left = int(n - counts.sum())
    order = np.argsort(-(raw - counts))
    for i in range(left):
        counts[int(order[i])] += 1
    return counts


def flip_protected_conditional(
    release: np.ndarray,
    cell: int,
    flipped: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Rewrite outcomes on one protected cell so the released conditional matches `flipped`."""
    out = np.array(release, dtype=int, copy=True)
    rows = np.where(out[:, 0] == cell)[0]
    if rows.size == 0:
        return out
    counts = counts_from_probs(flipped, int(rows.size))
    labels = np.repeat(np.arange(N_OUT), counts)
    rng.shuffle(labels)
    out[rows, 1] = labels
    return out
