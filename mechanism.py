"""Core mechanisms from "A Marketplace for Data: An Algorithmic Solution".

The implementation follows the paper's section 4 construction:

* AF degrades allocated data when bid is below the posted market price.
* RF is Myerson's payment rule for a single-parameter buyer.
* PF is multiplicative weights over a bounded price candidate net.
* PD estimates Shapley marginal contribution and applies an exponential
  similarity penalty for robustness to replication.

The closed-form ground truth uses the standard monotone quality curve
q_p(b) = min(b / p, 1), which is the deterministic counterpart of the paper's
"quality increases with bid" assumption and gives analytic RF and optimal
fixed-price formulas.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Callable, Iterable, Optional, Sequence

import numpy as np


Array = np.ndarray
GainFunction = Callable[[Array, Array], float]
ValueFunction = Callable[[Sequence[int]], float]


def _as_2d_features(x: Array) -> Array:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"X must be a 1-D or 2-D array, got shape {arr.shape}.")
    return arr


def _quality_ratio(price: float, bid: float) -> float:
    price = float(price)
    bid = float(bid)
    if price <= 0:
        return 1.0
    return float(np.clip(bid / price, 0.0, 1.0))


def normalized_rmse_gain(y_true: Array, y_pred: Array, *, eps: float = 1e-12) -> float:
    """Return a clipped gain score in [0, 1] using 1 - normalized RMSE."""

    y = np.asarray(y_true, dtype=float).reshape(-1)
    yhat = np.asarray(y_pred, dtype=float).reshape(-1)
    if y.shape != yhat.shape:
        raise ValueError(f"y_true and y_pred must have the same shape, got {y.shape} and {yhat.shape}.")
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
    scale = float(np.max(y) - np.min(y))
    if scale <= eps:
        scale = float(np.std(y))
    if scale <= eps:
        return 1.0 if rmse <= eps else 0.0
    return float(np.clip(1.0 - rmse / scale, 0.0, 1.0))


def fit_linear_prediction(x_features: Array, y: Array) -> Array:
    """Fit an OLS model on selected features and return in-sample predictions."""

    X = _as_2d_features(x_features).T
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    if X.shape[0] != y_arr.shape[0]:
        raise ValueError(f"Feature length {X.shape[0]} does not match y length {y_arr.shape[0]}.")
    design = np.column_stack([np.ones(X.shape[0]), X])
    coef, *_ = np.linalg.lstsq(design, y_arr, rcond=None)
    return design @ coef


def subset_linear_gain(y: Array, x_features: Array, subset: Sequence[int], gain_fn: GainFunction = normalized_rmse_gain) -> float:
    """Value function G(Y, M(X_S)) for Shapley calculations."""

    if len(subset) == 0:
        baseline = np.full_like(np.asarray(y, dtype=float), float(np.mean(y)), dtype=float)
        return gain_fn(y, baseline)
    X = _as_2d_features(x_features)
    yhat = fit_linear_prediction(X[list(subset)], y)
    return gain_fn(y, yhat)


def data_allocation_function(
    price: float,
    bid: float,
    x_features: Array,
    *,
    mode: str = "gaussian",
    noise_sigma: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> Array:
    """AF(p, b; X): allocate full data if b >= p, otherwise degrade data.

    For real-valued features, ``mode="gaussian"`` implements the paper's
    Example 4.1: X_tilde = X + max(0, p - b) * N(0, sigma^2).
    For binary features, ``mode="masking"`` implements Example 4.2 with
    Bernoulli keep probability min(b / p, 1).
    """

    X = np.asarray(x_features, dtype=float)
    shortfall = max(0.0, float(price) - float(bid))
    if shortfall <= 0.0:
        return X.copy()
    if rng is None:
        rng = np.random.default_rng()
    if mode == "gaussian":
        return X + shortfall * rng.normal(0.0, float(noise_sigma), size=X.shape)
    if mode == "masking":
        keep_prob = _quality_ratio(price, bid)
        return X * rng.binomial(1, keep_prob, size=X.shape)
    raise ValueError(f"Unsupported AF mode {mode!r}. Expected 'gaussian' or 'masking'.")


def closed_form_quality_gain(price: float, bid: float) -> float:
    """Analytic monotone quality curve h_p(b)=min(b/p, 1)."""

    return _quality_ratio(price, bid)


def closed_form_myerson_revenue(price: float, bid: float) -> float:
    """Closed-form RF for h_p(b)=min(b/p, 1)."""

    p = max(float(price), 1e-12)
    b = max(float(bid), 0.0)
    if b < p:
        return float((b * b) / (2.0 * p))
    return float(p / 2.0)


def myerson_revenue_function(
    price: float,
    bid: float,
    *,
    gain_at_bid: Optional[Callable[[float], float]] = None,
    integration_steps: int = 256,
) -> float:
    """RF(p, b, Y): Myerson payment b*h(b) - integral_0^b h(z) dz.

    If ``gain_at_bid`` is omitted, the analytic h_p(b)=min(b/p, 1) curve is
    used. Supplying ``gain_at_bid`` allows numerical integration around an
    actual ML pipeline while preserving the same payment formula.
    """

    if gain_at_bid is None:
        return closed_form_myerson_revenue(price, bid)
    b = max(float(bid), 0.0)
    if b == 0.0:
        return 0.0
    grid = np.linspace(0.0, b, int(max(2, integration_steps)))
    gains = np.array([float(np.clip(gain_at_bid(z), 0.0, 1.0)) for z in grid], dtype=float)
    integral = float(np.trapz(gains, grid))
    return float(max(0.0, b * gains[-1] - integral))


@dataclass
class PriceUpdateState:
    """PF state for multiplicative weights over scalar prices."""

    candidates: Array
    weights: Array
    delta: float = 0.1
    b_max: Optional[float] = None

    @classmethod
    def from_bounds(
        cls,
        lower: float,
        upper: float,
        *,
        epsilon: float,
        delta: float = 0.1,
    ) -> "PriceUpdateState":
        if upper <= lower:
            raise ValueError("upper must be greater than lower.")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive.")
        candidates = np.arange(float(lower), float(upper) + 0.5 * float(epsilon), float(epsilon))
        return cls(candidates=candidates, weights=np.ones_like(candidates, dtype=float), delta=delta, b_max=float(upper))

    @property
    def probabilities(self) -> Array:
        total = float(np.sum(self.weights))
        if total <= 0:
            return np.ones_like(self.weights) / len(self.weights)
        return self.weights / total

    def choose_price(self, rng: Optional[np.random.Generator] = None, *, deterministic: bool = False) -> float:
        if deterministic:
            return float(self.candidates[int(np.argmax(self.probabilities))])
        if rng is None:
            rng = np.random.default_rng()
        idx = int(rng.choice(len(self.candidates), p=self.probabilities))
        return float(self.candidates[idx])

    def update(self, bid: float, revenue_fn: Callable[[float, float], float] = closed_form_myerson_revenue) -> None:
        normalizer = float(self.b_max if self.b_max is not None else np.max(self.candidates))
        normalizer = max(normalizer, 1e-12)
        gains = np.array([revenue_fn(float(c), float(bid)) / normalizer for c in self.candidates], dtype=float)
        gains = np.clip(gains, 0.0, 1.0)
        self.weights *= (1.0 + float(self.delta) * gains)


def exact_shapley_values(num_features: int, value_fn: ValueFunction) -> Array:
    """Exact Shapley values by enumerating all permutations."""

    M = int(num_features)
    if M < 0:
        raise ValueError("num_features must be non-negative.")
    phi = np.zeros(M, dtype=float)
    if M == 0:
        return phi
    perms = list(permutations(range(M)))
    for perm in perms:
        prefix: list[int] = []
        prev = float(value_fn(prefix))
        for m in perm:
            prefix.append(m)
            new = float(value_fn(prefix))
            phi[m] += new - prev
            prev = new
    return phi / float(len(perms))


def shapley_approximation(
    num_features: int,
    value_fn: ValueFunction,
    *,
    num_permutations: int = 256,
    rng: Optional[np.random.Generator] = None,
) -> Array:
    """Algorithm 2: permutation-sampling Shapley approximation."""

    M = int(num_features)
    if M < 0:
        raise ValueError("num_features must be non-negative.")
    if rng is None:
        rng = np.random.default_rng()
    phi = np.zeros(M, dtype=float)
    if M == 0:
        return phi
    K = int(max(1, num_permutations))
    for _ in range(K):
        perm = list(rng.permutation(M))
        prefix: list[int] = []
        prev = float(value_fn(prefix))
        for m in perm:
            prefix.append(int(m))
            new = float(value_fn(prefix))
            phi[int(m)] += new - prev
            prev = new
    return phi / float(K)


def cosine_similarity_matrix(x_features: Array, *, eps: float = 1e-12) -> Array:
    """Cosine similarity matrix scaled to [0, 1] by absolute cosine."""

    X = _as_2d_features(x_features)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    denom = np.maximum(norms @ norms.T, eps)
    sim = np.abs((X @ X.T) / denom)
    np.fill_diagonal(sim, 1.0)
    return np.clip(sim, 0.0, 1.0)


@dataclass
class PaymentDivisionResult:
    shapley: Array
    penalties: Array
    shares: Array
    seller_payments: Array
    retained_by_platform: float


def robust_payment_division(
    revenue: float,
    x_features: Array,
    value_fn: ValueFunction,
    *,
    num_permutations: int = 256,
    lambda_penalty: float = np.log(2.0),
    similarity_matrix: Optional[Array] = None,
    exact: bool = False,
    normalize: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> PaymentDivisionResult:
    """Algorithm 3: Shapley values with exponential similarity penalties.

    ``normalize=False`` follows the paper's robust-to-replication construction,
    which intentionally need not satisfy balance. Set ``normalize=True`` if a
    budget-balanced payout vector is needed for experiments.
    """

    X = _as_2d_features(x_features)
    M = X.shape[0]
    shapley = exact_shapley_values(M, value_fn) if exact else shapley_approximation(
        M, value_fn, num_permutations=num_permutations, rng=rng
    )
    if similarity_matrix is None:
        similarity_matrix = cosine_similarity_matrix(X)
    sim = np.asarray(similarity_matrix, dtype=float)
    if sim.shape != (M, M):
        raise ValueError(f"similarity_matrix must have shape {(M, M)}, got {sim.shape}.")
    cumulative_similarity = np.sum(sim, axis=1) - np.diag(sim)
    penalties = np.exp(-float(lambda_penalty) * cumulative_similarity)
    shares = np.maximum(0.0, shapley * penalties)
    if normalize and float(np.sum(shares)) > 0:
        shares = shares / float(np.sum(shares))
    payments = float(revenue) * shares
    retained = float(revenue) - float(np.sum(payments))
    return PaymentDivisionResult(
        shapley=shapley,
        penalties=penalties,
        shares=shares,
        seller_payments=payments,
        retained_by_platform=retained,
    )


def _candidate_interval_points(mu_values: Array, lower: float, upper: float) -> Iterable[float]:
    vals = np.sort(np.asarray(mu_values, dtype=float))
    boundaries = [float(lower)] + [float(v) for v in vals if lower <= v <= upper] + [float(upper)]
    for point in boundaries:
        if lower <= point <= upper:
            yield point
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        mid = 0.5 * (left + right)
        k = int(np.sum(vals >= mid))
        under = vals[vals < mid]
        if k > 0:
            stationary = float(np.sqrt(np.sum(under ** 2) / k)) if under.size else 0.0
            if left <= stationary <= right and lower <= stationary <= upper:
                yield stationary


def optimal_fixed_price_closed_form(mu_values: Sequence[float], price_bounds: tuple[float, float]) -> tuple[float, float]:
    """Closed-form best fixed p for sum_n RF(p, mu_n) under q_p(b)."""

    mu = np.maximum(0.0, np.asarray(mu_values, dtype=float).reshape(-1))
    lower, upper = map(float, price_bounds)
    if mu.size == 0:
        return lower, 0.0
    if lower <= 0:
        positive = mu[mu > 0]
        lower = min(float(np.min(positive)), upper) if positive.size else 1e-12
    candidates = sorted(set(round(p, 12) for p in _candidate_interval_points(mu, lower, upper)))
    if not candidates:
        candidates = [lower, upper]
    revenues = np.array([np.sum([closed_form_myerson_revenue(p, m) for m in mu]) for p in candidates])
    best = int(np.argmax(revenues))
    return float(candidates[best]), float(revenues[best])


def closed_form_ground_truth(
    mu_values: Sequence[float],
    *,
    price_bounds: tuple[float, float],
    seller_costs: Optional[Sequence[float]] = None,
    payment_shares: Optional[Sequence[float]] = None,
) -> dict:
    """Theoretical benchmark equilibrium for the simplified numeric model.

    Truthfulness gives b_n*=mu_n. The platform's no-regret benchmark is the
    best fixed price in hindsight over ``price_bounds``. Utilities are computed
    using q_p(b)=min(b/p,1) and RF's closed form.
    """

    mu = np.maximum(0.0, np.asarray(mu_values, dtype=float).reshape(-1))
    p_star, total_revenue = optimal_fixed_price_closed_form(mu, price_bounds)
    bids = mu.copy()
    gains = np.array([closed_form_quality_gain(p_star, b) for b in bids], dtype=float)
    payments = np.array([closed_form_myerson_revenue(p_star, b) for b in bids], dtype=float)
    buyer_utilities = mu * gains - payments
    if payment_shares is None:
        shares = np.zeros(0, dtype=float)
    else:
        shares = np.asarray(payment_shares, dtype=float).reshape(-1)
    if seller_costs is None:
        seller_cost_arr = np.zeros_like(shares)
    else:
        seller_cost_arr = np.asarray(seller_costs, dtype=float).reshape(-1)
        if shares.size and seller_cost_arr.shape != shares.shape:
            raise ValueError("seller_costs and payment_shares must have the same length.")
    seller_payments = total_revenue * shares
    seller_utilities = seller_payments - seller_cost_arr if shares.size else np.zeros(0, dtype=float)
    platform_utility = total_revenue - float(np.sum(seller_payments))
    return {
        "bids_star": bids,
        "price_star": p_star,
        "total_revenue_star": total_revenue,
        "gain_star": gains,
        "buyer_payments_star": payments,
        "buyer_utilities_star": buyer_utilities,
        "seller_payments_star": seller_payments,
        "seller_utilities_star": seller_utilities,
        "platform_utility_star": platform_utility,
    }
