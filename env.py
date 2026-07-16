"""A lightweight numerical environment for the data-marketplace mechanism."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .mechanism import (
    PaymentDivisionResult,
    PriceUpdateState,
    closed_form_ground_truth,
    data_allocation_function,
    fit_linear_prediction,
    myerson_revenue_function,
    normalized_rmse_gain,
    robust_payment_division,
    subset_linear_gain,
)


@dataclass(frozen=True)
class Seller:
    seller_id: str
    feature: np.ndarray
    cost: float = 0.0


@dataclass(frozen=True)
class Buyer:
    buyer_id: str
    y: np.ndarray
    mu: float
    bid: Optional[float] = None


@dataclass
class StepResult:
    price: float
    bid: float
    allocated_features: np.ndarray
    gain: float
    revenue: float
    buyer_utility: float
    seller_utilities: np.ndarray
    platform_utility: float
    payment_division: PaymentDivisionResult


class MarketplaceForDataEnv:
    """Single-market environment with fixed sellers and sequential buyers.

    Buyers submit scalar bids. The platform chooses a scalar price using PF,
    allocates degraded data through AF, charges RF, and divides revenue through
    robust PD. By default buyers are truthful, i.e. bid = mu.
    """

    def __init__(
        self,
        sellers: Sequence[Seller],
        *,
        price_state: PriceUpdateState,
        af_mode: str = "gaussian",
        noise_sigma: float = 1.0,
        shapley_permutations: int = 256,
        lambda_penalty: float = float(np.log(2.0)),
        normalize_pd: bool = False,
        seed: Optional[int] = None,
    ):
        if len(sellers) == 0:
            raise ValueError("At least one seller is required.")
        self.sellers = list(sellers)
        self.price_state = price_state
        self.af_mode = af_mode
        self.noise_sigma = float(noise_sigma)
        self.shapley_permutations = int(shapley_permutations)
        self.lambda_penalty = float(lambda_penalty)
        self.normalize_pd = bool(normalize_pd)
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.history: list[StepResult] = []

    @property
    def X(self) -> np.ndarray:
        return np.vstack([np.asarray(s.feature, dtype=float).reshape(1, -1) for s in self.sellers])

    @property
    def seller_costs(self) -> np.ndarray:
        return np.array([float(s.cost) for s in self.sellers], dtype=float)

    def step(self, buyer: Buyer, *, bid: Optional[float] = None, deterministic_price: bool = False, external_price: Optional[float] = None) -> StepResult:
        actual_bid = float(buyer.mu if bid is None and buyer.bid is None else (buyer.bid if bid is None else bid))
        if external_price is not None:
            price = float(external_price)
        else:
            price = self.price_state.choose_price(self.rng, deterministic=deterministic_price)
        allocated = data_allocation_function(
            price,
            actual_bid,
            self.X,
            mode=self.af_mode,
            noise_sigma=self.noise_sigma,
            rng=self.rng,
        )
        y = np.asarray(buyer.y, dtype=float).reshape(-1)
        yhat = fit_linear_prediction(allocated, y)
        gain = normalized_rmse_gain(y, yhat)

        def gain_at_bid(z: float) -> float:
            q = 1.0 if price <= 0 else min(max(float(z) / float(price), 0.0), 1.0)
            return float(q * gain)

        revenue = myerson_revenue_function(price, actual_bid, gain_at_bid=gain_at_bid)
        value_fn = lambda subset: subset_linear_gain(y, allocated, subset)
        division = robust_payment_division(
            revenue,
            allocated,
            value_fn,
            num_permutations=self.shapley_permutations,
            lambda_penalty=self.lambda_penalty,
            normalize=self.normalize_pd,
            rng=self.rng,
        )
        seller_utils = division.seller_payments - self.seller_costs
        buyer_utility = float(float(buyer.mu) * gain - revenue)
        platform_utility = float(revenue - np.sum(division.seller_payments))
        result = StepResult(
            price=price,
            bid=actual_bid,
            allocated_features=allocated,
            gain=float(gain),
            revenue=float(revenue),
            buyer_utility=buyer_utility,
            seller_utilities=seller_utils,
            platform_utility=platform_utility,
            payment_division=division,
        )
        self.price_state.update(actual_bid)
        self.history.append(result)
        self.t += 1
        return result

    def utilities(self) -> dict:
        if not self.history:
            return {
                "platform": 0.0,
                "buyers": np.zeros(0, dtype=float),
                "sellers": np.zeros(len(self.sellers), dtype=float),
            }
        return {
            "platform": float(np.sum([h.platform_utility for h in self.history])),
            "buyers": np.array([h.buyer_utility for h in self.history], dtype=float),
            "sellers": np.sum([h.seller_utilities for h in self.history], axis=0),
        }

    def ground_truth(self, buyers: Sequence[Buyer]) -> dict:
        mus = [float(b.mu) for b in buyers]
        bounds = (float(np.min(self.price_state.candidates)), float(np.max(self.price_state.candidates)))
        if self.history:
            shares = self.history[-1].payment_division.shares
        else:
            shares = np.ones(len(self.sellers), dtype=float) / len(self.sellers)
        return closed_form_ground_truth(
            mus,
            price_bounds=bounds,
            seller_costs=self.seller_costs,
            payment_shares=shares,
        )
