"""Backend simulation payloads for the visual dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .mechanism.core import (
    closed_form_myerson_revenue,
    closed_form_quality_gain,
    optimal_fixed_price_closed_form,
    robust_payment_division,
)


def _buyer_values(n: int) -> np.ndarray:
    values = []
    for i in range(int(n)):
        wave = 0.5 + 0.5 * np.sin(i * 1.71)
        trend = (i % 17) / 17.0
        values.append(0.25 + 1.25 * (0.65 * wave + 0.35 * trend))
    return np.asarray(values, dtype=float)


def _strategy_stats(mu: np.ndarray, p_star: float, shade: float, overbid: float) -> list[dict[str, Any]]:
    specs = [
        ("low", "低报", shade * mu),
        ("truth", "诚实", mu),
        ("over", "高报", overbid * mu),
    ]
    out = []
    for key, label, bids in specs:
        quality = np.array([closed_form_quality_gain(p_star, b) for b in bids], dtype=float)
        payment = np.array([closed_form_myerson_revenue(p_star, b) for b in bids], dtype=float)
        utility = mu * quality - payment
        out.append(
            {
                "key": key,
                "label": label,
                "avg_bid": float(np.mean(bids)),
                "avg_quality": float(np.mean(quality)),
                "avg_payment": float(np.mean(payment)),
                "avg_utility": float(np.mean(utility)),
            }
        )
    return out


def _mwu_prices(mu: np.ndarray, lower: float, upper: float, *, step: float = 0.05, delta: float = 0.18) -> list[float]:
    candidates = np.arange(lower, upper + 0.5 * step, step, dtype=float)
    weights = np.ones_like(candidates, dtype=float)
    prices = []
    for value in mu:
        idx = int(np.argmax(weights))
        prices.append(float(candidates[idx]))
        gains = np.array([closed_form_myerson_revenue(p, value) / upper for p in candidates], dtype=float)
        weights *= 1.0 + delta * np.clip(gains, 0.0, 1.0)
    return prices


def _seller_share_demo(lambda_penalty: float) -> dict[str, Any]:
    # A and A_copy are perfect substitutes; B is unrelated for this task.
    x_features = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    def value_fn(subset: list[int]) -> float:
        has_signal = 0 in subset or 2 in subset
        return 1.0 if has_signal else 0.0

    result = robust_payment_division(
        1.0,
        x_features,
        value_fn,
        lambda_penalty=float(lambda_penalty),
        exact=True,
        normalize=False,
    )
    return {
        "labels": ["原始 A", "无关 B", "复制 A'"],
        "shapley": result.shapley.tolist(),
        "penalties": result.penalties.tolist(),
        "shares": result.shares.tolist(),
        "copy_ratio": float(result.shares[2] / result.shares[0]) if result.shares[0] > 0 else None,
    }


def simulate_dashboard(
    *,
    buyers: int = 120,
    shade: float = 0.65,
    overbid: float = 1.4,
    lambda_penalty: float = 0.7,
    lower_price: float = 0.1,
    upper_price: float = 1.6,
) -> dict[str, Any]:
    """Return all numeric data needed by the HTML dashboard."""

    buyer_count = int(np.clip(buyers, 20, 240))
    shade = float(np.clip(shade, 0.2, 0.95))
    overbid = float(np.clip(overbid, 1.05, 2.2))
    lambda_penalty = float(np.clip(lambda_penalty, 0.0, 2.0))
    mu = _buyer_values(buyer_count)
    p_star, revenue_star = optimal_fixed_price_closed_form(mu, (lower_price, upper_price))
    strategies = _strategy_stats(mu, p_star, shade, overbid)
    truth = next(s for s in strategies if s["key"] == "truth")
    best_alt = max(s["avg_utility"] for s in strategies if s["key"] != "truth")
    prices = _mwu_prices(mu, lower_price, upper_price)
    shares = _seller_share_demo(lambda_penalty)
    return {
        "params": {
            "buyers": buyer_count,
            "shade": shade,
            "overbid": overbid,
            "lambda_penalty": lambda_penalty,
            "lower_price": lower_price,
            "upper_price": upper_price,
        },
        "ground_truth": {
            "price_star": float(p_star),
            "total_revenue_star": float(revenue_star),
            "truthful_bidding": "b*=mu",
        },
        "strategies": strategies,
        "truth_gap": float(truth["avg_utility"] - best_alt),
        "mwu_prices": prices,
        "seller_demo": shares,
    }
