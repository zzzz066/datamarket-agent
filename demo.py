"""Small runnable example for the marketplace_for_data package."""

from __future__ import annotations

import numpy as np

from .env.base import Buyer, MarketplaceForDataEnv, Seller
from .mechanism.core import PriceUpdateState


def main() -> None:
    t = np.linspace(0.0, 1.0, 80)
    y = np.sin(2.0 * np.pi * t)
    sellers = [
        Seller("A", y + 0.05 * np.cos(6.0 * np.pi * t), cost=0.02),
        Seller("B", np.cos(2.0 * np.pi * t), cost=0.02),
        Seller("A_copy", y + 0.05 * np.cos(6.0 * np.pi * t), cost=0.02),
    ]
    buyers = [
        Buyer("buyer_1", y, mu=0.4),
        Buyer("buyer_2", y, mu=0.8),
        Buyer("buyer_3", y, mu=1.2),
    ]
    price_state = PriceUpdateState.from_bounds(0.1, 1.5, epsilon=0.1, delta=0.2)
    env = MarketplaceForDataEnv(sellers, price_state=price_state, seed=7, shapley_permutations=64)
    for buyer in buyers:
        result = env.step(buyer, deterministic_price=True)
        print(
            buyer.buyer_id,
            "price=", round(result.price, 3),
            "bid=", round(result.bid, 3),
            "gain=", round(result.gain, 3),
            "revenue=", round(result.revenue, 3),
            "seller_shares=", np.round(result.payment_division.shares, 3).tolist(),
        )
    print("utilities:", {k: np.round(v, 4).tolist() if hasattr(v, "tolist") else round(v, 4) for k, v in env.utilities().items()})
    gt = env.ground_truth(buyers)
    print("ground_truth:", {k: np.round(v, 4).tolist() if hasattr(v, "tolist") else round(v, 4) for k, v in gt.items()})


if __name__ == "__main__":
    main()
