"""Agent-facing wrapper around MarketplaceForDataEnv.

Splits each round into two decision points:
  1. Platform agent chooses a price p
  2. Buyer agent chooses a bid b

The underlying AF / RF / PD / PF mechanisms (mechanism.py) are unchanged.
This wrapper only controls *who* makes the price and bid decisions —
rule-based or LLM agent — and translates internal state into agent-readable
observation dicts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .env import Buyer, MarketplaceForDataEnv, Seller, StepResult
from .mechanism import PriceUpdateState


# ---------------------------------------------------------------------------
# Observation schemas (what each agent sees)
# ---------------------------------------------------------------------------

@dataclass
class PlatformObservation:
    """State visible to the platform agent before it picks a price."""

    round: int
    total_rounds: int
    candidates: List[float]            # price candidate grid
    weights: List[float]               # MWU weights (normalised to sum=1)
    best_mwu_price: float              # argmax of current weights
    seller_count: int
    recent_history: List[Dict[str, Any]]  # last K rounds


@dataclass
class BuyerObservation:
    """State visible to the buyer agent before it submits a bid."""

    round: int
    total_rounds: int
    mu: float                          # buyer's true valuation
    current_price: float               # price posted by platform this round
    my_history: List[Dict[str, Any]]   # this buyer's past rounds (bid, utility, …)


# ---------------------------------------------------------------------------
# Reward schema (returned after each round settles)
# ---------------------------------------------------------------------------

@dataclass
class RoundReward:
    buyer_id: str
    price: float
    bid: float
    buyer_utility: float
    platform_utility: float
    seller_utilities: List[float]
    revenue: float
    gain: float


def _make_platform_obs(
    round_idx: int,
    total_rounds: int,
    price_state: PriceUpdateState,
    sellers: List[Seller],
    history: List[StepResult],
    recent_k: int = 10,
) -> Dict[str, Any]:
    probs = price_state.probabilities
    best_idx = int(np.argmax(probs))
    recent = []
    for h in history[-recent_k:]:
        recent.append({
            "round": h.t if hasattr(h, 't') else len(recent),
            "price": round(float(h.price), 4),
            "bid": round(float(h.bid), 4),
            "revenue": round(float(h.revenue), 4),
            "platform_utility": round(float(h.platform_utility), 4),
            "buyer_utility": round(float(h.buyer_utility), 4),
            "gain": round(float(h.gain), 4),
        })
    return {
        "round": round_idx + 1,
        "total_rounds": total_rounds,
        "candidates": [round(float(c), 4) for c in price_state.candidates],
        "weights": [round(float(w), 6) for w in probs],
        "best_mwu_price": round(float(price_state.candidates[best_idx]), 4),
        "seller_count": len(sellers),
        "recent_history": recent,
    }


def _make_buyer_obs(
    round_idx: int,
    total_rounds: int,
    mu: float,
    price: float,
    buyer_history: List[Dict[str, Any]],
    recent_k: int = 5,
) -> Dict[str, Any]:
    return {
        "round": round_idx + 1,
        "total_rounds": total_rounds,
        "mu": round(float(mu), 4),
        "current_price": round(float(price), 4),
        "my_history": buyer_history[-recent_k:],
    }


# ---------------------------------------------------------------------------
# Main wrapper
# ---------------------------------------------------------------------------

class AgentMarketEnv:
    """Gymnasium-style env for the ADS 2019 data marketplace with agents.

    Usage
    -----
    >>> env = AgentMarketEnv(sellers, buyers, price_bounds=(0.1, 1.6, 0.05))
    >>> obs = env.reset()
    >>> while not env.done:
    ...     if env.current_role == "platform":
    ...         price = platform_agent.act(obs)
    ...         obs = env.step("platform", {"price": price})
    ...     elif env.current_role == "buyer":
    ...         bid = buyer_agent.act(obs)
    ...         obs, reward = env.step("buyer", {"bid": bid})
    """

    def __init__(
        self,
        sellers: Sequence[Seller],
        buyers: Sequence[Buyer],
        *,
        price_bounds: tuple[float, float, float] = (0.1, 1.6, 0.05),
        delta: float = 0.18,
        af_mode: str = "gaussian",
        noise_sigma: float = 1.0,
        shapley_permutations: int = 256,
        lambda_penalty: float = float(np.log(2.0)),
        seed: Optional[int] = None,
        platform_mode: str = "agent",     # "agent" | "mwu"
        buyer_mode: str = "agent",         # "agent" | "truthful" | "shade" | "overbid"
        shade_factor: float = 0.65,
        overbid_factor: float = 1.4,
    ):
        if len(sellers) == 0:
            raise ValueError("At least one seller is required.")
        if len(buyers) == 0:
            raise ValueError("At least one buyer is required.")
        lower, upper, epsilon = price_bounds
        self.sellers = list(sellers)
        self.buyers = list(buyers)
        self.total_rounds = len(buyers)
        self.price_bounds = (float(lower), float(upper), float(epsilon))
        self.delta = float(delta)
        self.af_mode = af_mode
        self.noise_sigma = float(noise_sigma)
        self.shapley_permutations = int(shapley_permutations)
        self.lambda_penalty = float(lambda_penalty)
        self.seed = seed
        self.platform_mode = platform_mode
        self.buyer_mode = buyer_mode
        self.shade_factor = float(shade_factor)
        self.overbid_factor = float(overbid_factor)

        # internal state — set in reset()
        self._env: Optional[MarketplaceForDataEnv] = None
        self._round_idx: int = 0
        self._pending_price: Optional[float] = None
        self._pending_buyer: Optional[Buyer] = None
        self._buyer_histories: Dict[str, List[Dict[str, Any]]] = {}
        self._done: bool = False

    # ------------------------------------------------------------------
    # Core loop API
    # ------------------------------------------------------------------

    def reset(self) -> Dict[str, Any]:
        """Start a new episode. Returns the first observation."""
        price_state = PriceUpdateState.from_bounds(
            self.price_bounds[0],
            self.price_bounds[1],
            epsilon=self.price_bounds[2],
            delta=self.delta,
        )
        self._env = MarketplaceForDataEnv(
            self.sellers,
            price_state=price_state,
            af_mode=self.af_mode,
            noise_sigma=self.noise_sigma,
            shapley_permutations=self.shapley_permutations,
            lambda_penalty=self.lambda_penalty,
            seed=self.seed,
        )
        self._round_idx = 0
        self._pending_price = None
        self._pending_buyer = None
        self._buyer_histories = {b.buyer_id: [] for b in self.buyers}
        self._done = False

        # If platform is rule-based, auto-step through platform until
        # we hit a buyer decision (or done).
        return self._advance_to_next_agent_decision()

    def step(self, role: str, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an agent decision and advance the round.

        Parameters
        ----------
        role : "platform" or "buyer"
        action : {"price": float} or {"bid": float}

        Returns
        -------
        obs : dict — next agent's observation, or the final-round reward dict
               when the episode is over.
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset().")
        if role != self.current_role:
            raise ValueError(
                f"Expected role={self.current_role}, got role={role}."
            )

        if role == "platform":
            return self._step_platform(action)
        else:
            return self._step_buyer(action)

    @property
    def current_role(self) -> str:
        """Which agent should act next?"""
        if self._done:
            return "done"
        if self._pending_price is None:
            return "platform"
        return "buyer"

    @property
    def done(self) -> bool:
        return self._done

    # ------------------------------------------------------------------
    # Observation helpers (can be called any time for debugging)
    # ------------------------------------------------------------------

    def get_obs(self, role: str) -> Dict[str, Any]:
        """Snapshot of what ``role`` would see right now."""
        if self._env is None:
            raise RuntimeError("Call reset() first.")
        if role == "platform":
            return _make_platform_obs(
                self._round_idx,
                self.total_rounds,
                self._env.price_state,
                self._env.sellers,
                self._env.history,
            )
        elif role == "buyer":
            if self._pending_buyer is None:
                raise RuntimeError("No pending buyer — platform must act first.")
            return _make_buyer_obs(
                self._round_idx,
                self.total_rounds,
                self._pending_buyer.mu,
                self._pending_price,  # type: ignore[arg-type]
                self._buyer_histories.get(self._pending_buyer.buyer_id, []),
            )
        else:
            raise ValueError(f"Unknown role: {role}")

    # ------------------------------------------------------------------
    # Reward & ground truth
    # ------------------------------------------------------------------

    def get_last_reward(self) -> Dict[str, Any]:
        """Reward for the most recently settled round."""
        if self._env is None or not self._env.history:
            return {}
        h = self._env.history[-1]
        return {
            "round": len(self._env.history),
            "buyer_id": self.buyers[min(self._round_idx - 1, len(self.buyers) - 1)].buyer_id,
            "price": round(float(h.price), 4),
            "bid": round(float(h.bid), 4),
            "buyer_utility": round(float(h.buyer_utility), 6),
            "platform_utility": round(float(h.platform_utility), 6),
            "seller_utilities": [round(float(u), 6) for u in h.seller_utilities],
            "revenue": round(float(h.revenue), 6),
            "gain": round(float(h.gain), 6),
        }

    def ground_truth(self) -> Dict[str, Any]:
        """Theoretical benchmark for the current set of buyers."""
        if self._env is None:
            raise RuntimeError("Call reset() first.")
        gt = self._env.ground_truth(self.buyers)
        return {
            "price_star": round(float(gt["price_star"]), 4),
            "total_revenue_star": round(float(gt["total_revenue_star"]), 4),
            "buyer_utilities_star": [round(float(u), 6) for u in gt["buyer_utilities_star"]],
            "platform_utility_star": round(float(gt["platform_utility_star"]), 6),
        }

    def episode_summary(self) -> Dict[str, Any]:
        """Aggregated utilities over the whole episode."""
        if self._env is None:
            return {}
        u = self._env.utilities()
        return {
            "total_platform_utility": round(float(u["platform"]), 6),
            "total_buyer_utilities": [round(float(v), 6) for v in u["buyers"]],
            "total_seller_utilities": [round(float(v), 6) for v in u["sellers"]],
            "num_rounds": len(self._env.history),
        }

    # ------------------------------------------------------------------
    # Internal step logic
    # ------------------------------------------------------------------

    def _advance_to_next_agent_decision(self) -> Dict[str, Any]:
        """Move state forward until an agent decision is needed (or done).

        If the platform is rule-based (MWU), automatically pick the price.
        If the buyer is rule-based, automatically submit the bid and loop.
        """
        while not self._done:
            if self._round_idx >= self.total_rounds:
                self._done = True
                return self.get_last_reward()

            if self._pending_price is None:
                # Need a platform decision
                if self.platform_mode == "mwu":
                    # Rule-based: auto-pick and skip to buyer
                    self._pending_price = self._env.price_state.choose_price(
                        self._env.rng, deterministic=True
                    )
                    self._pending_buyer = self.buyers[self._round_idx]
                    # fall through to buyer
                else:
                    self._pending_buyer = self.buyers[self._round_idx]
                    return self.get_obs("platform")

            if self._pending_price is not None:
                # Need a buyer decision
                if self.buyer_mode != "agent":
                    # Rule-based buyer: compute bid automatically
                    mu = self._pending_buyer.mu  # type: ignore[union-attr]
                    if self.buyer_mode == "truthful":
                        bid = mu
                    elif self.buyer_mode == "shade":
                        bid = mu * self.shade_factor
                    elif self.buyer_mode == "overbid":
                        bid = mu * self.overbid_factor
                    else:
                        raise ValueError(f"Unknown buyer_mode: {self.buyer_mode}")
                    self._settle_round(bid)
                    # loop to next round
                else:
                    return self.get_obs("buyer")

        return self.get_last_reward()

    def _step_platform(self, action: Dict[str, Any]) -> Dict[str, Any]:
        price = float(action["price"])
        lower, upper, _ = self.price_bounds
        price = max(lower, min(upper, price))  # clamp to valid range
        self._pending_price = price
        self._pending_buyer = self.buyers[self._round_idx]
        return self._advance_to_next_agent_decision()

    def _step_buyer(self, action: Dict[str, Any]) -> Dict[str, Any]:
        bid = float(action["bid"])
        bid = max(0.0, bid)  # bids must be non-negative
        self._settle_round(bid)
        return self._advance_to_next_agent_decision()

    def _settle_round(self, bid: float) -> None:
        """Execute AF/RF/PD/PF for this round."""
        buyer = self.buyers[self._round_idx]
        price = self._pending_price  # type: ignore[assignment]
        result = self._env.step(  # type: ignore[union-attr]
            buyer,
            bid=bid,
            external_price=price,
        )
        # Record in buyer's personal history
        hist_entry = {
            "round": self._round_idx + 1,
            "mu": round(float(buyer.mu), 4),
            "price": round(float(price), 4),  # type: ignore[arg-type]
            "bid": round(float(bid), 4),
            "utility": round(float(result.buyer_utility), 6),
            "gain": round(float(result.gain), 4),
        }
        self._buyer_histories[buyer.buyer_id].append(hist_entry)
        # Advance
        self._round_idx += 1
        self._pending_price = None
        self._pending_buyer = None
