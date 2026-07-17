"""面向 Agent 的市场环境封装层。

将 MarketplaceForDataEnv 的单步 step() 拆分为两个独立的 Agent 决策点：
  1. 平台 Agent 选择价格 p
  2. 买家 Agent 选择报价 b

底层 AF/RF/PD/PF 机制（mechanism/core.py）完全不变。
本层只控制"谁来做决策"——规则驱动还是 LLM Agent——并将内部状态
翻译成 Agent 可读的观察字典。

实验模式
--------
通过 platform_mode 和 buyer_mode 组合实现四种实验配置：

+------------------+------------------+-------------------------------+
| platform_mode    | buyer_mode       | 用途                          |
+==================+==================+===============================+
| "mwu"            | "truthful"       | 纯规则基线（论文原版行为）      |
+------------------+------------------+-------------------------------+
| "mwu"            | "agent"          | 仅测试买家 Agent 策略行为       |
+------------------+------------------+-------------------------------+
| "agent"          | "truthful"       | 仅测试平台 Agent 定价行为       |
+------------------+------------------+-------------------------------+
| "agent"          | "agent"          | 双边博弈（平台+买家都是Agent）   |
+------------------+------------------+-------------------------------+

买家额外支持 "shade"（低报）和 "overbid"（高报）两种规则策略，
用于与诚实报价对照。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .base import Buyer, MarketplaceForDataEnv, Seller
from ..mechanism import PriceUpdateState


# ---------------------------------------------------------------------------
# 观察构造（Agent 看到的"状态"）
# ---------------------------------------------------------------------------

def _make_platform_obs(
    round_idx: int,
    total_rounds: int,
    price_state: PriceUpdateState,
    seller_count: int,
    history: list,
    recent_k: int = 10,
) -> Dict[str, Any]:
    """构造平台 Agent 的观察字典。

    包含：候选价格网格、MWU 权重分布、最近 K 轮历史摘要。
    Agent 可利用 MWU 权重作为决策参考，也可以完全忽略它。
    """
    probs = price_state.probabilities
    best_idx = int(np.argmax(probs))
    recent = []
    for h in history[-recent_k:]:
        recent.append({
            "轮次": len(recent) + 1,
            "价格": round(float(h.price), 4),
            "报价": round(float(h.bid), 4),
            "平台收入": round(float(h.revenue), 4),
            "平台效用": round(float(h.platform_utility), 4),
            "买家效用": round(float(h.buyer_utility), 4),
            "增益": round(float(h.gain), 4),
        })
    return {
        "当前轮次": round_idx + 1,
        "总轮次": total_rounds,
        "候选价格": [round(float(c), 4) for c in price_state.candidates],
        "MWU权重": [round(float(w), 6) for w in probs],
        "MWU推荐价格": round(float(price_state.candidates[best_idx]), 4),
        "卖家数量": seller_count,
        "最近历史": recent,
    }


def _make_buyer_obs(
    round_idx: int,
    total_rounds: int,
    mu: float,
    price: float,
    buyer_history: List[Dict[str, Any]],
    recent_k: int = 5,
) -> Dict[str, Any]:
    """构造买家 Agent 的观察字典。

    包含：自身真实估值 mu、当前平台价格、自身历史决策记录。
    买家不知道其他买家的估值和历史（私有信息）。
    """
    return {
        "当前轮次": round_idx + 1,
        "总轮次": total_rounds,
        "我的估值_mu": round(float(mu), 4),
        "当前平台价格": round(float(price), 4),
        "我的历史": buyer_history[-recent_k:],
    }


# ---------------------------------------------------------------------------
# AgentMarketEnv：面向 Agent 的主环境类
# ---------------------------------------------------------------------------

class AgentMarketEnv:
    """Gymnasium 风格的数据市场环境，供 LLM Agent 驱动的模拟使用。

    使用示例
    --------
    >>> env = AgentMarketEnv(sellers, buyers, platform_mode="agent", buyer_mode="agent")
    >>> obs = env.reset()
    >>> while not env.done:
    ...     if env.current_role == "平台":
    ...         action = platform_agent.act(obs)
    ...         obs = env.step("平台", action)
    ...     elif env.current_role == "买家":
    ...         action = buyer_agent.act(obs)
    ...         obs = env.step("买家", action)
    >>> summary = env.episode_summary()
    >>> gt = env.ground_truth()   # 理论基准对照
    """

    def __init__(
        self,
        sellers: Sequence[Seller],
        buyers: Sequence[Buyer],
        *,
        # 价格区间设置
        price_bounds: tuple[float, float, float] = (0.1, 1.6, 0.05),
        delta: float = 0.18,
        # AF 设置
        af_mode: str = "gaussian",
        noise_sigma: float = 1.0,
        # PD 设置
        shapley_permutations: int = 256,
        lambda_penalty: float = float(np.log(2.0)),
        # 随机种子
        seed: Optional[int] = None,
        # 决策模式
        platform_mode: str = "agent",   # "agent" | "mwu"
        buyer_mode: str = "agent",       # "agent" | "truthful" | "shade" | "overbid"
        shade_factor: float = 0.65,      # 低报策略的折扣倍数
        overbid_factor: float = 1.4,     # 高报策略的膨胀倍数
    ):
        if len(sellers) == 0:
            raise ValueError("至少需要一个卖家。")
        if len(buyers) == 0:
            raise ValueError("至少需要一个买家。")

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

        # 内部状态（reset 时初始化）
        self._env: Optional[MarketplaceForDataEnv] = None
        self._round_idx: int = 0
        self._pending_price: Optional[float] = None   # 平台已选、等待买家报价的价格
        self._pending_buyer: Optional[Buyer] = None    # 当前等待决策的买家
        self._buyer_histories: Dict[str, List[Dict[str, Any]]] = {}
        self._done: bool = False

    # ------------------------------------------------------------------
    # 核心循环 API
    # ------------------------------------------------------------------

    def reset(self) -> Dict[str, Any]:
        """开始新 episode，返回第一个需要决策的 Agent 的观察。"""
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

        return self._advance_to_next_agent_decision()

    def step(self, role: str, action: Dict[str, Any]) -> Dict[str, Any]:
        """执行一个 Agent 决策，推进到下一个决策点。

        参数
        ----
        role : "平台" 或 "买家"
        action : {"价格": float} 或 {"报价": float}

        返回
        ----
        obs : 下一个 Agent 的观察字典；episode 结束时返回最后一轮的奖励摘要
        """
        if self._done:
            raise RuntimeError("Episode 已结束，请调用 reset()。")
        if role != self.current_role:
            raise ValueError(
                f"当前决策角色应为 {self.current_role}，但收到了 role={role}。"
            )

        if role == "平台":
            return self._step_platform(action)
        else:
            return self._step_buyer(action)

    @property
    def current_role(self) -> str:
        """当前轮到谁决策。"""
        if self._done:
            return "结束"
        if self._pending_price is None:
            return "平台"
        return "买家"

    @property
    def done(self) -> bool:
        """Episode 是否已结束。"""
        return self._done

    # ------------------------------------------------------------------
    # 观察获取（调试用，可在任意时刻调用）
    # ------------------------------------------------------------------

    def get_obs(self, role: str) -> Dict[str, Any]:
        """获取指定角色当前能看到的状态快照。"""
        if self._env is None:
            raise RuntimeError("请先调用 reset()。")
        if role == "平台":
            return _make_platform_obs(
                self._round_idx,
                self.total_rounds,
                self._env.price_state,
                len(self._env.sellers),
                self._env.history,
            )
        elif role == "买家":
            if self._pending_buyer is None:
                raise RuntimeError("尚无待决策买家——平台需先定价。")
            return _make_buyer_obs(
                self._round_idx,
                self.total_rounds,
                self._pending_buyer.mu,
                self._pending_price,  # type: ignore[arg-type]
                self._buyer_histories.get(self._pending_buyer.buyer_id, []),
            )
        else:
            raise ValueError(f"未知角色: {role}")

    # ------------------------------------------------------------------
    # 奖励与基准
    # ------------------------------------------------------------------

    def get_last_reward(self) -> Dict[str, Any]:
        """最近一轮结算后的奖励（各角色效用）。"""
        if self._env is None or not self._env.history:
            return {}
        h = self._env.history[-1]
        buyer_idx = min(self._round_idx - 1, len(self.buyers) - 1)
        return {
            "轮次": len(self._env.history),
            "买家ID": self.buyers[buyer_idx].buyer_id,
            "价格": round(float(h.price), 4),
            "报价": round(float(h.bid), 4),
            "买家效用": round(float(h.buyer_utility), 6),
            "平台效用": round(float(h.platform_utility), 6),
            "卖家效用": [round(float(u), 6) for u in h.seller_utilities],
            "平台收入": round(float(h.revenue), 6),
            "增益": round(float(h.gain), 4),
        }

    def transaction_history(self) -> List[Dict[str, Any]]:
        """返回每一轮真实交易结果。

        这份日志不区分该轮价格或报价来自规则还是 Agent，只记录最终进入市场机制
        结算的价格、报价和效用结果，因此适合用于跨模式对齐分析和画图。
        """
        if self._env is None:
            return []
        rows = []
        for idx, h in enumerate(self._env.history):
            buyer = self.buyers[idx] if idx < len(self.buyers) else None
            rows.append(
                {
                    "round_index": idx,
                    "buyer_id": buyer.buyer_id if buyer is not None else "",
                    "price": round(float(h.price), 4),
                    "bid": round(float(h.bid), 4),
                    "buyer_utility": round(float(h.buyer_utility), 6),
                    "platform_utility": round(float(h.platform_utility), 6),
                    "seller_utilities": [round(float(u), 6) for u in h.seller_utilities],
                    "platform_revenue": round(float(h.revenue), 6),
                    "gain": round(float(h.gain), 4),
                }
            )
        return rows

    def ground_truth(self) -> Dict[str, Any]:
        """理论均衡基准（假设所有买家诚实报价 b=mu，平台定价 p*）。"""
        if self._env is None:
            raise RuntimeError("请先调用 reset()。")
        gt = self._env.ground_truth(self.buyers)
        return {
            "最优价格_p_star": round(float(gt["price_star"]), 4),
            "最优总收入": round(float(gt["total_revenue_star"]), 4),
            "买家理论效用": [round(float(u), 6) for u in gt["buyer_utilities_star"]],
            "平台理论效用": round(float(gt["platform_utility_star"]), 6),
        }

    def episode_summary(self) -> Dict[str, Any]:
        """全 episode 累积效用汇总。"""
        if self._env is None:
            return {}
        u = self._env.utilities()
        return {
            "总轮次": len(self._env.history),
            "平台总效用": round(float(u["platform"]), 6),
            "买家总效用列表": [round(float(v), 6) for v in u["buyers"]],
            "买家平均效用": round(float(np.mean(u["buyers"])), 6),
            "卖家总效用列表": [round(float(v), 6) for v in u["sellers"]],
        }

    # ------------------------------------------------------------------
    # 内部：状态机推进逻辑
    # ------------------------------------------------------------------

    def _advance_to_next_agent_decision(self) -> Dict[str, Any]:
        """推进状态直到需要 Agent 决策（或 episode 结束）。

        如果当前角色的决策模式是规则驱动的（而非 agent），
        自动计算决策并跳过，直到遇到需要 Agent 的步骤。
        """
        while not self._done:
            # 所有买家都处理完毕
            if self._round_idx >= self.total_rounds:
                self._done = True
                return self.get_last_reward()

            # --- 需要平台决策 ---
            if self._pending_price is None:
                if self.platform_mode == "mwu":
                    # 规则模式：MWU 自动选价，跳过
                    self._pending_price = self._env.price_state.choose_price(
                        self._env.rng, deterministic=True
                    )
                    self._pending_buyer = self.buyers[self._round_idx]
                    # 继续到买家步骤
                else:
                    # Agent 模式：返回观察，等待外部决策
                    self._pending_buyer = self.buyers[self._round_idx]
                    return self.get_obs("平台")

            # --- 需要买家决策 ---
            if self._pending_price is not None:
                if self.buyer_mode != "agent":
                    # 规则模式：按策略自动报价
                    mu = self._pending_buyer.mu  # type: ignore[union-attr]
                    if self.buyer_mode == "truthful":
                        bid = mu
                    elif self.buyer_mode == "shade":
                        bid = mu * self.shade_factor
                    elif self.buyer_mode == "overbid":
                        bid = mu * self.overbid_factor
                    else:
                        raise ValueError(f"未知 buyer_mode: {self.buyer_mode}")
                    self._settle_round(bid)
                    # 结算完毕，继续下一轮
                else:
                    # Agent 模式：返回观察，等待外部决策
                    return self.get_obs("买家")

        return self.get_last_reward()

    def _step_platform(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """平台 Agent 提交价格决策。"""
        price = float(action["价格"])
        lower, upper, _ = self.price_bounds
        price = max(lower, min(upper, price))  # 裁剪到有效范围
        self._pending_price = price
        self._pending_buyer = self.buyers[self._round_idx]
        return self._advance_to_next_agent_decision()

    def _step_buyer(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """买家 Agent 提交报价决策。"""
        bid = float(action["报价"])
        bid = max(0.0, bid)  # 报价必须非负
        self._settle_round(bid)
        return self._advance_to_next_agent_decision()

    def _settle_round(self, bid: float) -> None:
        """执行 AF/RF/PD/PF 结算。"""
        buyer = self.buyers[self._round_idx]
        price = self._pending_price

        # 调用底层环境（external_price 跳过 MWU 定价）
        result = self._env.step(  # type: ignore[union-attr]
            buyer,
            bid=bid,
            external_price=price,
        )

        # 记录买家个人历史
        hist_entry = {
            "轮次": self._round_idx + 1,
            "估值": round(float(buyer.mu), 4),
            "价格": round(float(price), 4),   # type: ignore[arg-type]
            "报价": round(float(bid), 4),
            "效用": round(float(result.buyer_utility), 6),
            "增益": round(float(result.gain), 4),
        }
        self._buyer_histories[buyer.buyer_id].append(hist_entry)

        # 进入下一轮
        self._round_idx += 1
        self._pending_price = None
        self._pending_buyer = None
