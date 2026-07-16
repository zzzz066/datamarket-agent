"""数据市场数值环境 — 单市场、固定卖家、顺序买家的回合制模拟。

整个环境围绕论文 ADS 2019 的四个机制组件（AF/RF/PF/PD）构建：
- 平台维护 MWU 价格状态（PF），每轮选择一个价格
- 买家提交标量报价 b
- AF 根据 b vs p 的关系分配数据（可能加噪）
- RF 按 Myerson 规则计算支付
- PD 按 Shapley + 相似度惩罚分配收入给卖家

默认买家诚实报价 b = mu（即报真实估值）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ..mechanism import (
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


# ---------------------------------------------------------------------------
# 参与者数据结构
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Seller:
    """数据卖家。

    seller_id : 唯一标识
    feature : 特征向量 x_i ∈ R^d
    cost : 提供数据的固定成本
    """
    seller_id: str
    feature: np.ndarray
    cost: float = 0.0


@dataclass(frozen=True)
class Buyer:
    """数据买家。

    buyer_id : 唯一标识
    y : 目标变量（用于监督学习）
    mu : 买家对数据的真实估值
    bid : 预设报价（可选，为 None 时默认 b=mu）
    """
    buyer_id: str
    y: np.ndarray
    mu: float
    bid: Optional[float] = None


@dataclass
class StepResult:
    """单轮模拟的完整结果。

    包含价格、报价、分配特征、增益、支付、三方效用和收入分配详情。
    """
    price: float
    bid: float
    allocated_features: np.ndarray
    gain: float
    revenue: float
    buyer_utility: float
    seller_utilities: np.ndarray
    platform_utility: float
    payment_division: PaymentDivisionResult


# ---------------------------------------------------------------------------
# 主环境类
# ---------------------------------------------------------------------------

class MarketplaceForDataEnv:
    """单市场环境：固定卖家集合 + 顺序到达的买家。

    每轮流程：
    1. 平台选择价格 p（PF 的 choose_price）
    2. 买家提交报价 b
    3. AF：根据 b 和 p 的关系分配数据（可能加噪）
    4. RF：计算 Myerson 支付
    5. PD：Shapley + 相似度惩罚分配收入
    6. PF 更新：用本轮报价更新候选价格权重

    使用方式
    --------
    >>> price_state = PriceUpdateState.from_bounds(0.1, 1.5, epsilon=0.1)
    >>> env = MarketplaceForDataEnv(sellers, price_state=price_state, seed=42)
    >>> for buyer in buyers:
    ...     result = env.step(buyer)
    ...     print(result.buyer_utility)
    >>> print(env.utilities())
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
            raise ValueError("至少需要一个卖家。")
        self.sellers = list(sellers)
        self.price_state = price_state
        self.af_mode = af_mode
        self.noise_sigma = float(noise_sigma)
        self.shapley_permutations = int(shapley_permutations)
        self.lambda_penalty = float(lambda_penalty)
        self.normalize_pd = bool(normalize_pd)
        self.rng = np.random.default_rng(seed)
        self.t = 0                           # 当前轮次
        self.history: list[StepResult] = []  # 完整历史

    # ------------------------------------------------------------------
    # 便捷属性
    # ------------------------------------------------------------------

    @property
    def X(self) -> np.ndarray:
        """所有卖家的特征矩阵 (M × d)。"""
        return np.vstack([
            np.asarray(s.feature, dtype=float).reshape(1, -1) for s in self.sellers
        ])

    @property
    def seller_costs(self) -> np.ndarray:
        """所有卖家的成本数组。"""
        return np.array([float(s.cost) for s in self.sellers], dtype=float)

    # ------------------------------------------------------------------
    # 核心 step
    # ------------------------------------------------------------------

    def step(
        self,
        buyer: Buyer,
        *,
        bid: Optional[float] = None,
        deterministic_price: bool = False,
        external_price: Optional[float] = None,
    ) -> StepResult:
        """执行一轮完整的市场交互。

        参数
        ----
        buyer : 当前买家
        bid : 外部指定的报价（为 None 时使用 buyer.bid 或 buyer.mu）
        deterministic_price : True 时 PF 选权重最大候选而非随机采样
        external_price : 外部指定的价格（Agent 模式使用）；
                         为 None 时由 PF/MWU 自动选择

        返回
        ----
        StepResult : 包含本轮所有结算信息
        """
        # 确定实际报价
        actual_bid = float(
            buyer.mu if bid is None and buyer.bid is None else (buyer.bid if bid is None else bid)
        )

        # 确定当前价格：外部指定 > MWU 自动选择
        if external_price is not None:
            price = float(external_price)
        else:
            price = self.price_state.choose_price(self.rng, deterministic=deterministic_price)

        # AF：分配数据
        allocated = data_allocation_function(
            price, actual_bid, self.X,
            mode=self.af_mode, noise_sigma=self.noise_sigma, rng=self.rng,
        )

        # 计算信息增益
        y = np.asarray(buyer.y, dtype=float).reshape(-1)
        yhat = fit_linear_prediction(allocated, y)
        gain = normalized_rmse_gain(y, yhat)

        # RF：Myerson 支付（使用实际增益曲线做数值积分）
        def gain_at_bid(z: float) -> float:
            q = 1.0 if price <= 0 else min(max(float(z) / float(price), 0.0), 1.0)
            return float(q * gain)

        revenue = myerson_revenue_function(price, actual_bid, gain_at_bid=gain_at_bid)

        # PD：收入分配
        value_fn = lambda subset: subset_linear_gain(y, allocated, subset)
        division = robust_payment_division(
            revenue, allocated, value_fn,
            num_permutations=self.shapley_permutations,
            lambda_penalty=self.lambda_penalty,
            normalize=self.normalize_pd,
            rng=self.rng,
        )

        # 计算三方效用
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

        # PF 更新：根据本轮报价调整权重
        self.price_state.update(actual_bid)
        self.history.append(result)
        self.t += 1
        return result

    # ------------------------------------------------------------------
    # 汇总与基准
    # ------------------------------------------------------------------

    def utilities(self) -> dict:
        """返回全 episode 的累积效用。

        返回
        ----
        {"platform": float, "buyers": ndarray, "sellers": ndarray}
        """
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
        """计算理论均衡基准（假设所有买家诚实报价 b=mu，平台使用最优固定价格）。

        用于评估 Agent 策略与理论最优的差距。
        """
        mus = [float(b.mu) for b in buyers]
        bounds = (
            float(np.min(self.price_state.candidates)),
            float(np.max(self.price_state.candidates)),
        )
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
