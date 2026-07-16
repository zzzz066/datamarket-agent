"""ADS 2019 "A Marketplace for Data: An Algorithmic Solution" 核心机制实现。

实现论文 Section 4 的构造：

* AF — 数据分配函数：买家报价低于平台定价时，向数据注入噪声以降低质量。
* RF — Myerson 支付函数：针对单参数买家的最优支付规则，保证诚实报价占优。
* PF — 价格更新：在离散候选价格网格上做乘法权重更新，平台实现无遗憾定价。
* PD — 稳健支付分配：Shapley 边际贡献近似 + 余弦相似度指数惩罚，抑制数据复制。

闭式基准解使用标准单调质量曲线 q_p(b) = min(b/p, 1)，这是论文中
"质量随报价递增"假设的确定性对应版本，可导出 RF 的解析形式和最优固定价格公式。
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Callable, Iterable, Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# 类型别名
# ---------------------------------------------------------------------------

Array = np.ndarray
GainFunction = Callable[[Array, Array], float]
ValueFunction = Callable[[Sequence[int]], float]


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _as_2d_features(x: Array) -> Array:
    """将输入转换为二维特征矩阵，每行一个卖家特征。"""
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"X 必须是一维或二维数组，实际 shape={arr.shape}。")
    return arr


def _quality_ratio(price: float, bid: float) -> float:
    """质量曲线 h_p(b) = min(b/p, 1)，截断到 [0, 1]。"""
    price = float(price)
    bid = float(bid)
    if price <= 0:
        return 1.0
    return float(np.clip(bid / price, 0.0, 1.0))


# ---------------------------------------------------------------------------
# 信息增益：用分配到的数据做预测，衡量数据质量
# ---------------------------------------------------------------------------

def normalized_rmse_gain(y_true: Array, y_pred: Array, *, eps: float = 1e-12) -> float:
    """归一化 RMSE 增益，取值 [0, 1]。

    计算公式：gain = 1 - RMSE(y_true, y_pred) / (y_max - y_min)
    值越接近 1 表示预测越准确，即数据质量越高。
    """
    y = np.asarray(y_true, dtype=float).reshape(-1)
    yhat = np.asarray(y_pred, dtype=float).reshape(-1)
    if y.shape != yhat.shape:
        raise ValueError(f"y_true 和 y_pred 的 shape 必须相同，实际 {y.shape} vs {yhat.shape}。")
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
    scale = float(np.max(y) - np.min(y))
    if scale <= eps:
        scale = float(np.std(y))
    if scale <= eps:
        return 1.0 if rmse <= eps else 0.0
    return float(np.clip(1.0 - rmse / scale, 0.0, 1.0))


def fit_linear_prediction(x_features: Array, y: Array) -> Array:
    """在选定的特征子集上拟合 OLS 线性模型，返回样本内预测值。

    用于衡量"如果买家只获得部分卖家的数据，预测能力有多少"。
    """
    X = _as_2d_features(x_features).T
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    if X.shape[0] != y_arr.shape[0]:
        raise ValueError(f"特征长度 {X.shape[0]} 与 y 长度 {y_arr.shape[0]} 不匹配。")
    # 设计矩阵：加一列截距项
    design = np.column_stack([np.ones(X.shape[0]), X])
    coef, *_ = np.linalg.lstsq(design, y_arr, rcond=None)
    return design @ coef


def subset_linear_gain(
    y: Array,
    x_features: Array,
    subset: Sequence[int],
    gain_fn: GainFunction = normalized_rmse_gain,
) -> float:
    """Shapley 计算用的价值函数 G(Y, M(X_S))。

    对指定的卖家子集 S，用其数据做 OLS 预测，返回预测增益。
    空集时返回均值基准线的预测增益。
    """
    if len(subset) == 0:
        baseline = np.full_like(np.asarray(y, dtype=float), float(np.mean(y)), dtype=float)
        return gain_fn(y, baseline)
    X = _as_2d_features(x_features)
    yhat = fit_linear_prediction(X[list(subset)], y)
    return gain_fn(y, yhat)


# ---------------------------------------------------------------------------
# AF: 数据分配函数（Algorithm 1 的分配步骤）
# ---------------------------------------------------------------------------

def data_allocation_function(
    price: float,
    bid: float,
    x_features: Array,
    *,
    mode: str = "gaussian",
    noise_sigma: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> Array:
    """AF(p, b; X)：报价高于定价则分配完整数据，否则退化数据。

    对于连续特征，``mode="gaussian"`` 实现论文 Example 4.1：
        X_tilde = X + max(0, p - b) * N(0, sigma^2)

    对于二值特征，``mode="masking"`` 实现 Example 4.2：
        以 min(b/p, 1) 的伯努利概率保留每个维度。

    直观理解：买家出价不够高时，平台给的数据被加了噪声，
    差值越大噪声越强，数据预测能力越低。
    """
    X = np.asarray(x_features, dtype=float)
    shortfall = max(0.0, float(price) - float(bid))
    if shortfall <= 0.0:
        return X.copy()  # 出价够高，给完整数据
    if rng is None:
        rng = np.random.default_rng()
    if mode == "gaussian":
        # 高斯噪声，噪声标准差 = 价格-报价差额
        return X + shortfall * rng.normal(0.0, float(noise_sigma), size=X.shape)
    if mode == "masking":
        # 伯努利掩码，保留概率 = b/p
        keep_prob = _quality_ratio(price, bid)
        return X * rng.binomial(1, keep_prob, size=X.shape)
    raise ValueError(f"不支持的 AF 模式 {mode!r}，预期 'gaussian' 或 'masking'。")


# ---------------------------------------------------------------------------
# RF: Myerson 支付函数
# ---------------------------------------------------------------------------

def closed_form_quality_gain(price: float, bid: float) -> float:
    """解析质量曲线 h_p(b) = min(b/p, 1)。"""
    return _quality_ratio(price, bid)


def closed_form_myerson_revenue(price: float, bid: float) -> float:
    """h_p(b)=min(b/p,1) 下的 RF 闭式解。

    当 b < p 时：RF = b^2 / (2p)
    当 b >= p 时：RF = p / 2
    """
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
    """RF(p, b, Y)：Myerson 支付 b*h(b) - ∫_0^b h(z) dz。

    这是单参数拍卖的标准支付公式，保证买家诚实报价是占优策略。
    如果 ``gain_at_bid`` 为 None，使用解析质量曲线 h_p(b)=min(b/p, 1)；
    如果传入实际 ML 管道的增益函数，则用数值积分计算支付。

    参数
    ----
    price : 平台当前价格
    bid : 买家报价
    gain_at_bid : 可选，以报价为输入、输出增益的函数（用于数值积分）
    integration_steps : 数值积分的离散步数
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


# ---------------------------------------------------------------------------
# PF: 价格更新状态（Algorithm 1 的 MWU 部分）
# ---------------------------------------------------------------------------

@dataclass
class PriceUpdateState:
    """PF 状态：在离散候选价格网格上的乘法权重更新。

    论文 Algorithm 1 的核心：平台维护一组候选价格及其权重。
    每轮根据各候选价格"如果被选用能产生多少收入"来更新权重，
    收入越高的候选价格权重增长越快，平台下一轮更可能选它。
    多轮后权重自然收敛到最优固定价格附近，实现无遗憾（no-regret）。
    """

    candidates: Array       # 候选价格数组，如 [0.1, 0.15, ..., 1.6]
    weights: Array           # 各候选价格的 MWU 权重（非负）
    delta: float = 0.1      # MWU 学习率
    b_max: Optional[float] = None  # 收入归一化的上界

    @classmethod
    def from_bounds(
        cls,
        lower: float,
        upper: float,
        *,
        epsilon: float,
        delta: float = 0.1,
    ) -> "PriceUpdateState":
        """从价格范围和步长创建候选网格。

        参数
        ----
        lower : 价格下界
        upper : 价格上界
        epsilon : 候选价格间距（步长）
        delta : MWU 学习率
        """
        if upper <= lower:
            raise ValueError("upper 必须大于 lower。")
        if epsilon <= 0:
            raise ValueError("epsilon 必须为正数。")
        candidates = np.arange(float(lower), float(upper) + 0.5 * float(epsilon), float(epsilon))
        return cls(
            candidates=candidates,
            weights=np.ones_like(candidates, dtype=float),
            delta=delta,
            b_max=float(upper),
        )

    @property
    def probabilities(self) -> Array:
        """返回各候选价格被选中的概率分布（权重归一化）。"""
        total = float(np.sum(self.weights))
        if total <= 0:
            return np.ones_like(self.weights) / len(self.weights)
        return self.weights / total

    def choose_price(
        self,
        rng: Optional[np.random.Generator] = None,
        *,
        deterministic: bool = False,
    ) -> float:
        """从候选价格中选择一个。

        参数
        ----
        deterministic : 若为 True，选权重最大的候选（argmax）；
                       若为 False，按权重概率随机采样（论文原版 Algorithm 1）。
        """
        if deterministic:
            return float(self.candidates[int(np.argmax(self.probabilities))])
        if rng is None:
            rng = np.random.default_rng()
        idx = int(rng.choice(len(self.candidates), p=self.probabilities))
        return float(self.candidates[idx])

    def update(
        self,
        bid: float,
        revenue_fn: Callable[[float, float], float] = closed_form_myerson_revenue,
    ) -> None:
        """权重乘法更新：每个候选价格的权重 *= (1 + delta * 归一化收入)。

        参数
        ----
        bid : 本轮买家的报价
        revenue_fn : 计算给定价格和报价下平台收入的函数
        """
        normalizer = float(self.b_max if self.b_max is not None else np.max(self.candidates))
        normalizer = max(normalizer, 1e-12)
        gains = np.array(
            [revenue_fn(float(c), float(bid)) / normalizer for c in self.candidates],
            dtype=float,
        )
        gains = np.clip(gains, 0.0, 1.0)
        self.weights *= (1.0 + float(self.delta) * gains)


# ---------------------------------------------------------------------------
# PD: 稳健支付分配（Algorithm 2 & 3）
# ---------------------------------------------------------------------------

def exact_shapley_values(num_features: int, value_fn: ValueFunction) -> Array:
    """穷举所有排列计算精确 Shapley 值。

    仅适用于特征数很少的情况（M ≤ 8），复杂度 O(M! × M × 2^M)。
    """
    M = int(num_features)
    if M < 0:
        raise ValueError("num_features 必须非负。")
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
    """Algorithm 2：随机排列采样近似 Shapley 值。

    从 M! 种排列中随机采样 num_permutations 种，
    用样本均值近似边际贡献的期望值。
    """
    M = int(num_features)
    if M < 0:
        raise ValueError("num_features 必须非负。")
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
    """计算特征矩阵的余弦相似度矩阵（取绝对值，缩放到 [0, 1]）。

    两个卖家特征的余弦相似度越高，说明它们的数据越"像"——
    高度相似意味着可能存在复制行为。
    """
    X = _as_2d_features(x_features)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    denom = np.maximum(norms @ norms.T, eps)
    sim = np.abs((X @ X.T) / denom)
    np.fill_diagonal(sim, 1.0)  # 自己和自己相似度为 1
    return np.clip(sim, 0.0, 1.0)


@dataclass
class PaymentDivisionResult:
    """PD 分配结果的数据结构。

    shapley : 各卖家的（近似）Shapley 值
    penalties : 各卖家的相似度惩罚系数 exp(-lambda * sum_sim)
    shares : 最终份额 = Shapley * penalty
    seller_payments : 卖家实际收入 = revenue * shares
    retained_by_platform : 平台留存 = revenue - sum(seller_payments)
    """
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
    """Algorithm 3：Shapley 值 + 指数相似度惩罚的稳健收入分配。

    核心思想：
    1. 用 Shapley 值衡量每个卖家数据对预测任务的边际贡献
    2. 用余弦相似度惩罚与其他人高度相似的卖家（抑制复制）
    3. share_i = Shapley_i * exp(-lambda * Σ_j SM(X_i, X_j))

    参数
    ----
    revenue : 平台从买家收到的总支付
    x_features : 卖家特征矩阵 (M × d)
    value_fn : 价值函数 G(Y, M(X_S))
    num_permutations : Shapley 近似使用的随机排列数
    lambda_penalty : 相似度惩罚强度（越大惩罚越重）
    exact : 是否使用精确 Shapley（仅适用于 M ≤ 8）
    normalize : 是否归一化使分配总和等于 revenue（论文默认不归一化）
    """
    X = _as_2d_features(x_features)
    M = X.shape[0]

    # Step 1: Shapley 值
    shapley = (
        exact_shapley_values(M, value_fn)
        if exact
        else shapley_approximation(M, value_fn, num_permutations=num_permutations, rng=rng)
    )

    # Step 2: 余弦相似度矩阵
    if similarity_matrix is None:
        similarity_matrix = cosine_similarity_matrix(X)
    sim = np.asarray(similarity_matrix, dtype=float)
    if sim.shape != (M, M):
        raise ValueError(f"similarity_matrix 的 shape 必须为 {(M, M)}，实际 {sim.shape}。")

    # Step 3: 相似度惩罚（排除自身）
    cumulative_similarity = np.sum(sim, axis=1) - np.diag(sim)
    penalties = np.exp(-float(lambda_penalty) * cumulative_similarity)

    # Step 4: 最终份额 = Shapley * penalty
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


# ---------------------------------------------------------------------------
# 闭式 Ground Truth：理论上最优的固定价格及对应效用
# ---------------------------------------------------------------------------

def _candidate_interval_points(mu_values: Array, lower: float, upper: float) -> Iterable[float]:
    """生成闭式最优价格的可能候选点（区间边界 + 驻点）。"""
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


def optimal_fixed_price_closed_form(
    mu_values: Sequence[float],
    price_bounds: tuple[float, float],
) -> tuple[float, float]:
    """闭式计算最优固定价格 p* = argmax_p Σ_n RF(p, mu_n)。

    在所有候选价格点中穷举，找到使总 Myerson 收入最大化的价格。
    这是 PF 的"完美 hindsight"对照基准。
    """
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
    revenues = np.array(
        [np.sum([closed_form_myerson_revenue(p, m) for m in mu]) for p in candidates]
    )
    best = int(np.argmax(revenues))
    return float(candidates[best]), float(revenues[best])


def closed_form_ground_truth(
    mu_values: Sequence[float],
    *,
    price_bounds: tuple[float, float],
    seller_costs: Optional[Sequence[float]] = None,
    payment_shares: Optional[Sequence[float]] = None,
) -> dict:
    """简化数值模型的理论均衡基准。

    假设：
    - 所有买家诚实报价 b_n* = mu_n
    - 平台使用最优固定价格 p*
    - 质量曲线 q_p(b) = min(b/p, 1)

    返回内容包括最优价格、各买家效用、卖家效用、平台效用等。
    用于评估 Agent 策略偏离理论最优的程度。
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
            raise ValueError("seller_costs 和 payment_shares 的长度必须相同。")

    seller_payments = total_revenue * shares
    seller_utilities = (
        seller_payments - seller_cost_arr if shares.size else np.zeros(0, dtype=float)
    )
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
