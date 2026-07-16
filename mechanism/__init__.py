"""ADS 2019 数据市场核心机制模块。

实现论文 Section 4 的四个机制组件：
- AF: 数据分配函数（报价低于定价时加噪退化）
- RF: Myerson 支付规则（单参数买家的最优支付）
- PF: 乘法权重价格更新（平台无遗憾定价）
- PD: 稳健收入分配（Shapley 值 + 余弦相似度指数惩罚）
"""

from .core import (
    PaymentDivisionResult,
    PriceUpdateState,
    closed_form_ground_truth,
    cosine_similarity_matrix,
    data_allocation_function,
    exact_shapley_values,
    myerson_revenue_function,
    normalized_rmse_gain,
    robust_payment_division,
    shapley_approximation,
    fit_linear_prediction,
    subset_linear_gain,
)

__all__ = [
    "PaymentDivisionResult",
    "PriceUpdateState",
    "closed_form_ground_truth",
    "cosine_similarity_matrix",
    "data_allocation_function",
    "exact_shapley_values",
    "myerson_revenue_function",
    "normalized_rmse_gain",
    "robust_payment_division",
    "shapley_approximation",
    "fit_linear_prediction",
    "subset_linear_gain",
]
