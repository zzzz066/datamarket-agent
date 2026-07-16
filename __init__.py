"""数据市场 Agent 模拟平台。

基于 Agarwal, Dahleh, and Sarkar (2019)
"A Marketplace for Data: An Algorithmic Solution" 的数值环境与 Agent 系统。

目录结构
--------
mechanism/   核心机制（AF/RF/PF/PD）
env/         环境层（规则环境 + Agent 封装）
agents/      Agent 决策层（规则基线 + LLM Agent 基类）
experiments/ 实验脚本与评估指标
"""

# 机制层
from .mechanism.core import (
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

# 环境层
from .env.base import Buyer, MarketplaceForDataEnv, Seller, StepResult
from .env.wrapper import AgentMarketEnv

# Agent 层
from .agents.base import (
    BaseAgent,
    LLMAgent,
    RuleBasedPlatformAgent,
    TruthfulBuyerAgent,
    ShadeBuyerAgent,
    OverbidBuyerAgent,
)

__all__ = [
    # 机制
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
    # 环境
    "Buyer",
    "MarketplaceForDataEnv",
    "Seller",
    "StepResult",
    "AgentMarketEnv",
    # Agent
    "BaseAgent",
    "LLMAgent",
    "RuleBasedPlatformAgent",
    "TruthfulBuyerAgent",
    "ShadeBuyerAgent",
    "OverbidBuyerAgent",
]
