"""数据市场 Agent 模拟平台。

基于 Agarwal, Dahleh, and Sarkar (2019)
"A Marketplace for Data: An Algorithmic Solution" 的数值环境与 Agent 系统。

目录结构
--------
mechanism/   核心机制（AF/RF/PF/PD）
env/         环境层（规则环境 + Agent 封装）
agents/      Agent 决策接口
experiments/ 实验脚本
"""

# 机制层
from .mechanism.core import (
    PaymentDivisionResult,
    PriceUpdateState,
    closed_form_ground_truth,
    cosine_similarity_matrix,
    data_allocation_function,
    exact_shapley_values,
    fit_linear_prediction,
    myerson_revenue_function,
    normalized_rmse_gain,
    robust_payment_division,
    shapley_approximation,
    subset_linear_gain,
)

# 环境层
from .env.base import Buyer, MarketplaceForDataEnv, Seller, StepResult
from .env.wrapper import AgentMarketEnv

# Agent 接口
from .agents.base import (
    ActionParser,
    AgentActionParseError,
    BaseAgent,
    LLMAgent,
    OpenAICompatibleClient,
    OverbidBuyerAgent,
    RuleBasedPlatformAgent,
    ShadeBuyerAgent,
    TruthfulBuyerAgent,
)

__all__ = [
    # 机制
    "PaymentDivisionResult",
    "PriceUpdateState",
    "closed_form_ground_truth",
    "cosine_similarity_matrix",
    "data_allocation_function",
    "exact_shapley_values",
    "fit_linear_prediction",
    "myerson_revenue_function",
    "normalized_rmse_gain",
    "robust_payment_division",
    "shapley_approximation",
    "subset_linear_gain",
    # 环境
    "Buyer",
    "MarketplaceForDataEnv",
    "Seller",
    "StepResult",
    "AgentMarketEnv",
    # Agent
    "ActionParser",
    "AgentActionParseError",
    "BaseAgent",
    "LLMAgent",
    "OpenAICompatibleClient",
    "OverbidBuyerAgent",
    "RuleBasedPlatformAgent",
    "ShadeBuyerAgent",
    "TruthfulBuyerAgent",
]
