"""Agent 模块 — 市场参与者的决策接口。"""

from .base import (
    ActionDict,
    ActionParser,
    AgentActionParseError,
    BaseAgent,
    ChatClient,
    LLMAgent,
    OpenAICompatibleClient,
    OverbidBuyerAgent,
    RuleBasedPlatformAgent,
    ShadeBuyerAgent,
    TruthfulBuyerAgent,
    default_system_prompt,
    fallback_action,
    infer_role_from_obs,
)

__all__ = [
    "ActionDict",
    "ActionParser",
    "AgentActionParseError",
    "BaseAgent",
    "ChatClient",
    "LLMAgent",
    "OpenAICompatibleClient",
    "OverbidBuyerAgent",
    "RuleBasedPlatformAgent",
    "ShadeBuyerAgent",
    "TruthfulBuyerAgent",
    "default_system_prompt",
    "fallback_action",
    "infer_role_from_obs",
]
