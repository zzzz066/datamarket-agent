"""Agent 接口定义。

Agent 开发者只需继承 BaseAgent，实现 act(obs) 方法。
环境在轮到该 Agent 决策时调用 act()：
- 平台 Agent 应返回 {"价格": float}
- 买家 Agent 应返回 {"报价": float}
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseAgent(ABC):
    """所有 Agent 的抽象基类。

    子类必须实现 act(obs) 方法。
    """

    @abstractmethod
    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """根据当前观察返回动作字典。"""
        ...

    def reset(self) -> None:
        """每个 episode 开始时调用。重写以清理 Agent 内部状态。"""
        pass
