"""Agent 基类 — 市场参与者的决策接口。

提供三种层次的 Agent 实现：
1. BaseAgent（抽象基类）：定义 act(obs) → action 接口
2. 规则 Agent：RuleBasedPlatformAgent / TruthfulBuyerAgent / ShadeBuyerAgent / OverbidBuyerAgent
3. LLMAgent：基于大语言模型的 Agent（需要 openai 包）

Agent 开发者只需：
- 继承 BaseAgent，实现 act(obs) 方法
- 或继承 LLMAgent，自定义 _format_obs() 和 _parse_response()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------

class BaseAgent(ABC):
    """所有 Agent 的抽象基类。

    子类必须实现 act(obs) 方法。
    环境在轮到该 Agent 决策时调用 act()。

    平台 Agent 应返回 {"价格": float}
    买家 Agent 应返回 {"报价": float}
    """

    @abstractmethod
    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """根据当前观察返回动作字典。"""
        ...

    def reset(self) -> None:
        """每个 episode 开始时调用。重写以清理 Agent 内部状态。"""
        pass


# ---------------------------------------------------------------------------
# 规则驱动的对照 Agent（不依赖 LLM，用于基线实验）
# ---------------------------------------------------------------------------

class RuleBasedPlatformAgent(BaseAgent):
    """遵循 MWU 规则的平台 Agent。

    直接选择 MWU 权重最大的候选价格，等价于论文原版 PF 的确定性模式。
    用于与 LLM 平台 Agent 做对照。
    """

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        return {"价格": obs["MWU推荐价格"]}


class TruthfulBuyerAgent(BaseAgent):
    """诚实报价买家 Agent：始终 b = mu。

    论文理论预测此为占优策略，用于 Agent 实验的基准对照。
    """

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        return {"报价": obs["我的估值_mu"]}


class ShadeBuyerAgent(BaseAgent):
    """低报策略买家：b = shade_factor * mu。

    用于验证"偏离诚实报价是否会降低效用"。
    """

    def __init__(self, shade_factor: float = 0.65):
        self.shade_factor = float(shade_factor)

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        return {"报价": obs["我的估值_mu"] * self.shade_factor}


class OverbidBuyerAgent(BaseAgent):
    """高报策略买家：b = overbid_factor * mu。

    用于验证"虚高报价是否会降低效用"。
    """

    def __init__(self, overbid_factor: float = 1.4):
        self.overbid_factor = float(overbid_factor)

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        return {"报价": obs["我的估值_mu"] * self.overbid_factor}


# ---------------------------------------------------------------------------
# LLM Agent 基类
# ---------------------------------------------------------------------------

class LLMAgent(BaseAgent):
    """使用大语言模型做决策的 Agent 基类。

    参数
    ----
    model : 模型名称，如 "gpt-4o-mini"
    system_prompt : 系统提示词（角色说明 + 机制规则）
    temperature : 采样温度（0.0 = 确定性输出）
    max_tokens : 最大生成 token 数
    api_key : API 密钥，为 None 时从环境变量 OPENAI_API_KEY 读取
    base_url : 自定义 API 地址（用于本地模型如 vLLM / Ollama）
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        system_prompt: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._api_key = api_key
        self._base_url = base_url
        self._client = None
        self.history: List[Dict[str, Any]] = []  # 完整对话日志

    def _get_client(self):
        """延迟初始化 OpenAI 客户端。"""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "LLMAgent 需要 openai 包。请运行: pip install openai"
                )
            kwargs = {}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """调用 LLM 获取决策。"""
        user_prompt = self._format_obs(obs)
        response = self._call_llm(user_prompt)
        self.history.append({"观察": obs, "LLM响应": response})
        return self._parse_response(response)

    def reset(self) -> None:
        """清空对话历史。"""
        self.history.clear()

    # ---- 子类可重写以下方法以自定义 prompt 和解析逻辑 ----

    def _format_obs(self, obs: Dict[str, Any]) -> str:
        """将观察字典转为 LLM 可读的文本。重写以自定义 prompt 格式。"""
        import json
        return json.dumps(obs, ensure_ascii=False, indent=2)

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """从 LLM 文本输出中解析出动作字典。重写以匹配自定义输出格式。"""
        import json
        import re

        # 尝试提取 JSON 对象
        json_match = re.search(r'\{[^{}]*\}', response)
        if json_match:
            return json.loads(json_match.group(0))
        # 兜底：提取第一个浮点数作为报价
        nums = re.findall(r'[\d.]+', response)
        if nums:
            return {"报价": float(nums[0])}
        raise ValueError(f"无法从 LLM 响应中解析动作: {response[:200]}")

    def _call_llm(self, user_prompt: str) -> str:
        """调用 LLM API。"""
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""
