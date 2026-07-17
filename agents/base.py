"""Agent interfaces and LLM-backed decision helpers.

The LLM layer follows a role-aware, mechanism-guided design: the prompt states
what the platform or buyer can observe, asks the model to reflect briefly on the
market mechanism, and requires a machine-parseable JSON action.  The parser is
intentionally tolerant because real LLMs often wrap JSON in Markdown fences or
prepend a short explanation despite instructions.
"""

from __future__ import annotations

import ast
import json
import os
import re
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence


ActionDict = Dict[str, Any]
Message = Dict[str, str]


class BaseAgent(ABC):
    """所有 Agent 的抽象基类。

    子类必须实现 ``act(obs)``，返回环境可执行的动作字典：
    平台 Agent 返回 ``{"价格": float}``，买家 Agent 返回 ``{"报价": float}``。
    """

    @abstractmethod
    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """根据当前观察返回动作字典。"""
        ...

    def reset(self) -> None:
        """每个 episode 开始时调用。重写以清理 Agent 内部状态。"""
        pass


class RuleBasedPlatformAgent(BaseAgent):
    """使用环境给出的 MWU 推荐价格作为平台基线。"""

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        return {"价格": float(obs["MWU推荐价格"])}


@dataclass
class TruthfulBuyerAgent(BaseAgent):
    """诚实买家基线：报价等于自身真实估值 ``mu``。"""

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        return {"报价": float(obs["我的估值_mu"])}


@dataclass
class ShadeBuyerAgent(BaseAgent):
    """策略性低报价买家，用于检验动态压价效应。"""

    factor: float = 0.65

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        return {"报价": max(0.0, float(obs["我的估值_mu"]) * float(self.factor))}


@dataclass
class OverbidBuyerAgent(BaseAgent):
    """高报价买家基线，用于和诚实/低报策略对照。"""

    factor: float = 1.4

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        return {"报价": max(0.0, float(obs["我的估值_mu"]) * float(self.factor))}


class AgentActionParseError(ValueError):
    """Raised when an LLM response cannot be converted into an executable action."""


class ChatClient(Protocol):
    """Minimal chat-completion client protocol used by ``LLMAgent``."""

    def complete(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float = 0.2,
        response_format: Optional[Mapping[str, str]] = None,
    ) -> str:
        """Return the assistant message content."""
        ...


@dataclass
class OpenAICompatibleClient:
    """Tiny OpenAI-compatible chat completions client.

    It intentionally uses the Python standard library, so the project does not
    require the ``openai`` package.  Any service exposing ``/chat/completions``
    can be used by setting ``OPENAI_API_KEY`` and optionally ``OPENAI_BASE_URL``.
    """

    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: float = 60.0
    extra_headers: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.api_key = self.api_key if self.api_key is not None else os.getenv("OPENAI_API_KEY")
        self.base_url = (self.base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        if not self.api_key:
            raise RuntimeError("未找到 OPENAI_API_KEY，无法调用 LLM 接口。")

    def complete(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float = 0.2,
        response_format: Optional[Mapping[str, str]] = None,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            "temperature": float(temperature),
        }
        if response_format is not None:
            payload["response_format"] = dict(response_format)
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **dict(self.extra_headers),
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM request failed: {exc.reason}") from exc
        result = json.loads(body)
        try:
            content = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected LLM response: {result!r}") from exc
        if not isinstance(content, str):
            raise RuntimeError(f"Unexpected LLM content: {content!r}")
        return content


@dataclass
class ActionParser:
    """Parse and validate JSON-like LLM action outputs.

    Supported response shapes include:
    - ``{"价格": 0.8}`` or ``{"报价": 1.2}``
    - fenced JSON blocks
    - explanatory text followed by a JSON object
    - nested objects such as ``{"action": {"bid": 0.7}}``
    - English aliases ``price/p`` and ``bid/b``
    """

    platform_aliases: Sequence[str] = ("价格", "price", "p", "平台价格")
    buyer_aliases: Sequence[str] = ("报价", "bid", "b", "出价")
    action_containers: Sequence[str] = ("action", "动作", "decision", "决策")
    clamp_to_obs: bool = True

    def parse(self, raw: Any, *, role: Optional[str] = None, obs: Optional[Mapping[str, Any]] = None) -> ActionDict:
        inferred_role = role or infer_role_from_obs(obs)
        if inferred_role not in {"平台", "买家"}:
            raise AgentActionParseError(f"无法判断动作角色: {inferred_role!r}")
        payload = self._coerce_payload(raw)
        payload = self._unwrap_action_container(payload)
        if inferred_role == "平台":
            value = self._extract_number(payload, self.platform_aliases, target="价格")
            value = self._sanitize_platform_price(value, obs)
            return {"价格": value}
        value = self._extract_number(payload, self.buyer_aliases, target="报价")
        return {"报价": max(0.0, float(value))}

    def _coerce_payload(self, raw: Any) -> Mapping[str, Any]:
        if isinstance(raw, Mapping):
            return raw
        if not isinstance(raw, str):
            raise AgentActionParseError(f"动作输出必须是 dict 或 str，实际为 {type(raw).__name__}")
        candidates = list(self._json_candidates(raw))
        errors: List[str] = []
        for candidate in candidates:
            for loader in (json.loads, ast.literal_eval):
                try:
                    loaded = loader(candidate)
                except Exception as exc:  # noqa: BLE001 - collect parser diagnostics
                    errors.append(str(exc))
                    continue
                if isinstance(loaded, Mapping):
                    return loaded
                errors.append(f"JSON 顶层不是对象: {type(loaded).__name__}")
        raise AgentActionParseError("无法从 LLM 输出中解析动作 JSON。" + (" 最近错误: " + errors[-1] if errors else ""))

    def _json_candidates(self, text: str) -> Iterable[str]:
        stripped = text.strip()
        if stripped:
            yield stripped
        for match in re.finditer(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL):
            block = match.group(1).strip()
            if block:
                yield block
        for obj in _balanced_json_objects(text):
            yield obj

    def _unwrap_action_container(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        current: Mapping[str, Any] = payload
        for _ in range(3):
            lowered = {str(k).lower(): k for k in current.keys()}
            found = None
            for name in self.action_containers:
                key = lowered.get(name.lower())
                if key is not None:
                    found = current[key]
                    break
            if isinstance(found, Mapping):
                current = found
                continue
            return current
        return current

    def _extract_number(self, payload: Mapping[str, Any], aliases: Sequence[str], *, target: str) -> float:
        lowered = {str(k).lower(): k for k in payload.keys()}
        for alias in aliases:
            key = lowered.get(alias.lower())
            if key is not None:
                return _to_float(payload[key], field=target)
        if len(payload) == 1:
            only_value = next(iter(payload.values()))
            return _to_float(only_value, field=target)
        raise AgentActionParseError(f"JSON 中缺少 {target} 字段，可用字段: {list(payload.keys())}")

    def _sanitize_platform_price(self, value: float, obs: Optional[Mapping[str, Any]]) -> float:
        price = float(value)
        if not self.clamp_to_obs or obs is None:
            return price
        candidates = obs.get("候选价格") if isinstance(obs, Mapping) else None
        if isinstance(candidates, Sequence) and not isinstance(candidates, (str, bytes)) and candidates:
            numeric = [float(x) for x in candidates]
            return min(max(price, min(numeric)), max(numeric))
        return price


@dataclass
class LLMAgent(BaseAgent):
    """OpenAI-compatible LLM Agent for platform or buyer decisions.

    The default design combines three ideas useful for data-market ABM:
    role-aware prompts, short memory over recent decisions, and mechanism-guided
    reflection before emitting a strict JSON action.  The final returned value is
    always normalized to the environment contract: ``{"价格": x}`` or ``{"报价": x}``.
    """

    model: str = "gpt-4o-mini"
    role: Optional[str] = None
    system_prompt: Optional[str] = None
    prompt_profile: str = "balanced"
    client: Optional[ChatClient] = None
    parser: ActionParser = field(default_factory=ActionParser)
    temperature: float = 0.2
    max_retries: int = 2
    retry_sleep: float = 0.2
    memory_size: int = 8
    fallback_on_error: bool = True
    response_format: Optional[Mapping[str, str]] = field(default_factory=lambda: {"type": "json_object"})
    _memory: List[Dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = OpenAICompatibleClient()

    def reset(self) -> None:
        self._memory.clear()

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        role = self.role or infer_role_from_obs(obs)
        messages = self._build_messages(obs, role)
        last_error: Optional[Exception] = None
        for attempt in range(int(self.max_retries) + 1):
            try:
                raw = self.client.complete(
                    messages,
                    model=self.model,
                    temperature=self.temperature,
                    response_format=self.response_format,
                )
                action = self.parser.parse(raw, role=role, obs=obs)
                self._remember(obs, action, raw)
                return action
            except Exception as exc:  # noqa: BLE001 - retry on parser/client failures
                last_error = exc
                messages.append({"role": "assistant", "content": raw if "raw" in locals() else ""})
                messages.append({"role": "user", "content": self._repair_prompt(role, exc)})
                if attempt < int(self.max_retries) and self.retry_sleep > 0:
                    time.sleep(float(self.retry_sleep))
        if self.fallback_on_error:
            action = fallback_action(obs, role=role)
            self._remember(obs, action, f"FALLBACK_AFTER_ERROR: {last_error}")
            return action
        raise AgentActionParseError(f"LLM Agent 生成动作失败: {last_error}")

    def _build_messages(self, obs: Mapping[str, Any], role: str) -> List[Message]:
        system = self.system_prompt or default_system_prompt(role, profile=self.prompt_profile)
        user = {
            "role": "user",
            "content": default_user_prompt(
                role,
                obs,
                self._memory[-self.memory_size:],
                profile=self.prompt_profile,
            ),
        }
        return [{"role": "system", "content": system}, user]

    def _repair_prompt(self, role: str, error: Exception) -> str:
        key = "价格" if role == "平台" else "报价"
        example = {"reasoning": "根据机制与历史选择一个稳健动作", key: 0.8}
        return (
            f"上一次输出无法解析: {error}。请只返回一个 JSON 对象，不要 Markdown。"
            f"格式示例: {json.dumps(example, ensure_ascii=False)}"
        )

    def _remember(self, obs: Mapping[str, Any], action: Mapping[str, Any], raw: str) -> None:
        entry = {
            "轮次": obs.get("当前轮次"),
            "action": dict(action),
            "raw": raw[:500],
        }
        self._memory.append(entry)
        if len(self._memory) > self.memory_size:
            del self._memory[: len(self._memory) - self.memory_size]


def default_system_prompt(role: str, *, profile: str = "balanced") -> str:
    """Return the role-aware system prompt used by ``LLMAgent``.

    ``profile`` is an experiment knob rather than a parser requirement.  It lets
    the same market environment compare truthful, strategic, welfare-aware,
    risk-averse, revenue-maximizing, and fairness-aware LLM agents.
    """
    schema = json.dumps(output_schema_for_role(role), ensure_ascii=False)
    mechanism = (
        "市场机制: 平台先选择价格 p，买家随后提交报价 b。"
        "若 b >= p，买家通常获得完整数据；若 b < p，数据会被加噪或遮蔽，质量下降。"
        "支付由 Myerson 规则计算；固定价格下，买家诚实报价 b=mu 是理论基线。"
        "平台价格状态由 MWU 根据历史报价和收入更新，短期动作可能影响后续价格路径。"
    )
    if role == "平台":
        return (
            "你是一个 Agent-based Modeling 数据市场中的平台定价 Agent。"
            "你的可见信息只包括当前观察里的候选价格、MWU 推荐价格、卖家数量和最近交易历史；"
            "不要假设观察中没有给出的买家私有估值。"
            "你的主要目标是在多轮交易中提高平台效用和收入，同时避免价格剧烈波动导致买家长期退出。"
            f"{mechanism}"
            f"{profile_guidance(role, profile)}"
            "决策时先比较 MWU 推荐价格、最近成交报价、平台收入和买家效用，再选择一个有效价格。"
            f"输出必须是一个 JSON 对象，schema 示例: {schema}。不要输出 Markdown、代码块或额外解释。"
        )
    if role == "买家":
        return (
            "你是一个 Agent-based Modeling 数据市场中的买家 Agent。"
            "你的可见信息只包括当前观察里的自己的估值 mu、当前平台价格和自己的历史；"
            "你看不到其他买家的估值、策略或未来动作。"
            "你的主要目标是在多轮交易中最大化自身效用，效用约为 mu * 数据质量 - 支付。"
            f"{mechanism}"
            f"{profile_guidance(role, profile)}"
            "决策时比较诚实报价、低报节省支付、高报获得完整数据这三类选择的收益和风险。"
            f"输出必须是一个 JSON 对象，schema 示例: {schema}。不要输出 Markdown、代码块或额外解释。"
        )
    raise ValueError(f"未知角色: {role}")


def default_user_prompt(
    role: str,
    obs: Mapping[str, Any],
    memory: Sequence[Mapping[str, Any]],
    *,
    profile: str = "balanced",
) -> str:
    schema = json.dumps(output_schema_for_role(role), ensure_ascii=False)
    return (
        f"当前角色: {role}\n"
        f"Prompt profile: {profile}\n"
        f"信息边界: {observation_contract(role)}\n"
        f"输出 JSON schema: {schema}\n"
        f"最近记忆(JSON): {json.dumps(list(memory), ensure_ascii=False)}\n"
        f"当前观察(JSON): {json.dumps(obs, ensure_ascii=False)}\n\n"
        "请按以下步骤在内部做判断，但最终只输出 JSON: "
        "1. 识别当前价格/估值/历史趋势；"
        "2. 判断本轮动作对效用、收入、数据质量和后续价格的影响；"
        "3. 选择一个数字动作，并在 reasoning 中用一句话说明依据。"
    )


def output_schema_for_role(role: str) -> Dict[str, Any]:
    if role == "平台":
        return {"reasoning": "一句话说明定价依据", "价格": 0.8}
    if role == "买家":
        return {"reasoning": "一句话说明报价依据", "报价": 0.8}
    raise ValueError(f"未知角色: {role}")


def observation_contract(role: str) -> str:
    if role == "平台":
        return "只能使用候选价格、MWU权重/MWU推荐价格、卖家数量和最近历史，不知道当前买家的真实估值。"
    if role == "买家":
        return "只能使用自己的估值mu、当前平台价格和自己的历史，不知道其他买家或卖家的私有信息。"
    raise ValueError(f"未知角色: {role}")


def profile_guidance(role: str, profile: str) -> str:
    normalized = profile.lower().strip()
    common = {
        "balanced": "Profile: balanced。兼顾理论基线、历史经验和风险，不要因为单轮异常值做极端动作。",
        "strategic": "Profile: strategic。允许考虑当前动作对未来价格轨迹的影响，但仍需避免明显负效用或无效交易。",
        "welfare": "Profile: welfare。优先考虑市场总福利和长期稳定，避免只让单方收益极端化。",
        "risk_averse": "Profile: risk_averse。偏好稳健动作，优先降低负效用、过高支付或过高价格导致的风险。",
    }
    platform_only = {
        "revenue_max": "Profile: revenue_max。优先最大化平台收入和平台效用，可参考 MWU 推荐并适度提高价格试探支付意愿。",
        "fairness_aware": "Profile: fairness_aware。除平台收入外，也关注买家效用和卖家参与激励，避免长期压低某一方收益。",
    }
    buyer_only = {
        "truthful": "Profile: truthful。将诚实报价 b=mu 作为强基线；除非历史显示明显风险，否则不要策略性偏离。",
        "shade": "Profile: shade。可以适度低报以节省支付并观察价格反馈，但不能忽视数据质量下降和负效用风险。",
    }
    if normalized in common:
        return common[normalized]
    if role == "平台" and normalized in platform_only:
        return platform_only[normalized]
    if role == "买家" and normalized in buyer_only:
        return buyer_only[normalized]
    supported = sorted(set(common) | (set(platform_only) if role == "平台" else set(buyer_only)))
    raise ValueError(f"未知 prompt profile: {profile!r}，{role} 支持: {supported}")


def infer_role_from_obs(obs: Optional[Mapping[str, Any]]) -> str:
    if obs is None:
        raise AgentActionParseError("未提供 role，且无法从空 obs 推断角色。")
    if "MWU推荐价格" in obs or "候选价格" in obs:
        return "平台"
    if "我的估值_mu" in obs or "当前平台价格" in obs:
        return "买家"
    raise AgentActionParseError(f"无法从观察字段推断角色: {list(obs.keys())}")


def fallback_action(obs: Mapping[str, Any], *, role: Optional[str] = None) -> ActionDict:
    inferred_role = role or infer_role_from_obs(obs)
    if inferred_role == "平台":
        if "MWU推荐价格" in obs:
            return {"价格": float(obs["MWU推荐价格"])}
        candidates = obs.get("候选价格", [])
        if candidates:
            return {"价格": float(candidates[0])}
        raise AgentActionParseError("平台 fallback 缺少 MWU推荐价格 或 候选价格。")
    if inferred_role == "买家":
        return {"报价": max(0.0, float(obs["我的估值_mu"]))}
    raise AgentActionParseError(f"未知角色: {inferred_role}")


def _to_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise AgentActionParseError(f"{field} 必须是数字，不能是 bool。")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text)
        if match:
            return float(match.group(0))
    raise AgentActionParseError(f"{field} 无法转换为数字: {value!r}")


def _balanced_json_objects(text: str) -> Iterable[str]:
    start = None
    depth = 0
    in_string = False
    escape = False
    for idx, char in enumerate(text):
        if start is None:
            if char == "{":
                start = idx
                depth = 1
                in_string = False
                escape = False
            continue
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start is not None:
                yield text[start : idx + 1]
                start = None


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
    "default_user_prompt",
    "observation_contract",
    "output_schema_for_role",
    "profile_guidance",
    "fallback_action",
    "infer_role_from_obs",
]
