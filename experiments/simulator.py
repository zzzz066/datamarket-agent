"""Module 3: 多轮模拟实验流水线。

本模块负责创建 AgentMarketEnv、驱动多轮交易、记录逐轮日志，并返回结构化结果。
Agent 的 prompt、推理和接口实现由模块 2 提供，这里只调用统一的 act(obs) 接口。
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Protocol, Sequence

from ..agents.base import BaseAgent
from ..env import AgentMarketEnv, Buyer, Seller


class _ActingAgent(Protocol):
    """兼容任意带 act(obs) 方法的 Agent。"""

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        ...


AgentLike = Optional[BaseAgent | _ActingAgent | Callable[[Dict[str, Any]], Dict[str, Any]]]


@dataclass
class EpisodeStepLog:
    """一次外部 Agent 决策的日志。"""

    step_index: int
    role: str
    observation: Dict[str, Any]
    action: Dict[str, Any]
    reward: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeRunResult:
    """一次完整 episode 的输出。"""

    steps: list[EpisodeStepLog]
    final_reward: Dict[str, Any]
    summary: Dict[str, Any]
    ground_truth: Dict[str, Any]


def _call_agent(agent: AgentLike, obs: Dict[str, Any]) -> Dict[str, Any]:
    """统一调用 Agent 对象或函数式策略。"""

    if agent is None:
        raise ValueError("当前角色需要 Agent，但没有提供可用的决策器。")
    if hasattr(agent, "act"):
        return getattr(agent, "act")(obs)
    return agent(obs)


def _hide_mwu_fields(obs: Dict[str, Any]) -> Dict[str, Any]:
    """返回不包含 MWU 内部状态的观测副本。

    这个过滤只发生在 simulator 层，不修改 env/wrapper.py 中的环境观测定义。
    """

    hidden_keys = {"MWU权重", "MWU推荐价格"}
    return {key: value for key, value in obs.items() if key not in hidden_keys}


def run_episode(
    sellers: Sequence[Seller],
    buyers: Sequence[Buyer],
    *,
    platform_agent: AgentLike = None,
    buyer_agent: AgentLike = None,
    platform_mode: str = "agent",
    buyer_mode: str = "agent",
    price_bounds: tuple[float, float, float] = (0.1, 1.6, 0.05),
    delta: float = 0.18,
    af_mode: str = "gaussian",
    noise_sigma: float = 1.0,
    shapley_permutations: int = 256,
    lambda_penalty: float = 0.7,
    seed: Optional[int] = None,
    shade_factor: float = 0.65,
    overbid_factor: float = 1.4,
    expose_mwu_to_platform_agent: bool = False,
) -> EpisodeRunResult:
    """运行一个完整 episode，并返回结构化日志。"""

    env = AgentMarketEnv(
        sellers,
        buyers,
        price_bounds=price_bounds,
        delta=delta,
        af_mode=af_mode,
        noise_sigma=noise_sigma,
        shapley_permutations=shapley_permutations,
        lambda_penalty=lambda_penalty,
        seed=seed,
        platform_mode=platform_mode,
        buyer_mode=buyer_mode,
        shade_factor=shade_factor,
        overbid_factor=overbid_factor,
    )

    steps: list[EpisodeStepLog] = []
    obs = env.reset()
    step_index = 0

    while not env.done:
        role = env.current_role
        current_obs = obs
        agent_obs = current_obs
        if role == "平台":
            if not expose_mwu_to_platform_agent:
                agent_obs = _hide_mwu_fields(current_obs)
            action = _call_agent(platform_agent, agent_obs)
        elif role == "买家":
            action = _call_agent(buyer_agent, agent_obs)
        else:
            break

        obs = env.step(role, action)
        steps.append(
            EpisodeStepLog(
                step_index=step_index,
                role=role,
                observation=agent_obs,
                action=action,
                reward=env.get_last_reward(),
            )
        )
        step_index += 1

    final_reward = obs if isinstance(obs, dict) else env.get_last_reward()
    return EpisodeRunResult(
        steps=steps,
        final_reward=final_reward,
        summary=env.episode_summary(),
        ground_truth=env.ground_truth(),
    )


def episode_to_dict(result: EpisodeRunResult) -> Dict[str, Any]:
    """把 episode 结果转换为 JSON 兼容 dict。"""

    return {
        "steps": [
            {
                "step_index": item.step_index,
                "role": item.role,
                "observation": item.observation,
                "action": item.action,
                "reward": item.reward,
            }
            for item in result.steps
        ],
        "final_reward": result.final_reward,
        "summary": result.summary,
        "ground_truth": result.ground_truth,
    }


def save_episode_logs(result: EpisodeRunResult, output_dir: str | Path) -> Dict[str, str]:
    """保存单次 episode 的 JSON 和 CSV 日志。"""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "episode_log.json"
    csv_path = out / "episode_steps.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(episode_to_dict(result), f, ensure_ascii=False, indent=2)

    rows = []
    for step in result.steps:
        reward = step.reward or {}
        rows.append(
            {
                "step_index": step.step_index,
                "role": step.role,
                "price": reward.get("价格"),
                "bid": reward.get("报价"),
                "buyer_utility": reward.get("买家效用"),
                "platform_utility": reward.get("平台效用"),
                "platform_revenue": reward.get("平台收入"),
                "gain": reward.get("增益"),
            }
        )
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step_index",
                "role",
                "price",
                "bid",
                "buyer_utility",
                "platform_utility",
                "platform_revenue",
                "gain",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return {"json": str(json_path), "csv": str(csv_path)}


def run_baseline_suite(
    sellers: Sequence[Seller],
    buyers: Sequence[Buyer],
    *,
    seed: Optional[int] = 42,
    shade_factor: float = 0.65,
    overbid_factor: float = 1.4,
    **env_kwargs: Any,
) -> Dict[str, EpisodeRunResult]:
    """运行 truthful、shade、overbid 三组规则买家基线。"""

    common = dict(env_kwargs)
    return {
        "truthful": run_episode(
            sellers,
            buyers,
            platform_mode="mwu",
            buyer_mode="truthful",
            seed=seed,
            shade_factor=shade_factor,
            overbid_factor=overbid_factor,
            **common,
        ),
        "shade": run_episode(
            sellers,
            buyers,
            platform_mode="mwu",
            buyer_mode="shade",
            seed=seed,
            shade_factor=shade_factor,
            overbid_factor=overbid_factor,
            **common,
        ),
        "overbid": run_episode(
            sellers,
            buyers,
            platform_mode="mwu",
            buyer_mode="overbid",
            seed=seed,
            shade_factor=shade_factor,
            overbid_factor=overbid_factor,
            **common,
        ),
    }


def run_four_mode_suite(
    sellers: Sequence[Seller],
    buyers: Sequence[Buyer],
    *,
    seller_agent: AgentLike = None,
    buyer_agent: AgentLike = None,
    seed: Optional[int] = 42,
    expose_mwu_to_seller_agent: bool = False,
    **env_kwargs: Any,
) -> Dict[str, EpisodeRunResult]:
    """运行 all_rule、seller_agent、buyer_agent、both_agent 四种模式。

    当前没有卖家逐轮行动接口，因此 seller_agent 表示平台/卖方定价侧 Agent。
    """

    results: Dict[str, EpisodeRunResult] = {}
    common = dict(env_kwargs)

    results["all_rule"] = run_episode(
        sellers,
        buyers,
        platform_mode="mwu",
        buyer_mode="truthful",
        seed=seed,
        **common,
    )

    if seller_agent is not None:
        results["seller_agent"] = run_episode(
            sellers,
            buyers,
            platform_agent=seller_agent,
            platform_mode="agent",
            buyer_mode="truthful",
            seed=seed,
            expose_mwu_to_platform_agent=expose_mwu_to_seller_agent,
            **common,
        )

    if buyer_agent is not None:
        results["buyer_agent"] = run_episode(
            sellers,
            buyers,
            buyer_agent=buyer_agent,
            platform_mode="mwu",
            buyer_mode="agent",
            seed=seed,
            **common,
        )

    if seller_agent is not None and buyer_agent is not None:
        results["both_agent"] = run_episode(
            sellers,
            buyers,
            platform_agent=seller_agent,
            buyer_agent=buyer_agent,
            platform_mode="agent",
            buyer_mode="agent",
            seed=seed,
            expose_mwu_to_platform_agent=expose_mwu_to_seller_agent,
            **common,
        )

    return results
