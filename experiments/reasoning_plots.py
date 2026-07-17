"""基于 LLM reasoning 与决策日志生成解释性图表。

这些图不是替代 summary_table，而是用于解释“为什么 Agent 结果会偏离规则基准”：
1. 买家 Agent 在不同模式下的诚实/低报/高报比例。
2. 买家是否主要根据 price 与 mu 的关系做决策。
3. 平台 Agent 的价格探索与稳定过程。
4. 四种模式下平均价格、报价、平台收入和数据增益的对比。
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


MODE_ORDER = ["all_rule", "seller_agent", "buyer_agent", "both_agent"]


def _read_csv(path: str | Path) -> list[dict[str, Any]]:
    """读取 CSV 文件。"""

    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _read_json_field(value: str | None) -> dict[str, Any]:
    """读取 CSV 中保存的 JSON 字符串。"""

    if not value:
        return {}
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _number(value: Any, default: float = 0.0) -> float:
    """把字段转成 float。"""

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _action_value(action: dict[str, Any], chinese_key: str, english_key: str) -> float:
    """从 action 中兼容读取中文或英文动作字段。"""

    if chinese_key in action:
        return _number(action[chinese_key])
    if english_key in action:
        return _number(action[english_key])
    return _number(next(iter(action.values()))) if action else 0.0


def _decision_rows(output_dir: str | Path, mode: str) -> list[dict[str, Any]]:
    """读取某个模式的 Agent 决策日志。"""

    path = Path(output_dir) / "modes" / mode / "decision_steps.csv"
    if not path.exists():
        return []
    return _read_csv(path)


def _buyer_decisions(output_dir: str | Path, mode: str) -> list[dict[str, Any]]:
    """提取买家 Agent 决策，附加 mu、price、bid、行为类别。"""

    rows = []
    for row in _decision_rows(output_dir, mode):
        if row.get("role") != "买家":
            continue
        obs = _read_json_field(row.get("observation"))
        action = _read_json_field(row.get("action"))
        debug = _read_json_field(row.get("agent_debug"))
        mu = _number(obs.get("我的估值_mu"))
        price = _number(obs.get("当前平台价格"))
        bid = _action_value(action, "报价", "bid")
        if mu <= 0:
            continue
        if abs(bid - mu) <= 0.02:
            behavior = "truthful"
        elif bid < mu:
            behavior = "shade"
        else:
            behavior = "overbid"
        rows.append(
            {
                "round": int(_number(obs.get("当前轮次"), _number(row.get("step_index")))),
                "mu": mu,
                "price": price,
                "bid": bid,
                "bid_mu_ratio": bid / mu,
                "price_minus_mu": price - mu,
                "behavior": behavior,
                "reasoning": _read_json_field(debug.get("raw")).get("reasoning", debug.get("raw", "")),
            }
        )
    return rows


def _platform_decisions(output_dir: str | Path, mode: str) -> list[dict[str, Any]]:
    """提取平台 Agent 决策，附加价格与 reasoning。"""

    rows = []
    for row in _decision_rows(output_dir, mode):
        if row.get("role") != "平台":
            continue
        obs = _read_json_field(row.get("observation"))
        action = _read_json_field(row.get("action"))
        debug = _read_json_field(row.get("agent_debug"))
        price = _action_value(action, "价格", "price")
        rows.append(
            {
                "round": int(_number(obs.get("当前轮次"), _number(row.get("step_index")))),
                "price": price,
                "reasoning": _read_json_field(debug.get("raw")).get("reasoning", debug.get("raw", "")),
            }
        )
    return rows


def plot_buyer_behavior_counts(output_dir: str | Path, figure_dir: str | Path) -> str:
    """画买家 Agent 的行为分类柱状图。"""

    import matplotlib.pyplot as plt

    modes = ["buyer_agent", "both_agent"]
    behaviors = ["truthful", "shade", "overbid"]
    labels = {"truthful": "Truthful", "shade": "Shading", "overbid": "Overbid"}
    colors = {"truthful": "#4C78A8", "shade": "#F58518", "overbid": "#54A24B"}

    counts = {mode: Counter(item["behavior"] for item in _buyer_decisions(output_dir, mode)) for mode in modes}
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=160)
    bottom = [0] * len(modes)
    for behavior in behaviors:
        values = [counts[mode].get(behavior, 0) for mode in modes]
        ax.bar(modes, values, bottom=bottom, label=labels[behavior], color=colors[behavior])
        bottom = [b + v for b, v in zip(bottom, values)]

    ax.set_ylabel("Decision Count")
    ax.set_title("Buyer Agent Behavior Classification")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    path = Path(figure_dir) / "reason_buyer_behavior_counts.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_buyer_price_mu_rule(output_dir: str | Path, figure_dir: str | Path) -> str:
    """画 price-mu 与 bid/mu 的关系，展示买家低报机制。"""

    import matplotlib.pyplot as plt

    data = []
    for mode in ["buyer_agent", "both_agent"]:
        for item in _buyer_decisions(output_dir, mode):
            item = dict(item)
            item["mode"] = mode
            data.append(item)

    colors = {"truthful": "#4C78A8", "shade": "#F58518", "overbid": "#54A24B"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), dpi=160, sharey=True)
    for ax, mode in zip(axes, ["buyer_agent", "both_agent"]):
        rows = [item for item in data if item["mode"] == mode]
        for behavior, color in colors.items():
            xs = [item["price_minus_mu"] for item in rows if item["behavior"] == behavior]
            ys = [item["bid_mu_ratio"] for item in rows if item["behavior"] == behavior]
            ax.scatter(xs, ys, s=28, color=color, alpha=0.85, label=behavior)
        ax.axvline(0.0, color="black", linewidth=1.0, linestyle="--")
        ax.axhline(1.0, color="gray", linewidth=1.0, linestyle=":")
        ax.set_title(mode)
        ax.set_xlabel("price - mu")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("bid / mu")
    axes[1].legend(title="Behavior", fontsize=8)

    path = Path(figure_dir) / "reason_buyer_price_mu_rule.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_platform_price_discovery(output_dir: str | Path, figure_dir: str | Path) -> str:
    """画平台 Agent 的价格探索过程。"""

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=160)
    for mode, color in [("seller_agent", "#4C78A8"), ("both_agent", "#F58518")]:
        rows = _platform_decisions(output_dir, mode)
        xs = [item["round"] for item in rows]
        ys = [item["price"] for item in rows]
        ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=3.0, color=color, label=mode)
    ax.set_xlabel("Round")
    ax.set_ylabel("Platform Price")
    ax.set_title("Platform Agent Price Discovery")
    ax.grid(True, alpha=0.25)
    ax.legend()
    path = Path(figure_dir) / "reason_platform_price_discovery.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_mode_behavior_metrics(output_dir: str | Path, figure_dir: str | Path) -> str:
    """画四种模式的关键行为均值。"""

    import matplotlib.pyplot as plt

    rows = _read_csv(Path(output_dir) / "step_table.csv")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["mode"]].append(row)

    metrics = [
        ("price", "Avg Price"),
        ("bid", "Avg Bid"),
        ("platform_revenue", "Avg Revenue"),
        ("gain", "Avg Gain"),
    ]
    modes = [mode for mode in MODE_ORDER if mode in grouped]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.8), dpi=160)
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        values = []
        for mode in modes:
            mode_rows = grouped[mode]
            values.append(sum(_number(row[metric]) for row in mode_rows) / len(mode_rows))
        ax.bar(modes, values, color=["#4C78A8", "#F58518", "#54A24B", "#B279A2"][: len(modes)])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=18)
        ax.grid(True, axis="y", alpha=0.25)
    path = Path(figure_dir) / "reason_mode_behavior_metrics.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_reasoning_charts(output_dir: str | Path, figure_dir: str | Path | None = None) -> dict[str, str]:
    """生成全部 reasoning 辅助图。"""

    out = Path(figure_dir) if figure_dir is not None else Path(output_dir) / "figures_reasoning"
    out.mkdir(parents=True, exist_ok=True)
    return {
        "buyer_behavior_counts": plot_buyer_behavior_counts(output_dir, out),
        "buyer_price_mu_rule": plot_buyer_price_mu_rule(output_dir, out),
        "platform_price_discovery": plot_platform_price_discovery(output_dir, out),
        "mode_behavior_metrics": plot_mode_behavior_metrics(output_dir, out),
    }


def main() -> None:
    """命令行入口。"""

    parser = argparse.ArgumentParser(description="Plot LLM reasoning diagnostics.")
    parser.add_argument("--output-dir", required=True, help="实验输出目录，例如 outputs/llm_test")
    parser.add_argument("--figure-dir", default=None, help="图片输出目录，默认 output-dir/figures_reasoning")
    args = parser.parse_args()

    paths = plot_reasoning_charts(args.output_dir, args.figure_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
