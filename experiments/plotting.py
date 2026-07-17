"""实验结果画图脚本。

图表设计围绕四个实验问题：
1. 平台定价和买家报价是否发生偏离？
2. 报价与定价变化是否影响数据增益和平台收入？
3. 不同模式下逐轮效用与累计效用如何变化？
4. 各模式整体结果相对 all_rule 基准偏离多少？

用法示例：
    python experiments/plotting.py \
        --summary-csv outputs/exp/summary_table.csv \
        --step-csv outputs/exp/step_table.csv \
        --output-dir outputs/exp/figures

也可以直接读取 analysis_summary.json：
    python experiments/plotting.py \
        --analysis-json outputs/exp/analysis_summary.json \
        --output-dir outputs/exp/figures
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


MODE_ORDER = ["all_rule", "seller_agent", "buyer_agent", "both_agent"]


def _to_float(value: Any) -> float | None:
    """把 CSV/JSON 中的数字字段转成 float，无法转换时返回 None。"""

    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _round_x(row: Mapping[str, Any]) -> float | None:
    """读取横轴轮次，优先使用 round_index，兼容旧版 step_index。"""

    value = row.get("round_index")
    if value in (None, ""):
        value = row.get("step_index")
    return _to_float(value)


def _mode_sort_key(mode: str) -> tuple[int, str]:
    """让图例和柱状图按固定模式顺序排列。"""

    if mode in MODE_ORDER:
        return (MODE_ORDER.index(mode), mode)
    return (len(MODE_ORDER), mode)


def _group_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    """按 mode 分组，并保证每组内部按交易轮次排序。"""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("mode", "unknown"))].append(row)
    for mode in grouped:
        grouped[mode].sort(key=lambda item: _round_x(item) if _round_x(item) is not None else -1)
    return dict(sorted(grouped.items(), key=lambda item: _mode_sort_key(item[0])))


def _series(rows: Iterable[Mapping[str, Any]], metric: str) -> tuple[list[float], list[float]]:
    """提取某个逐轮指标的 x/y 序列。"""

    points = []
    for row in rows:
        x = _round_x(row)
        y = _to_float(row.get(metric))
        if x is not None and y is not None:
            points.append((x, y))
    points.sort(key=lambda item: item[0])
    if not points:
        return [], []
    xs, ys = zip(*points)
    return list(xs), list(ys)


def _cumulative(values: Iterable[float]) -> list[float]:
    """计算累计值序列。"""

    total = 0.0
    out = []
    for value in values:
        total += float(value)
        out.append(total)
    return out


def _seller_utility_sum(raw: Any) -> float | None:
    """把 seller_utilities 字段解析为卖家总效用。"""

    if raw in (None, ""):
        return None
    if isinstance(raw, list):
        values = raw
    else:
        try:
            values = ast.literal_eval(str(raw))
        except (SyntaxError, ValueError):
            return None
    try:
        return float(sum(float(v) for v in values))
    except (TypeError, ValueError):
        return None


def load_csv_table(path: str | Path) -> list[dict[str, Any]]:
    """读取 CSV 表格为字典列表。"""

    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_analysis_json(path: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """从 analysis_summary.json 中读取汇总表和逐轮交易表。"""

    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("summary_table", []), payload.get("step_table", [])


def _save(fig: Any, output_path: str | Path) -> str:
    """统一保存图片。"""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    return str(output)


def plot_price_bid(step_table: list[dict[str, Any]], output_path: str | Path) -> str:
    """展示平台定价与买家报价轨迹，用于观察双边行为是否偏离基准。"""

    import matplotlib.pyplot as plt

    grouped = _group_rows(step_table)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.2), dpi=160, sharex=True)
    for mode, rows in grouped.items():
        xs, prices = _series(rows, "price")
        _, bids = _series(rows, "bid")
        if xs and prices:
            axes[0].plot(xs, prices, marker="o", linewidth=1.8, markersize=3.2, label=mode)
        if xs and bids:
            axes[1].plot(xs, bids, marker="o", linewidth=1.8, markersize=3.2, label=mode)

    axes[0].set_title("Platform Price by Round")
    axes[0].set_ylabel("Price")
    axes[1].set_title("Buyer Bid by Round")
    axes[1].set_ylabel("Bid")
    axes[1].set_xlabel("Round")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=2, fontsize=8)
    return _save(fig, output_path)


def plot_trajectory(
    step_table: list[dict[str, Any]],
    metric: str,
    output_path: str | Path,
    *,
    ylabel: str | None = None,
    title: str | None = None,
) -> str:
    """兼容旧接口：绘制单个逐轮指标轨迹。

    新版标准图不再主动使用这个函数，但包入口仍会导出它，避免旧脚本导入失败。
    """

    import matplotlib.pyplot as plt

    grouped = _group_rows(step_table)
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=160)
    for mode, rows in grouped.items():
        xs, ys = _series(rows, metric)
        if xs and ys:
            ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=3.2, label=mode)

    ax.set_xlabel("Round")
    ax.set_ylabel(ylabel or metric)
    ax.set_title(title or f"{metric} by Round")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    return _save(fig, output_path)


def plot_gain_revenue(step_table: list[dict[str, Any]], output_path: str | Path) -> str:
    """展示数据增益和平台收入，用于连接交易行为与机制结算结果。"""

    import matplotlib.pyplot as plt

    grouped = _group_rows(step_table)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.2), dpi=160, sharex=True)
    for mode, rows in grouped.items():
        xs, gains = _series(rows, "gain")
        _, revenues = _series(rows, "platform_revenue")
        if xs and gains:
            axes[0].plot(xs, gains, marker="o", linewidth=1.8, markersize=3.2, label=mode)
        if xs and revenues:
            axes[1].plot(xs, revenues, marker="o", linewidth=1.8, markersize=3.2, label=mode)

    axes[0].set_title("Allocated Data Gain by Round")
    axes[0].set_ylabel("Gain")
    axes[1].set_title("Platform Revenue by Round")
    axes[1].set_ylabel("Revenue")
    axes[1].set_xlabel("Round")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=2, fontsize=8)
    return _save(fig, output_path)


def plot_round_utility(step_table: list[dict[str, Any]], output_path: str | Path) -> str:
    """展示每轮买家效用和平台效用，用于观察收益分配的动态变化。"""

    import matplotlib.pyplot as plt

    grouped = _group_rows(step_table)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.2), dpi=160, sharex=True)
    for mode, rows in grouped.items():
        xs, buyer_utility = _series(rows, "buyer_utility")
        _, platform_utility = _series(rows, "platform_utility")
        if xs and buyer_utility:
            axes[0].plot(xs, buyer_utility, marker="o", linewidth=1.8, markersize=3.2, label=mode)
        if xs and platform_utility:
            axes[1].plot(xs, platform_utility, marker="o", linewidth=1.8, markersize=3.2, label=mode)

    axes[0].set_title("Buyer Utility by Round")
    axes[0].set_ylabel("Buyer Utility")
    axes[1].set_title("Platform Utility by Round")
    axes[1].set_ylabel("Platform Utility")
    axes[1].set_xlabel("Round")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=2, fontsize=8)
    return _save(fig, output_path)


def plot_cumulative_utility(step_table: list[dict[str, Any]], output_path: str | Path) -> str:
    """展示累计平台效用和累计买家效用，用于比较长期收益走势。"""

    import matplotlib.pyplot as plt

    grouped = _group_rows(step_table)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.2), dpi=160, sharex=True)
    for mode, rows in grouped.items():
        xs, buyer_utility = _series(rows, "buyer_utility")
        _, platform_utility = _series(rows, "platform_utility")
        if xs and buyer_utility:
            axes[0].plot(xs, _cumulative(buyer_utility), linewidth=2.0, label=mode)
        if xs and platform_utility:
            axes[1].plot(xs, _cumulative(platform_utility), linewidth=2.0, label=mode)

    axes[0].set_title("Cumulative Buyer Utility")
    axes[0].set_ylabel("Cumulative Utility")
    axes[1].set_title("Cumulative Platform Utility")
    axes[1].set_ylabel("Cumulative Utility")
    axes[1].set_xlabel("Round")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=2, fontsize=8)
    return _save(fig, output_path)


def plot_summary_utility(summary_table: list[dict[str, Any]], output_path: str | Path) -> str:
    """展示模式级总效用，直接回答哪种模式让平台/买家更受益。"""

    import matplotlib.pyplot as plt

    rows = sorted(summary_table, key=lambda row: _mode_sort_key(str(row.get("mode", ""))))
    modes = [str(row.get("mode", "unknown")) for row in rows]
    platform = [_to_float(row.get("platform_utility")) or 0.0 for row in rows]
    buyer_avg = [_to_float(row.get("buyer_average_utility")) or 0.0 for row in rows]
    x = list(range(len(modes)))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=160)
    ax.bar([i - width / 2 for i in x], platform, width=width, label="platform_total_utility")
    ax.bar([i + width / 2 for i in x], buyer_avg, width=width, label="buyer_average_utility")
    ax.set_xticks(x)
    ax.set_xticklabels(modes, rotation=15, ha="right")
    ax.set_ylabel("Utility")
    ax.set_title("Mode-Level Utility Comparison")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    return _save(fig, output_path)


def plot_utility_gap(
    summary_table: list[dict[str, Any]],
    output_path: str | Path,
    *,
    epsilon: float | None = None,
) -> str:
    """展示相对 all_rule 的效用差距，用于判断是否偏离理论/规则基准。"""

    import matplotlib.pyplot as plt

    rows = sorted(summary_table, key=lambda row: _mode_sort_key(str(row.get("mode", ""))))
    modes = [str(row.get("mode", "unknown")) for row in rows]
    platform_gap = [_to_float(row.get("platform_gap")) or 0.0 for row in rows]
    buyer_gap = [_to_float(row.get("buyer_average_gap")) or 0.0 for row in rows]
    x = list(range(len(modes)))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=160)
    ax.bar([i - width / 2 for i in x], platform_gap, width=width, label="platform_gap")
    ax.bar([i + width / 2 for i in x], buyer_gap, width=width, label="buyer_average_gap")
    ax.axhline(0.0, color="black", linewidth=0.8)
    if epsilon is not None:
        ax.axhline(float(epsilon), color="gray", linewidth=1.0, linestyle="--", label="epsilon")
        ax.axhline(-float(epsilon), color="gray", linewidth=1.0, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(modes, rotation=15, ha="right")
    ax.set_ylabel("Baseline Utility - Experiment Utility")
    ax.set_title("Utility Gap Relative to all_rule")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    return _save(fig, output_path)


def plot_seller_utility(step_table: list[dict[str, Any]], output_path: str | Path) -> str:
    """展示卖家总效用轨迹，用于观察平台收入分账对卖家侧的影响。"""

    import matplotlib.pyplot as plt

    grouped = _group_rows(step_table)
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=160)
    for mode, rows in grouped.items():
        points = []
        for row in rows:
            x = _round_x(row)
            y = _seller_utility_sum(row.get("seller_utilities"))
            if x is not None and y is not None:
                points.append((x, y))
        if not points:
            continue
        points.sort(key=lambda item: item[0])
        xs, ys = zip(*points)
        ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=3.2, label=mode)

    ax.set_xlabel("Round")
    ax.set_ylabel("Seller Utility Sum")
    ax.set_title("Seller Utility by Round")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    return _save(fig, output_path)


def plot_standard_charts(
    summary_table: list[dict[str, Any]],
    step_table: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    epsilon: float | None = None,
) -> dict[str, str]:
    """生成报告中最有解释力的一组标准图。"""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return {
        "price_bid": plot_price_bid(step_table, out / "price_bid_trajectory.png"),
        "gain_revenue": plot_gain_revenue(step_table, out / "gain_revenue_trajectory.png"),
        "round_utility": plot_round_utility(step_table, out / "round_utility_trajectory.png"),
        "cumulative_utility": plot_cumulative_utility(
            step_table,
            out / "cumulative_utility.png",
        ),
        "summary_utility": plot_summary_utility(summary_table, out / "summary_utility.png"),
        "utility_gap": plot_utility_gap(summary_table, out / "utility_gap.png", epsilon=epsilon),
        "seller_utility": plot_seller_utility(step_table, out / "seller_utility_trajectory.png"),
    }


def main() -> None:
    """命令行入口：读取分析结果并生成标准图表。"""

    parser = argparse.ArgumentParser(description="Plot marketplace experiment results.")
    parser.add_argument("--analysis-json", type=str, default=None)
    parser.add_argument("--summary-csv", type=str, default=None)
    parser.add_argument("--step-csv", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--epsilon", type=float, default=None)
    args = parser.parse_args()

    if args.analysis_json:
        summary_table, step_table = load_analysis_json(args.analysis_json)
    else:
        if not args.summary_csv or not args.step_csv:
            raise SystemExit("请提供 --analysis-json，或同时提供 --summary-csv 和 --step-csv。")
        summary_table = load_csv_table(args.summary_csv)
        step_table = load_csv_table(args.step_csv)

    paths = plot_standard_charts(
        summary_table,
        step_table,
        args.output_dir,
        epsilon=args.epsilon,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
