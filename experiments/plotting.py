"""实验结果画图脚本。

用法示例：

    python -m marketplace_for_data_agent.experiments.plotting \
        --analysis-json outputs/analysis_summary.json \
        --output-dir outputs/figures

也可以直接传 CSV：

    python -m marketplace_for_data_agent.experiments.plotting \
        --summary-csv outputs/summary_table.csv \
        --step-csv outputs/step_table.csv \
        --output-dir outputs/figures
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


def _to_float(value: Any) -> float | None:
    """把 CSV/JSON 中的数字字段转成 float，无法转换时返回 None。"""

    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_csv_table(path: str | Path) -> list[dict[str, Any]]:
    """读取 CSV 表格为字典列表。"""

    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_analysis_json(path: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """从 analysis_summary.json 中读取汇总表和逐轮表。"""

    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("summary_table", []), payload.get("step_table", [])


def _group_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    """按 mode 分组，方便为每种实验模式画一条线。"""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("mode", "unknown"))].append(row)
    return grouped


def plot_trajectory(
    step_table: list[dict[str, Any]],
    metric: str,
    output_path: str | Path,
    *,
    ylabel: str | None = None,
    title: str | None = None,
) -> str:
    """绘制某个逐轮指标的轨迹图。

    例如 metric="price" 可画价格轨迹，metric="bid" 可画报价轨迹。
    """

    import matplotlib.pyplot as plt

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
    for mode, rows in _group_rows(step_table).items():
        points = []
        for row in rows:
            x = _to_float(row.get("step_index"))
            y = _to_float(row.get(metric))
            if x is not None and y is not None:
                points.append((x, y))
        if not points:
            continue
        points.sort(key=lambda item: item[0])
        xs, ys = zip(*points)
        ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=3.5, label=mode)

    ax.set_xlabel("Round")
    ax.set_ylabel(ylabel or metric)
    ax.set_title(title or f"{metric} trajectory")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return str(output)


def plot_utility_gap(
    summary_table: list[dict[str, Any]],
    output_path: str | Path,
) -> str:
    """绘制平台和买家 Utility Gap 的分组柱状图。"""

    import matplotlib.pyplot as plt

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    modes = [str(row.get("mode", "unknown")) for row in summary_table]
    platform_gap = [_to_float(row.get("platform_gap")) or 0.0 for row in summary_table]
    buyer_gap = [_to_float(row.get("buyer_average_gap")) or 0.0 for row in summary_table]
    x = list(range(len(modes)))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
    ax.bar([i - width / 2 for i in x], platform_gap, width=width, label="platform_gap")
    ax.bar([i + width / 2 for i in x], buyer_gap, width=width, label="buyer_average_gap")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(modes, rotation=20, ha="right")
    ax.set_ylabel("Utility Gap")
    ax.set_title("Utility Gap by Mode")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return str(output)


def plot_seller_utility(
    step_table: list[dict[str, Any]],
    output_path: str | Path,
) -> str:
    """绘制卖家总效用轨迹。

    step_table 中 seller_utilities 是列表字符串时，会先解析再求和。
    """

    import matplotlib.pyplot as plt

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
    for mode, rows in _group_rows(step_table).items():
        points = []
        for row in rows:
            x = _to_float(row.get("step_index"))
            raw = row.get("seller_utilities")
            if x is None or raw in (None, ""):
                continue
            if isinstance(raw, list):
                values = raw
            else:
                try:
                    values = ast.literal_eval(str(raw))
                except (SyntaxError, ValueError):
                    values = []
            y = sum(float(v) for v in values) if values else None
            if y is not None:
                points.append((x, y))
        if not points:
            continue
        points.sort(key=lambda item: item[0])
        xs, ys = zip(*points)
        ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=3.5, label=mode)

    ax.set_xlabel("Round")
    ax.set_ylabel("Seller Utility Sum")
    ax.set_title("Seller Utility Trajectory")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return str(output)


def plot_standard_charts(
    summary_table: list[dict[str, Any]],
    step_table: list[dict[str, Any]],
    output_dir: str | Path,
) -> dict[str, str]:
    """生成实验报告常用图表。"""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "price": plot_trajectory(
            step_table,
            "price",
            out / "price_trajectory.png",
            ylabel="Price",
            title="Price Trajectory",
        ),
        "bid": plot_trajectory(
            step_table,
            "bid",
            out / "bid_trajectory.png",
            ylabel="Bid",
            title="Bid Trajectory",
        ),
        "gain": plot_trajectory(
            step_table,
            "gain",
            out / "gain_trajectory.png",
            ylabel="Gain",
            title="Data Gain Trajectory",
        ),
        "platform_revenue": plot_trajectory(
            step_table,
            "platform_revenue",
            out / "platform_revenue_trajectory.png",
            ylabel="Platform Revenue",
            title="Platform Revenue Trajectory",
        ),
        "buyer_utility": plot_trajectory(
            step_table,
            "buyer_utility",
            out / "buyer_utility_trajectory.png",
            ylabel="Buyer Utility",
            title="Buyer Utility Trajectory",
        ),
        "platform_utility": plot_trajectory(
            step_table,
            "platform_utility",
            out / "platform_utility_trajectory.png",
            ylabel="Platform Utility",
            title="Platform Utility Trajectory",
        ),
        "utility_gap": plot_utility_gap(summary_table, out / "utility_gap.png"),
        "seller_utility": plot_seller_utility(step_table, out / "seller_utility_trajectory.png"),
    }
    return paths


def main() -> None:
    """命令行入口：读取分析结果并生成标准图表。"""

    parser = argparse.ArgumentParser(description="Plot marketplace experiment results.")
    parser.add_argument("--analysis-json", type=str, default=None)
    parser.add_argument("--summary-csv", type=str, default=None)
    parser.add_argument("--step-csv", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    if args.analysis_json:
        summary_table, step_table = load_analysis_json(args.analysis_json)
    else:
        if not args.summary_csv or not args.step_csv:
            raise SystemExit("请提供 --analysis-json，或同时提供 --summary-csv 和 --step-csv。")
        summary_table = load_csv_table(args.summary_csv)
        step_table = load_csv_table(args.step_csv)

    paths = plot_standard_charts(summary_table, step_table, args.output_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
