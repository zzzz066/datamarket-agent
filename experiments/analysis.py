"""Module 4: 实验结果分析工具。

本模块只做数值整理和表格输出：
1. 以 all_rule 规则组作为基准，计算各实验模式的 Utility Gap。
2. 计算 Pass@epsilon。
3. 生成模式汇总表和逐轮轨迹表，供报告与画图脚本使用。
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from .simulator import EpisodeRunResult, episode_to_dict


@dataclass(frozen=True)
class UtilityGapReport:
    """单个实验模式相对规则基准组的效用差距。"""

    label: str
    reference_label: str
    platform_utility: float
    platform_baseline: float
    platform_gap: float
    buyer_average_utility: float
    buyer_average_baseline: float
    buyer_average_gap: float
    pass_epsilon: bool


def _number(value: Any) -> float:
    """把日志里的数字统一转成 float，缺失值按 0 处理。"""

    if value is None:
        return 0.0
    return float(value)


def _lookup_number(data: Mapping[str, Any], *keys: str) -> float:
    """按候选键读取数字，兼容字段名调整。"""

    for key in keys:
        if key in data:
            return _number(data[key])
    return 0.0


def _reference_result(
    results: Mapping[str, EpisodeRunResult],
    reference_mode: str,
) -> tuple[str, EpisodeRunResult]:
    """读取基准组；默认使用 all_rule，不存在时退回第一组结果。"""

    if reference_mode in results:
        return reference_mode, results[reference_mode]
    if not results:
        raise ValueError("results 不能为空。")
    name = next(iter(results))
    return name, results[name]


def utility_gap(
    result: EpisodeRunResult,
    reference: EpisodeRunResult,
    *,
    label: str = "experiment",
    reference_label: str = "all_rule",
    epsilon: float = 0.05,
) -> UtilityGapReport:
    """计算单个 episode 相对规则基准组的效用差距。

    Utility Gap = 基准组效用 - 当前实验效用。
    因此 all_rule 自身的 gap 应为 0。
    """

    summary = result.summary
    ref_summary = reference.summary
    platform_utility = _lookup_number(summary, "平台总效用")
    platform_baseline = _lookup_number(ref_summary, "平台总效用")
    buyer_avg = _lookup_number(summary, "买家平均效用")
    buyer_baseline_avg = _lookup_number(ref_summary, "买家平均效用")
    platform_gap = platform_baseline - platform_utility
    buyer_gap = buyer_baseline_avg - buyer_avg
    pass_epsilon = abs(platform_gap) <= epsilon and abs(buyer_gap) <= epsilon
    return UtilityGapReport(
        label=label,
        reference_label=reference_label,
        platform_utility=platform_utility,
        platform_baseline=platform_baseline,
        platform_gap=float(platform_gap),
        buyer_average_utility=buyer_avg,
        buyer_average_baseline=buyer_baseline_avg,
        buyer_average_gap=float(buyer_gap),
        pass_epsilon=bool(pass_epsilon),
    )


def pass_at_epsilon(reports: Mapping[str, UtilityGapReport], epsilon: float = 0.05) -> float:
    """统计多组实验中通过 epsilon 阈值的比例。"""

    if not reports:
        return 0.0
    passed = 0
    for report in reports.values():
        if abs(report.platform_gap) <= epsilon and abs(report.buyer_average_gap) <= epsilon:
            passed += 1
    return float(passed / len(reports))


def make_summary_table(
    results: Mapping[str, EpisodeRunResult],
    *,
    epsilon: float = 0.05,
    reference_mode: str = "all_rule",
) -> list[Dict[str, Any]]:
    """生成模式级汇总表。

    每一行对应一种实验模式，用于比较平台效用、买家效用和相对规则基准的 Utility Gap。
    """

    ref_name, ref_result = _reference_result(results, reference_mode)
    table = []
    for name, result in results.items():
        report = utility_gap(
            result,
            ref_result,
            label=name,
            reference_label=ref_name,
            epsilon=epsilon,
        )
        table.append(
            {
                "mode": name,
                "reference_mode": ref_name,
                "rounds": int(_lookup_number(result.summary, "总轮次")),
                "platform_utility": report.platform_utility,
                "platform_baseline": report.platform_baseline,
                "platform_gap": report.platform_gap,
                "buyer_average_utility": report.buyer_average_utility,
                "buyer_average_baseline": report.buyer_average_baseline,
                "buyer_average_gap": report.buyer_average_gap,
                "pass_epsilon": report.pass_epsilon,
            }
        )
    return table


def make_step_table(results: Mapping[str, EpisodeRunResult]) -> list[Dict[str, Any]]:
    """生成逐轮轨迹表。

    该表按真实交易轮次记录，而不是按 Agent 决策步记录。
    因此 all_rule 和混合模式中的规则侧行为也会被纳入同一条时间轴。
    """

    rows = []
    for mode, result in results.items():
        for item in result.rounds:
            rows.append(
                {
                    "mode": mode,
                    "step_index": item.round_index,
                    "round_index": item.round_index,
                    "buyer_id": item.buyer_id,
                    "price": item.price,
                    "bid": item.bid,
                    "buyer_utility": item.buyer_utility,
                    "platform_utility": item.platform_utility,
                    "seller_utilities": item.seller_utilities,
                    "platform_revenue": item.platform_revenue,
                    "gain": item.gain,
                }
            )
    return rows


def make_single_step_table(mode: str, result: EpisodeRunResult) -> list[Dict[str, Any]]:
    """生成单个实验模式的逐轮轨迹表。

    单模式日志不再重复保存 mode 字段，便于人工查看。
    """

    rows = []
    for item in result.rounds:
        rows.append(
            {
                "step_index": item.round_index,
                "round_index": item.round_index,
                "buyer_id": item.buyer_id,
                "price": item.price,
                "bid": item.bid,
                "buyer_utility": item.buyer_utility,
                "platform_utility": item.platform_utility,
                "seller_utilities": item.seller_utilities,
                "platform_revenue": item.platform_revenue,
                "gain": item.gain,
            }
        )
    return rows


def make_decision_step_table(result: EpisodeRunResult) -> list[Dict[str, Any]]:
    """生成 Agent 决策日志表，用于检查 Agent 看到什么、输出什么。"""

    rows = []
    for step in result.steps:
        rows.append(
            {
                "step_index": step.step_index,
                "role": step.role,
                "observation": json.dumps(step.observation, ensure_ascii=False),
                "action": json.dumps(step.action, ensure_ascii=False),
                "reward": json.dumps(step.reward, ensure_ascii=False),
            }
        )
    return rows


def make_chart_payload(results: Mapping[str, EpisodeRunResult]) -> Dict[str, Any]:
    """生成画图脚本使用的紧凑时间序列数据。"""

    payload: Dict[str, Any] = {}
    for mode, result in results.items():
        series = {
            "step_index": [],
            "price": [],
            "bid": [],
            "gain": [],
            "platform_revenue": [],
            "buyer_utility": [],
            "platform_utility": [],
        }
        for item in result.rounds:
            series["step_index"].append(item.round_index)
            series["price"].append(item.price)
            series["bid"].append(item.bid)
            series["gain"].append(item.gain)
            series["platform_revenue"].append(item.platform_revenue)
            series["buyer_utility"].append(item.buyer_utility)
            series["platform_utility"].append(item.platform_utility)
        payload[mode] = series
    return payload


def compare_experiments(
    results: Mapping[str, EpisodeRunResult],
    *,
    epsilon: float = 0.05,
    reference_mode: str = "all_rule",
) -> Dict[str, Any]:
    """汇总多组实验，返回可 JSON 序列化的比较结果。"""

    ref_name, ref_result = _reference_result(results, reference_mode)
    reports = {
        name: utility_gap(
            result,
            ref_result,
            label=name,
            reference_label=ref_name,
            epsilon=epsilon,
        )
        for name, result in results.items()
    }
    return {
        "epsilon": float(epsilon),
        "reference_mode": ref_name,
        "pass_at_epsilon": pass_at_epsilon(reports, epsilon=epsilon),
        "summary_table": make_summary_table(
            results,
            epsilon=epsilon,
            reference_mode=reference_mode,
        ),
        "step_table": make_step_table(results),
        "chart_payload": make_chart_payload(results),
        "reports": {
            name: {
                "label": report.label,
                "reference_label": report.reference_label,
                "platform_utility": report.platform_utility,
                "platform_baseline": report.platform_baseline,
                "platform_gap": report.platform_gap,
                "buyer_average_utility": report.buyer_average_utility,
                "buyer_average_baseline": report.buyer_average_baseline,
                "buyer_average_gap": report.buyer_average_gap,
                "pass_epsilon": report.pass_epsilon,
            }
            for name, report in reports.items()
        },
    }


def save_analysis(
    results: Mapping[str, EpisodeRunResult],
    output_dir: str | Path,
    *,
    epsilon: float = 0.05,
    reference_mode: str = "all_rule",
) -> Dict[str, str]:
    """保存实验原始结果、分析 JSON 和 CSV 表格。"""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / "baseline_results.json"
    analysis_path = out / "analysis_summary.json"
    summary_csv_path = out / "summary_table.csv"
    step_csv_path = out / "step_table.csv"
    modes_dir = out / "modes"

    with raw_path.open("w", encoding="utf-8") as f:
        json.dump(
            {name: episode_to_dict(result) for name, result in results.items()},
            f,
            ensure_ascii=False,
            indent=2,
        )
    with analysis_path.open("w", encoding="utf-8") as f:
        json.dump(
            compare_experiments(
                results,
                epsilon=epsilon,
                reference_mode=reference_mode,
            ),
            f,
            ensure_ascii=False,
            indent=2,
        )

    summary_rows = make_summary_table(
        results,
        epsilon=epsilon,
        reference_mode=reference_mode,
    )
    step_rows = make_step_table(results)
    if summary_rows:
        with summary_csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
    if step_rows:
        with step_csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(step_rows[0].keys()))
            writer.writeheader()
            writer.writerows(step_rows)

    mode_paths: Dict[str, Dict[str, str]] = {}
    modes_dir.mkdir(parents=True, exist_ok=True)
    for mode, result in results.items():
        mode_dir = modes_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        mode_json_path = mode_dir / "episode.json"
        mode_rounds_path = mode_dir / "rounds.csv"
        mode_decision_steps_path = mode_dir / "decision_steps.csv"
        mode_summary_path = mode_dir / "summary.json"

        with mode_json_path.open("w", encoding="utf-8") as f:
            json.dump(episode_to_dict(result), f, ensure_ascii=False, indent=2)
        with mode_summary_path.open("w", encoding="utf-8") as f:
            json.dump(result.summary, f, ensure_ascii=False, indent=2)

        mode_step_rows = make_single_step_table(mode, result)
        round_fieldnames = [
            "step_index",
            "round_index",
            "buyer_id",
            "price",
            "bid",
            "buyer_utility",
            "platform_utility",
            "seller_utilities",
            "platform_revenue",
            "gain",
        ]
        with mode_rounds_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=round_fieldnames)
            writer.writeheader()
            writer.writerows(mode_step_rows)

        mode_decision_rows = make_decision_step_table(result)
        with mode_decision_steps_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["step_index", "role", "observation", "action", "reward"],
            )
            writer.writeheader()
            writer.writerows(mode_decision_rows)
        mode_paths[mode] = {
            "episode": str(mode_json_path),
            "summary": str(mode_summary_path),
            "rounds": str(mode_rounds_path),
            "decision_steps": str(mode_decision_steps_path),
        }

    return {
        "raw": str(raw_path),
        "analysis": str(analysis_path),
        "summary_csv": str(summary_csv_path),
        "step_csv": str(step_csv_path),
        "modes_dir": str(modes_dir),
        "mode_files": mode_paths,
    }
