"""实验模块。

运行 Agent 模拟实验、评估指标、结果分析。
"""

from .analysis import (
    UtilityGapReport,
    compare_experiments,
    make_chart_payload,
    make_step_table,
    make_summary_table,
    pass_at_epsilon,
    save_analysis,
    utility_gap,
)
from .plotting import (
    load_analysis_json,
    load_csv_table,
    plot_standard_charts,
    plot_trajectory,
    plot_utility_gap,
)
from .simulator import (
    EpisodeRunResult,
    EpisodeStepLog,
    episode_to_dict,
    run_baseline_suite,
    run_four_mode_suite,
    run_episode,
    save_episode_logs,
)

__all__ = [
	"EpisodeRunResult",
	"EpisodeStepLog",
	"UtilityGapReport",
	"compare_experiments",
	"episode_to_dict",
	"make_chart_payload",
	"make_step_table",
	"make_summary_table",
	"pass_at_epsilon",
	"load_analysis_json",
	"load_csv_table",
	"plot_standard_charts",
	"plot_trajectory",
	"plot_utility_gap",
	"run_baseline_suite",
	"run_four_mode_suite",
	"run_episode",
	"save_analysis",
	"save_episode_logs",
	"utility_gap",
]
