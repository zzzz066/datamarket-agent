"""四模式实验入口脚本。

该脚本负责：
1. 构造默认 sellers / buyers。
2. 根据 buyers 数量决定交易轮数。
3. 可选接入外部 seller_agent / buyer_agent。
4. 运行 all_rule、seller_agent、buyer_agent、both_agent 四类模式。
5. 保存分析结果，并在 matplotlib 可用时生成图表。

运行示例：

    python experiments/run_four_modes.py --rounds 50 --output-dir outputs/exp_001

接入同伴实现的 Agent：

    python experiments/run_four_modes.py \
        --seller-agent my_agents:PlatformAgent \
        --buyer-agent my_agents:BuyerAgent \
        --rounds 50 \
        --plot \
        --output-dir outputs/exp_agents
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _load_package_when_run_as_script() -> Any:
    """允许直接用 python experiments/run_four_modes.py 运行。

    项目目录名含有连字符时，直接按普通包名导入会不方便；
    这里在脚本模式下为当前目录注册一个临时包名。
    """

    root = Path(__file__).resolve().parents[1]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    package_name = "marketplace_for_data_agent"
    if package_name in sys.modules:
        return sys.modules[package_name]
    spec = importlib.util.spec_from_file_location(
        package_name,
        root / "__init__.py",
        submodule_search_locations=[str(root)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载项目包。")
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = pkg
    spec.loader.exec_module(pkg)
    return pkg


try:
    from ..env import Buyer, Seller
    from .analysis import save_analysis
    from .plotting import plot_standard_charts
    from .simulator import run_four_mode_suite
except ImportError:
    pkg = _load_package_when_run_as_script()
    Buyer = pkg.Buyer
    Seller = pkg.Seller
    save_analysis = pkg.save_analysis
    plot_standard_charts = pkg.plot_standard_charts
    run_four_mode_suite = pkg.run_four_mode_suite


def load_agent(spec: str | None) -> Any:
    """从 module:object 或 module.object 字符串加载外部 Agent。

    如果加载到的是类，则直接无参实例化；如果是函数或已有对象，则原样返回。
    """

    if not spec:
        return None
    module_name, sep, attr_name = spec.partition(":")
    if not sep:
        module_name, _, attr_name = spec.rpartition(".")
    if not module_name or not attr_name:
        raise ValueError("Agent 路径格式应为 module:object 或 module.object。")
    module = importlib.import_module(module_name)
    obj = getattr(module, attr_name)
    if isinstance(obj, type):
        return obj()
    return obj


def build_default_sellers(sample_count: int = 80) -> tuple[list[Seller], np.ndarray]:
    """构造默认卖家数据池，并返回预测目标 y。

    这里保留一个重复卖家 A_copy，方便后续观察 PD 中重复数据惩罚的影响。
    """

    t = np.linspace(0.0, 1.0, int(sample_count))
    y = np.sin(2.0 * np.pi * t)
    sellers = [
        Seller("A", y + 0.05 * np.cos(6.0 * np.pi * t), cost=0.02),
        Seller("B", np.cos(2.0 * np.pi * t), cost=0.02),
        Seller("A_copy", y + 0.05 * np.cos(6.0 * np.pi * t), cost=0.02),
    ]
    return sellers, y


def build_buyers(
    y: np.ndarray,
    *,
    rounds: int,
    seed: int,
    buyer_id_mode: str = "distinct",
    mu_mode: str = "linspace",
    mu_low: float = 0.3,
    mu_high: float = 1.3,
) -> list[Buyer]:
    """根据实验参数构造买家序列。

    rounds 控制 buyers 列表长度，也就是 episode 的交易轮数。
    buyer_id_mode="same" 时表示同一个买家重复交易；
    buyer_id_mode="distinct" 时表示不同买家依次到达。
    """

    rng = np.random.default_rng(seed)
    rounds = int(max(1, rounds))
    if mu_mode == "linspace":
        mus = np.linspace(float(mu_low), float(mu_high), rounds)
    elif mu_mode == "random":
        mus = rng.uniform(float(mu_low), float(mu_high), size=rounds)
    elif mu_mode == "cycle":
        base = np.array([0.4, 0.8, 1.2], dtype=float)
        mus = np.resize(base, rounds)
    else:
        raise ValueError(f"未知 mu_mode: {mu_mode}")

    buyers = []
    for i, mu in enumerate(mus):
        buyer_id = "buyer_A" if buyer_id_mode == "same" else f"buyer_{i + 1}"
        buyers.append(Buyer(buyer_id, y, mu=float(mu)))
    return buyers


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Run four-mode marketplace experiments.")
    parser.add_argument("--rounds", type=int, default=30, help="交易轮数，即 buyers 长度。")
    parser.add_argument(
        "--buyer-id-mode",
        choices=["distinct", "same"],
        default="distinct",
        help="distinct 表示不同买家依次到达；same 表示同一个买家重复交易。",
    )
    parser.add_argument(
        "--mu-mode",
        choices=["linspace", "random", "cycle"],
        default="random",
        help="买家估值 mu 的生成方式。",
    )
    parser.add_argument("--mu-low", type=float, default=0.3)
    parser.add_argument("--mu-high", type=float, default=1.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--output-dir", type=str, default="outputs/four_modes")
    parser.add_argument("--seller-agent", type=str, default=None)
    parser.add_argument("--buyer-agent", type=str, default=None)
    parser.add_argument(
        "--expose-mwu-to-seller-agent",
        action="store_true",
        help="默认不向平台/卖方定价侧 Agent 暴露 MWU；设置该参数后才暴露。",
    )
    parser.add_argument("--price-lower", type=float, default=0.1)
    parser.add_argument("--price-upper", type=float, default=1.6)
    parser.add_argument("--price-step", type=float, default=0.05)
    parser.add_argument("--delta", type=float, default=0.18)
    parser.add_argument("--af-mode", choices=["gaussian", "masking"], default="gaussian")
    parser.add_argument("--noise-sigma", type=float, default=1.0)
    parser.add_argument("--shapley-permutations", type=int, default=64)
    parser.add_argument("--lambda-penalty", type=float, default=0.7)
    parser.add_argument("--plot", action="store_true", help="如果 matplotlib 可用，则生成标准图表。")
    parser.add_argument("--verbose", action="store_true", help="打印每个模式、轮次和 Agent 调用进度。")
    return parser.parse_args()


def main() -> None:
    """运行实验、保存分析结果，并按需生成图表。"""

    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sellers, y = build_default_sellers()
    buyers = build_buyers(
        y,
        rounds=args.rounds,
        seed=args.seed,
        buyer_id_mode=args.buyer_id_mode,
        mu_mode=args.mu_mode,
        mu_low=args.mu_low,
        mu_high=args.mu_high,
    )
    seller_agent = load_agent(args.seller_agent)
    buyer_agent = load_agent(args.buyer_agent)

    results = run_four_mode_suite(
        sellers,
        buyers,
        seller_agent=seller_agent,
        buyer_agent=buyer_agent,
        seed=args.seed,
        expose_mwu_to_seller_agent=args.expose_mwu_to_seller_agent,
        price_bounds=(args.price_lower, args.price_upper, args.price_step),
        delta=args.delta,
        af_mode=args.af_mode,
        noise_sigma=args.noise_sigma,
        shapley_permutations=args.shapley_permutations,
        lambda_penalty=args.lambda_penalty,
        verbose=args.verbose,
    )
    paths = save_analysis(results, out, epsilon=args.epsilon)

    print(f"交易轮数: {len(buyers)}")
    print(f"运行模式: {', '.join(results.keys())}")
    print("输出文件:")
    for name, path in paths.items():
        if name == "mode_files":
            continue
        print(f"  {name}: {path}")

    if args.plot:
        try:
            from .analysis import make_step_table, make_summary_table
        except ImportError:
            pkg = _load_package_when_run_as_script()
            make_step_table = pkg.make_step_table
            make_summary_table = pkg.make_summary_table
        try:
            figure_paths = plot_standard_charts(
                make_summary_table(results, epsilon=args.epsilon),
                make_step_table(results),
                out / "figures",
                epsilon=args.epsilon,
            )
        except ModuleNotFoundError as exc:
            if exc.name == "matplotlib":
                print("未安装 matplotlib，已跳过画图。")
                return
            raise
        print("图表文件:")
        for name, path in figure_paths.items():
            print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
