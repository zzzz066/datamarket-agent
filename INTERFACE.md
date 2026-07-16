# 对接接口文档

## 一、项目概述

基于 Agarwal, Dahleh, and Sarkar (2019) *A Marketplace for Data: An Algorithmic Solution* 的数据市场 Agent 模拟平台。

**核心问题**：论文证明了 Myerson 机制下诚实报价 b=mu 是占优策略——但这是假设平台价格外生给定。在实际动态博弈中，买家的策略性报价可能通过 PF(MWU) 的权重更新**间接操纵未来价格**。Agent 实验的目标就是观察双边 LLM Agent 博弈下，理论机制是否依然稳健。

## 二、目录结构

```
marketplace_for_data_agent/
├── mechanism/               # 论文四大机制（不改动）
│   ├── __init__.py
│   └── core.py              # AF / RF / PF / PD + 闭式 Ground Truth
│
├── env/                     # 环境层
│   ├── __init__.py
│   ├── base.py              # MarketplaceForDataEnv — 底层规则环境
│   └── wrapper.py           # AgentMarketEnv — Agent 可调用的封装
│
├── agents/                  # Agent 层（Agent 开发者工作在这里）
│   ├── __init__.py
│   └── base.py              # BaseAgent / LLMAgent / 规则 Agent
│
├── experiments/             # 实验脚本（待实现）
│   └── __init__.py
│
├── simulation.py            # Dashboard 后端（原有，不变）
├── server.py                # HTTP 服务（原有，不变）
├── dashboard.html           # 可视化前端（原有，不变）
├── demo.py                  # 最小示例（原有，不变）
├── README.md
└── INTERFACE.md             # 本文档
```

## 三、已实现内容

### 机制层 `mechanism/core.py`

| 组件 | 函数/类 | 论文对应 |
|---|---|---|
| AF | `data_allocation_function(p, b, X)` | 数据分配：b<p 时加噪退化 |
| RF | `myerson_revenue_function(p, b)` | Myerson 支付：b·h(b)-∫h |
| PF | `PriceUpdateState` | MWU 价格更新状态 |
| PD | `robust_payment_division(revenue, X)` | Shapley + 余弦相似度惩罚 |
| GT | `closed_form_ground_truth(mus, bounds)` | 理论均衡基准 |

### 环境层 `env/`

| 类 | 用途 |
|---|---|
| `MarketplaceForDataEnv` | 底层规则环境，逐轮 step(buyer) |
| `AgentMarketEnv` | Agent 封装，拆成 `step("平台", ...)` + `step("买家", ...)` |

### Agent 层 `agents/base.py`

| 类 | 用途 |
|---|---|
| `BaseAgent` | 抽象基类，定义 `act(obs) → action` |
| `RuleBasedPlatformAgent` | MWU 规则平台（基线对照） |
| `TruthfulBuyerAgent` | 诚实买家 b=mu（基线对照） |
| `ShadeBuyerAgent` | 低报买家 b=0.65·mu |
| `OverbidBuyerAgent` | 高报买家 b=1.4·mu |
| `LLMAgent` | LLM Agent 基类（OpenAI 兼容 API） |

## 四、Agent 开发者对接指南

### 你只需要知道一个类和两个方法

```python
from marketplace_for_data_agent import AgentMarketEnv, BaseAgent

# 1. 构造环境
env = AgentMarketEnv(
    sellers, buyers,
    platform_mode="agent",   # 平台由 Agent 决策
    buyer_mode="agent",      # 买家由 Agent 决策
    seed=42,
)

# 2. 跑一个 episode
obs = env.reset()
while not env.done:
    if env.current_role == "平台":
        action = platform_agent.act(obs)   # → {"价格": 0.8}
        obs = env.step("平台", action)
    elif env.current_role == "买家":
        action = buyer_agent.act(obs)      # → {"报价": 1.2}
        obs = env.step("买家", action)

# 3. 拿结果
summary = env.episode_summary()  # 累积效用
gt = env.ground_truth()          # 理论基准
```

### 观察字典格式（你的 Agent 的 act() 会收到这个）

**平台 Agent 收到：**
```python
{
    "当前轮次": 5,
    "总轮次": 120,
    "候选价格": [0.1, 0.15, 0.2, ..., 1.6],
    "MWU权重": [0.008, 0.012, ...],     # 归一化后的权重分布
    "MWU推荐价格": 0.8,                  # argmax(权重)
    "卖家数量": 3,
    "最近历史": [
        {"价格": 0.7, "报价": 0.8, "平台收入": 0.35, "平台效用": 0.29, ...},
        ...
    ],
}
```

**买家 Agent 收到：**
```python
{
    "当前轮次": 5,
    "总轮次": 120,
    "我的估值_mu": 0.8,        # 你对数据的真实估值
    "当前平台价格": 0.7,        # 平台本轮定的价格
    "我的历史": [               # 你过去各轮的决策和结果
        {"报价": 0.8, "效用": 0.35, "增益": 0.98},
        ...
    ],
}
```

### 动作字典格式（你的 Agent 必须返回这个）

**平台 Agent 返回：**
```python
{"价格": 0.8}   # float，范围 [price_lower, price_upper]
```

**买家 Agent 返回：**
```python
{"报价": 1.2}   # float，必须 ≥ 0
```

### 四种实验模式

| platform_mode | buyer_mode | 谁决策 | 用途 |
|---|---|---|---|
| `"mwu"` | `"truthful"` | 全部规则 | 论文原版基线 |
| `"mwu"` | `"agent"` | 买家 Agent | 测试买家策略行为 |
| `"agent"` | `"truthful"` | 平台 Agent | 测试平台定价能力 |
| `"agent"` | `"agent"` | 双方 Agent | 双边博弈 |

### 写一个 Agent 的最小代码

```python
from marketplace_for_data_agent.agents.base import LLMAgent

class MyBuyerAgent(LLMAgent):
    def __init__(self):
        super().__init__(
            model="gpt-4o-mini",
            system_prompt="""你是一个数据市场的买家。
每轮你会看到平台价格 p 和自己的真实估值 mu。
你的目标是最大化效用：效用 = mu * 数据质量 - Myerson支付。
规则：报 b≥p 得完整数据但支付固定；报 b<p 数据降质但支付减少。
诚实报价 b=mu 在理论上是最优的。
请根据当前观察决定你的报价。""",
        )

    def _format_obs(self, obs):
        # 把观察字典转为自然语言，放进 LLM prompt
        return f"""当前平台价格: {obs['当前平台价格']}
我的估值 mu: {obs['我的估值_mu']}
我的历史: {obs['我的历史']}
请输出你的报价。"""

# 使用
agent = MyBuyerAgent()
action = agent.act(obs)  # → {"报价": 0.8}
```

## 五、机制关键性质（设计备忘）

### 诚实占优的成立条件

在**价格 p 外生给定**的前提下，买家效用函数：

```
U(b) = mu · min(b/p, 1) - RF(p, b)

当 b < p: U(b) = mu·b/p - b²/(2p)    → dU/db = (mu-b)/p → 最优 b=mu
当 b ≥ p: U(b) = mu - p/2             → 与 b 无关，报更高不会更好
```

所以单个买家偏离诚实不会获得更高效用。

### 动态博弈下的潜在漏洞

但 MWU 的价格更新 `weights[i] *= 1 + delta * RF(c_i, b)` 意味着：
- 买家报低价 → MWU 感知到的收入低 → 权重向低价偏移
- 后续买家面对更低的价格 → 形成了"集体压价"的正反馈

**如果全部买家联合低报，可以压低整个价格轨迹，获得比诚实更高的集体效用。**
但在单买方偏离的视角下，个体的低报不影响价格（价格由历史决定），
所以诚实仍是单边最优——这是典型的"囚徒困境"结构。

Agent 实验的核心问题即：**LLM Agent 能否发现并利用这个集体行动漏洞？**
