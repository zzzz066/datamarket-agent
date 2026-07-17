# 数据市场 Agent 模拟平台

基于 Agarwal, Dahleh, and Sarkar (2019) *A Marketplace for Data: An Algorithmic Solution*
实现的数据市场数值环境与 Agent 模拟系统。

## 目录结构

```text
marketplace_for_data_agent/
├── mechanism/               # 论文四大机制实现
│   └── core.py              # AF / RF / PF / PD + 闭式 Ground Truth
├── env/                     # 环境层
│   ├── base.py              # MarketplaceForDataEnv — 规则结算引擎
│   └── wrapper.py           # AgentMarketEnv — Agent 决策封装
├── agents/                  # Agent 接口、规则 Agent、LLM Agent、JSON 动作解析器
│   └── base.py              # BaseAgent / LLMAgent / ActionParser / 规则基线
├── experiments/             # 实验脚本
├── demo.py                  # 最小运行示例
├── simulation.py            # Dashboard 后端
├── server.py                # HTTP 服务
└── dashboard.html           # 可视化前端
```

## 论文机制 → 代码对应

论文描述了一个数据市场，包含三种参与者（平台、买家、卖家）和四个机制组件：

| 论文机制 | 代码实现 | 做了什么 |
|---|---|---|
| AF | `data_allocation_function(p, b, X)` | 买家报价 b ≥ 平台价格 p 时给完整数据；b < p 时加高斯噪声退化 |
| RF | `myerson_revenue_function(p, b)` | Myerson 支付规则：b·h(b) − ∫₀ᵇ h(z)dz，保证诚实报价占优 |
| PF | `PriceUpdateState` | 候选价格网格上的乘法权重（MWU）更新，平台实现无遗憾定价 |
| PD | `robust_payment_division(revenue, X)` | Shapley 边际贡献 + 余弦相似度指数惩罚，抑制数据复制 |
| GT | `closed_form_ground_truth(mus, …)` | 假设所有买家诚实 + 平台定最优价 p* 时的理论均衡基准 |

## env 层的两层设计

### base.py — 结算引擎

`MarketplaceForDataEnv` 实现**一轮交易的完整结算**，不关心决策是谁做的：

```python
# 输入：买家 + 报价 + 价格
result = env.step(buyer, bid=0.8, external_price=1.2)

# 内部依次执行（与论文 Algorithm 1 完全对应）：
# ① AF:   data_allocation_function(price, bid, X)   → 分配数据（可能加噪）
# ② Gain: OLS 拟合 y，计算 1−RMSE 增益                → 衡量数据质量
# ③ RF:   myerson_revenue_function(price, bid)       → 计算支付
# ④ PD:   robust_payment_division(revenue, X)        → 分配给卖家
# ⑤ PF:   price_state.update(bid)                    → 更新 MWU 权重

# 输出：StepResult（价格、报价、增益、收入、三方效用、分配详情）
```

每一步的输入输出都是确定性的——给定相同的 `(price, bid, seller_features)`，结果完全可复现。

### wrapper.py — Agent 决策封装

`AgentMarketEnv` 在结算引擎外面包了一层**决策流控制**，把一轮拆成两个独立的决策点：

```text
Round N:
  ┌─ 该平台了 ─→ obs = get_obs("平台")  ─→ Agent 返回 {"价格": p}
  │
  ├─ 该买家了 ─→ obs = get_obs("买家")  ─→ Agent 返回 {"报价": b}
  │
  └─ 结算 ────→ base.step(buyer, bid=b, external_price=p) → 下一轮
```

两种模式通过构造参数切换，不需要改代码：

| platform_mode | buyer_mode | 含义 |
|---|---|---|
| `"mwu"` | `"truthful"` | 纯规则基线（等于论文原版） |
| `"mwu"` | `"agent"` | 只测买家 Agent |
| `"agent"` | `"truthful"` | 只测平台 Agent |
| `"agent"` | `"agent"` | 双方都是 Agent 博弈 |

**规则模式下，wrapper 内部自动完成决策并跳过**，只有 `"agent"` 模式才会把观察暴露出来等待外部输入。

### 观察翻译

wrapper 的另一个职责是把内部状态（numpy 数组、StepResult 对象）翻译成 Agent 可读的 dict：

**平台 Agent 看到：**
```python
{
    "当前轮次": 5,
    "总轮次": 120,
    "候选价格": [0.1, 0.15, ..., 1.6],
    "MWU权重": [0.008, 0.012, ...],    # 归一化概率
    "MWU推荐价格": 0.8,                # argmax(权重)
    "卖家数量": 3,
    "最近历史": [{"价格": …, "报价": …, "平台收入": …}, ...],
}
```

**买家 Agent 看到：**
```python
{
    "当前轮次": 5,
    "总轮次": 120,
    "我的估值_mu": 0.8,
    "当前平台价格": 0.7,
    "我的历史": [{"报价": …, "效用": …, "增益": …}, ...],
}
```

## Agent 层设计

`agents/base.py` 提供三类能力：

| 类/函数 | 用途 |
|---|---|
| `BaseAgent` | 所有 Agent 的统一接口，子类实现 `act(obs) -> action` |
| `RuleBasedPlatformAgent` | 平台规则基线，直接采用 `MWU推荐价格` |
| `TruthfulBuyerAgent` | 买家诚实报价基线，返回 `报价 = 我的估值_mu` |
| `ShadeBuyerAgent` | 买家低报基线，用于观察动态压价效应 |
| `OverbidBuyerAgent` | 买家高报基线，用于对照实验 |
| `ActionParser` | 解析 LLM 输出，支持 JSON、Markdown 代码块、解释文本中夹带 JSON、英文别名 `price/bid` |
| `OpenAICompatibleClient` | 标准库实现的 OpenAI-compatible `/chat/completions` 客户端 |
| `LLMAgent` | 角色感知、带短期记忆、机制引导反思的 LLM Agent |

LLM Agent 的动作最终会被规范化为环境需要的格式：

```python
{"价格": 0.8}   # 平台 Agent
{"报价": 1.2}   # 买家 Agent
```

### Prompt profiles

`LLMAgent` 支持通过 `prompt_profile` 切换实验人格/目标函数，便于比较不同 prompt 设计对市场结果的影响：

| profile | 适用角色 | 含义 |
|---|---|---|
| `balanced` | 平台/买家 | 默认设置，兼顾理论基线、历史经验和风险 |
| `strategic` | 平台/买家 | 允许考虑当前动作对未来价格轨迹的影响 |
| `welfare` | 平台/买家 | 更关注市场总福利和长期稳定 |
| `risk_averse` | 平台/买家 | 更保守，优先避免负效用或极端动作 |
| `truthful` | 买家 | 强调诚实报价 `b=mu` 作为基线 |
| `shade` | 买家 | 允许适度低报，观察动态压价效应 |
| `revenue_max` | 平台 | 更强调平台收入和平台效用 |
| `fairness_aware` | 平台 | 同时关注买家效用和卖家参与激励 |

默认 prompt 包含四部分：角色目标、信息边界、机制说明、JSON 输出 schema。每轮 user prompt 会注入当前观察和短期记忆，要求模型只输出 JSON，例如：

```json
{"reasoning": "参考历史报价和MWU推荐价，选择稳健价格", "价格": 0.8}
```

买家输出：

```json
{"reasoning": "固定价格下诚实报价接近理论最优", "报价": 0.8}
```

## 配置环境

推荐使用 conda 单独建环境：

```bash
conda create -n datamarket-agent python=3.12 numpy -y
conda activate datamarket-agent
```

如果后续运行 Dashboard 或画图脚本时缺少依赖，可以补装常用科学计算包：

```bash
conda install scipy matplotlib -y
```

本项目目录名是 `datamarket-agent`，包含连字符，Python 不能直接把它当包名导入。推荐在父目录建立一个合法包名软链接：

```bash
cd /mnt/data/home/yangxinyi
ln -sfn datamarket-agent datamarket_agent
```

之后都从父目录运行包命令：

```bash
cd /mnt/data/home/yangxinyi
python3 -m datamarket_agent.demo
```

也可以留在项目目录里运行，但需要显式设置 `PYTHONPATH`：

```bash
cd /mnt/data/home/yangxinyi/datamarket-agent
PYTHONPATH=/mnt/data/home/yangxinyi python3 -m datamarket_agent.demo
```

## 如何运行

### 1. 运行测试

```bash
conda activate datamarket-agent
cd /mnt/data/home/yangxinyi
python3 -m unittest discover -s datamarket_agent/tests
```

### 2. 运行最小机制示例

```bash
conda activate datamarket-agent
cd /mnt/data/home/yangxinyi
python3 -m datamarket_agent.demo
```

输出会包含每个买家的价格、报价、数据增益、平台收入、卖家分成，以及理论基准 `ground_truth`。可用于对比实际模拟和理论最优结果。

### 3. 运行 Dashboard 可视化

```bash
conda activate datamarket-agent
cd /mnt/data/home/yangxinyi
python3 -m datamarket_agent.server
```

浏览器打开：

```text
http://127.0.0.1:8000/
```

三张图分别验证：诚实报价占优、MWU 价格收敛、复制惩罚生效。

## 设置 LLM API

`LLMAgent` 默认读取两个环境变量：

```bash
export OPENAI_API_KEY="你的_api_key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

`OPENAI_BASE_URL` 可省略，默认就是 `https://api.openai.com/v1`。如果使用第三方 OpenAI 兼容服务、本地模型网关或中转服务，把它改成对应地址即可：

```bash
export OPENAI_API_KEY="你的_key"
export OPENAI_BASE_URL="https://你的服务地址/v1"
```

不要把 API key 写进 git 提交。临时运行可直接在当前终端 `export`；长期使用可以写入 `~/.zshrc`：

```bash
echo 'export OPENAI_API_KEY="你的_api_key"' >> ~/.zshrc
source ~/.zshrc
```

也可以在代码里显式传入 client：

```python
from datamarket_agent.agents import LLMAgent, OpenAICompatibleClient

client = OpenAICompatibleClient(
    api_key="你的_api_key",
    base_url="https://api.openai.com/v1",
)

buyer_agent = LLMAgent(client=client, model="gpt-4o-mini", role="买家")
```

## Agent 实验示例

规则 Agent 示例，不需要 API key：

```python
from datamarket_agent import AgentMarketEnv
from datamarket_agent.agents import RuleBasedPlatformAgent, TruthfulBuyerAgent

platform_agent = RuleBasedPlatformAgent()
buyer_agent = TruthfulBuyerAgent()

env = AgentMarketEnv(sellers, buyers, platform_mode="agent", buyer_mode="agent")
obs = env.reset()
while not env.done:
    if env.current_role == "平台":
        obs = env.step("平台", platform_agent.act(obs))
    elif env.current_role == "买家":
        obs = env.step("买家", buyer_agent.act(obs))

print(env.episode_summary())
print(env.ground_truth())
```

LLM Agent 示例，需要先设置 `OPENAI_API_KEY`：

```python
from datamarket_agent import AgentMarketEnv
from datamarket_agent.agents import LLMAgent

platform_agent = LLMAgent(model="gpt-4o-mini", role="平台", prompt_profile="revenue_max")
buyer_agent = LLMAgent(model="gpt-4o-mini", role="买家", prompt_profile="strategic")

env = AgentMarketEnv(sellers, buyers, platform_mode="agent", buyer_mode="agent")
obs = env.reset()
while not env.done:
    if env.current_role == "平台":
        obs = env.step("平台", platform_agent.act(obs))
    elif env.current_role == "买家":
        obs = env.step("买家", buyer_agent.act(obs))

print(env.episode_summary())
print(env.ground_truth())
```

`LLMAgent` 会要求模型输出 JSON，并通过 `ActionParser` 解析为 `{"价格": ...}` 或 `{"报价": ...}`。如果模型输出 Markdown 代码块或先解释再给 JSON，解析器也会尽量抽取有效动作；解析失败时默认回退到规则动作，避免一次坏输出中断实验。

## 机制性质

在价格 p 外生给定的前提下，买家效用函数为：

```text
当 b < p:  U(b) = mu·b/p − b²/(2p)    → dU/db = (mu−b)/p → 最优 b=mu
当 b ≥ p:  U(b) = mu − p/2             → 与 b 无关，报更高不会更好
```

因此**单买家面对固定价格时，诚实报价 b=mu 是占优策略**。

但在动态博弈中，MWU 的价格更新 `weights[i] *= 1 + delta·RF(cᵢ, b)` 意味着：
买家如果集体低报，MWU 感知到的收入变低，权重向低价偏移，后续价格下降。


详见 `INTERFACE.md`。

## 模拟实验方法

```
python experiments\run_four_modes.py `
   --rounds 5 `
   --seller-agent agents.base:LLMAgent `
   --buyer-agent agents.base:LLMAgent `
   --output-dir outputs\llm_test `
   --verbose
```