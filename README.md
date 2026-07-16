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
├── agents/                  # Agent 接口
│   └── base.py              # BaseAgent 抽象基类
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

## 运行

### 最小示例（验证机制层）

```powershell
python -m marketplace_for_data_agent.demo
```

### Dashboard 可视化

```powershell
python -m marketplace_for_data_agent.server
# 浏览器打开 http://127.0.0.1:8000/
```

三张图分别验证：诚实报价占优、MWU 价格收敛、复制惩罚生效。

### Agent 实验（Agent 开发者用）

```python
from marketplace_for_data_agent import AgentMarketEnv, BaseAgent

class MyBuyer(BaseAgent):
    def act(self, obs):
        # obs 是上面的买家观察 dict
        return {"报价": obs["我的估值_mu"]}  # 例如：诚实报价

env = AgentMarketEnv(sellers, buyers, platform_mode="mwu", buyer_mode="agent")
obs = env.reset()
while not env.done:
    if env.current_role == "平台":
        obs = env.step("平台", {"价格": obs["MWU推荐价格"]})
    elif env.current_role == "买家":
        obs = env.step("买家", agent.act(obs))

print(env.episode_summary())   # 累积效用
print(env.ground_truth())      # 理论基准对照
```

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
