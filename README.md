# A Marketplace for Data 数值环境

这个项目实现了 Agarwal, Dahleh, and Sarkar (2019) *A Marketplace for Data: An Algorithmic Solution* 的第一部分机制模拟。目标是把论文中的市场机制写成可运行的 Python 数值环境，并提供一个简单前端来观察：

- 买家在 Myerson 支付规则下诚实报价是否占优。
- 平台价格更新是否朝理论最优固定价格收敛。
- Shapley 收益分配和相似度惩罚是否抑制复制数据。

## 文件结构

```text
marketplace_for_data/
  mechanism.py      # AF / RF / PF / PD 和闭式 Ground Truth
  env.py            # 平台、买家、卖家效用计算环境
  demo.py           # Python 最小示例
  dashboard.html    # 可视化前端，直接用浏览器打开
```

## 论文机制到代码的对应

| 论文机制 | 代码 | 含义 |
| --- | --- | --- |
| `AF(p_n, b_n; X_M)` | `data_allocation_function` | 当 `b < p` 时退化数据；当 `b >= p` 时完整分配数据 |
| `RF(p_n, b_n, Y_n)` | `myerson_revenue_function` | Myerson 支付：`b h(b) - integral_0^b h(z) dz` |
| `PF` | `PriceUpdateState` | 对价格候选池做乘法权重更新 |
| `PD` | `robust_payment_division` | Shapley 近似 + 余弦相似度指数惩罚 |
| Ground Truth | `closed_form_ground_truth` | 简化质量曲线下的理论诚实报价和最优固定价格 |

当前闭式解使用简化质量曲线：

```text
h_p(b) = min(b / p, 1)
```

在这个设定下，Myerson 规则保证买家最优报价为：

```text
b* = mu
```

也就是“诚实报价占优”。前端会直接比较三种策略：低报、诚实、高报。

## 运行 Python 示例

需要本机有 Python 和 NumPy：

```powershell
python -m marketplace_for_data_agent.demo
```

输出会包含每轮价格、买家报价、收益、卖家分成，以及闭式 Ground Truth。

## 启动 Python 后端和可视化前端

现在前端通过 Python 后端获取模拟结果。启动服务：

```powershell
python -m marketplace_for_data_agent.server
```

然后打开：

```text
http://127.0.0.1:8000/
```

前端只负责展示和交互；下面这些结果都由 Python 后端计算：

- 策略效用柱状图：低报、诚实、高报的买家平均效用。
- 价格收敛折线：MWU 动态价格与理论最优固定价格 `p*`。
- 复制惩罚柱状图：原始数据、无关数据、复制数据的收益份额。

后端 API：

```text
GET /api/simulate?buyers=120&shade=0.65&overbid=1.4&lambda=0.7
```

如果机制实现正确，默认参数下应看到：

```text
诚实报价的平均买家效用最高或并列最高；
MWU 价格逐步靠近 Ground Truth 的 p*；
复制数据不会获得双倍分成，且会被相似度惩罚压低。
```

## 结果怎么读

### 买家诚实占优

买家真实估值是 `mu`。前端比较：

- 低报：`b = shade * mu`
- 诚实：`b = mu`
- 高报：`b = overbid * mu`

效用计算为：

```text
U_buyer = mu * h_p(b) - RF(p, b)
```

由于 `RF` 是 Myerson 支付，且 `h_p(b)` 单调，理论上 `b = mu` 是最优响应。图中“诚实”柱应最高。

### 平台价格收敛

平台维护一个价格候选池，对每个候选价格计算如果当轮使用该价格能获得多少收入，再用乘法权重更新。足够多轮后，动态价格应接近闭式 Ground Truth 中的最优固定价格。

### 复制稳健性

普通 Shapley 值会让复制数据获得额外分成。论文的 robust PD 使用：

```text
share_i = shapley_i * exp(-lambda * sum_j SM(X_i, X_j))
```

所以与已有数据高度相似的复制特征会被压低分成。


价格现在由 Python 后端的 MWU 乘法权重算法生成，代码在 marketplace_for_data/simulation.py。

  核心逻辑是：

  candidates = np.arange(lower, upper + 0.5 * step, step)
  weights = np.ones_like(candidates)

  也就是先生成一组候选价格，例如默认：

  0.10, 0.15, 0.20, ..., 1.60

  每一轮来一个买家，买家的真实估值是 mu。平台当前价格取“权重最高”的候选价格：

  idx = int(np.argmax(weights))
  p_now = candidates[idx]

  然后后端会假设：如果这一轮分别使用每个候选价格，会从该买家那里收到多少 Myerson 收入：

  gain_i = RF(candidate_price_i, mu) / upper_price

  再用乘法权重更新：

  weights[i] *= 1 + delta * gain_i

  收入越高的候选价格，权重涨得越快。下一轮平台就更可能选择它。

  所以价格路径不是随机生成的，而是由历史买家估值和每个候选价格的收入表现逐步推出来的。

  当前前端里为了让曲线稳定、容易解释，使用的是确定性版本：

  当前价格 = 权重最大的候选价格

  论文 Algorithm 1 原版是按权重概率随机抽一个价格：

  p_n = c_i with probability w_i / W

  如果要更贴近论文，可以把 argmax(weights) 改成按 weights / sum(weights) 随机采样。