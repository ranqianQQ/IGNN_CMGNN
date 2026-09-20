# 固定全局兼容矩阵修正与尺度消融

## 1. 最终方法边界

最终模型使用固定结构：

```text
IGNN (SN + IN + NR) -> SFD -> classifier -> global compatibility correction
```

所有数据集和所有 split 都使用同一种全局修正。不根据 validation accuracy 在多个预测头之间选择结果。节点局部修正和全局+局部双尺度只用于回答“局部容量是否必要”这一消融问题。

## 2. 兼容矩阵估计

训练标签记为 `Y_train`，邻接矩阵记为 `A`。兼容矩阵 `C` 描述一类节点与各类别邻居相连的相对频率。实现沿用 CMGNN 的核心估计思想：只用训练标签和图结构构造类别关系，进行平滑与行归一化，避免读取 validation 或 test 标签。

主干输出 logits `Z`，令：

```text
P = softmax(Z)
```

训练节点处用真实 one-hot 标签替换 `P`，得到 `P_seed`。兼容证据为：

```text
E = normalize(A P_seed C^T)
Q = center(log(E))
```

其中 `normalize` 对每个节点的类别证据做归一化，`center` 在类别维去均值。中心化使修正表达类别间相对偏好，避免只改变全部 logits 的共同偏置。

## 3. 固定全局修正

最终采用：

```text
s = s_max * sigmoid(theta)
Z_out = Z + s Q
```

`s` 是全图共享的有界标量。它控制兼容证据对原 logits 的修正幅度，而不形成新的分类支路。实验统一使用 validation 筛选出的 `classwise_h8_bound05` 配置，之后在 8 个数据集的全部 10 个 split 上保持不变。

## 4. 局部与双尺度消融

为检验节点级修正是否有必要，消融实验还实现了局部偏差：

```text
B = center(log softmax(Z))
r_i = [B_i || Q_i || |Q_i-B_i| || B_i⊙Q_i]
u_i = tanh(MLP(r_i))
delta_i = lambda/2 * (u_i - mean_nodes(u))
```

对应预测头为：

```text
local: Z_l = Z + delta_i⊙Q_i
dual:  Z_d = Z + (s+delta_i)⊙Q_i
```

局部 MLP 末层零初始化，节点维均值约束让 `delta` 表示相对全局平均的偏差。该设计在结构上自洽，但最终是否保留由固定模型的完整 10-split 消融决定。

## 5. 严格消融结果

各方法使用同一批 SFD checkpoint、固定 split、seed 和 validation checkpoint 选择规则。

| 方法 | 宏平均配对增益 (pp) | 去极值平均增益 (pp) |
|---|---:|---:|
| classifier-only | +0.2208 | +0.0060 |
| **固定全局修正** | **+0.2774** | **+0.0815** |
| 仅节点局部修正 | +0.1910 | +0.0169 |
| 固定全局+局部 | +0.2823 | +0.0806 |

固定双尺度的普通宏平均比固定全局高 `0.0050 pp`，但去极值主指标低 `0.0009 pp`；仅局部修正也明显低于固定全局。局部尺度没有提供可重复的额外收益，所以最终模型删除局部预测分支，只保留固定全局修正。

## 6. 最终 10-split 结果

| Dataset | SFD mean±std | SFD + global mean±std | Gain (pp) | W/T/L |
|---|---:|---:|---:|---:|
| Chameleon | 49.045±4.633 | 48.933±4.770 | -0.112±0.443 | 2/4/4 |
| Actor | 38.447±1.028 | 38.507±0.958 | +0.059±0.358 | 6/1/3 |
| PubMed | 90.198±0.504 | 90.304±0.558 | +0.106±0.157 | 6/1/3 |
| Roman-Empire | 90.982±0.357 | 91.090±0.409 | +0.108±0.199 | 7/0/3 |
| Squirrel | 38.921±1.930 | 40.764±1.732 | +1.843±0.928 | 10/0/0 |
| Photo | 95.575±0.327 | 95.686±0.376 | +0.111±0.127 | 7/2/1 |
| Amazon-Ratings | 53.146±1.025 | 53.164±1.036 | +0.018±0.144 | 5/1/4 |
| WikiCS | 86.087±0.434 | 86.173±0.469 | +0.085±0.045 | 9/1/0 |

普通宏平均配对增益为 `+0.2774 pp`。去掉最高的 Squirrel 和最低的 Chameleon 后，剩余六个数据集的平均增益为 `+0.0815 pp`。

## 7. 可支持的结论

当前证据支持：兼容矩阵修正能在不改变 IGNN 主干的条件下，为 7/8 个数据集带来平均正增益，主要效果来自固定全局类别关系。当前证据不支持“所有数据集都提升”或“去极值平均提升至少 0.5 pp”。

逐 split 的 validation、test、checkpoint 和候选配置记录保存在 [evidence_residual_ablation_10splits.json](../experiments/evidence_residual_ablation_10splits.json)。
