# SFD 与统一全局兼容修正

## 1. 最终模型

项目只保留一条增强路径：

```text
IGNN (SN + IN + NR)
  -> SFD
  -> classifier
  -> global compatibility correction
```

所有数据集和 split 使用相同结构及相同的兼容修正训练规则。模型选择只读取 validation accuracy，test 标签不参与训练、超参数选择或 checkpoint 选择。

## 2. 频谱特征解耦（SFD）

对相邻 hop 表示构造差分：

```text
delta_k = h_k - h_(k-1)
h'_k = h_k + tanh(g_k) * delta_k
```

`g_k` 是逐 hop、逐通道的可学习门控。门控零初始化，因此初始输出严格等于原 IGNN。`spectral_decoupling` 继续使用原 concat + MLP；`spectral_decoupling_raw` 保留原始 raw-hop 拼接路径。

## 3. 全局类别兼容矩阵

兼容矩阵估计采用 CMGNN 的核心思想。令主干 logits 为 `Z`：

```text
P = softmax(Z)
```

训练节点处用真实 one-hot 标签替换 `P`，得到 `P_seed`。未标注节点只使用模型软预测。估计过程结合图结构、预测置信度和节点度权重，且不读取 validation 或 test 标签。

由训练阶段估计的矩阵初始化全局类别关系 `C`。训练时允许一个全局共享的矩阵残差对 `C` 做小幅修正，每行始终通过 softmax 归一化。

## 4. 全局修正

兼容证据和最终 logits 为：

```text
E = normalize(A P_seed C^T)
Q = center(log(E))
s = 4 * sigmoid(theta)
Z_out = Z + s Q
```

`center` 在类别维去均值，`s` 是全图共享的有界标量。最终阶段联合微调原 classifier、矩阵残差和 `s`，损失由修正 logits 的交叉熵、原 logits 的辅助交叉熵及矩阵残差正则组成。统一配置为：初始 `s=0.1`、classifier 学习率系数 `0.1`、修正模块学习率 `0.01`、矩阵正则 `0.001`。

## 5. 正式实验

实验覆盖 8 个数据集，每个数据集使用 10 个固定 split。逐 split 保存 seed、split hash、最佳 validation epoch、test accuracy、训练时间及兼容矩阵诊断信息。

| Dataset | 官方 IGNN | SFD | SFD + global | GCC 增量 (pp) | 完整增益 (pp) |
|---|---:|---:|---:|---:|---:|
| Chameleon | 48.202±3.790 | 49.045±4.884 | 48.933±4.770 | -0.112 | +0.730 |
| Actor | 38.289±1.254 | 38.447±1.084 | 38.507±0.958 | +0.059 | +0.217 |
| PubMed | 90.213±0.572 | 90.198±0.531 | 90.304±0.558 | +0.106 | +0.091 |
| Roman-Empire | 90.940±0.711 | 90.982±0.376 | 91.090±0.409 | +0.108 | +0.150 |
| Squirrel | 38.742±2.140 | 38.921±2.034 | 40.764±1.732 | +1.843 | +2.022 |
| Photo | 95.490±0.419 | 95.575±0.345 | 95.686±0.376 | +0.111 | +0.196 |
| Amazon-Ratings | 53.376±1.012 | 53.146±1.081 | 53.164±1.036 | +0.018 | -0.212 |
| WikiCS | 86.245±0.576 | 86.087±0.457 | 86.173±0.469 | +0.085 | -0.073 |

| 配对消融 | 宏平均增益 (pp) | 去极值平均增益 (pp) |
|---|---:|---:|
| SFD 相对官方 IGNN | +0.1129 | +0.0485 |
| 全局兼容修正相对 SFD | +0.2774 | +0.0815 |
| 完整模型相对官方 IGNN | +0.3903 | +0.2187 |

这些结果支持完整模型在 6/8 个数据集上取得平均正增益，并表明全局兼容修正在 SFD 之上提供了小幅边际贡献。结果不支持“所有数据集都提升”或“去极值平均提升至少 0.5 pp”。

逐 split 记录：

- [SFD 与官方 IGNN](../experiments/sfd_backbone_10splits.json)
- [最终全局兼容修正](../experiments/final_global_correction_10splits.json)
- [汇总 CSV](../results/final_summary.csv)