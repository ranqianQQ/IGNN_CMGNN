# IGNN-CMGNN：SFD 与可靠性控制的双尺度兼容修正

本仓库在 [IGNN](https://github.com/galogm/IGNN) 主干上实现两个连续模块：

1. **SFD (Signed Frequency Decoupling)**：对相邻 hop 表示差分进行带符号、逐通道调制，保留 IGNN 完整 concat 路径。
2. **RC-DSCC (Reliability-Controlled Dual-Scale Compatibility Correction)**：基于 CMGNN 风格兼容矩阵，分别建模全局类别修正和节点局部修正，仅根据 validation accuracy 决定启用全局、局部或两者。

在 8 个数据集、每个 10 个固定 split 上，最终方法相对同一组 SFD checkpoint 的宏平均配对 test 增益为 `+0.3443 pp`。为避免 Squirrel 的大增益主导结论，主指标改为：先计算每个数据集的 10-split 平均配对增益，删除最高和最低数据集，再对剩余 6 个求平均。该**去极值平均增益为 `+0.1050 pp`**。

## 最终结构

```text
Graph (X, A) -> IGNN + SFD -> logits Z
                                  |
                          CMGNN compatibility C
                                  |
                     compatibility evidence Q
                                  |
                 +----------------+----------------+
                 |                                 |
       global correction s*Q          node-local deviation δ_i*Q
                 |                                 |
                 +-------- validation reliability-+
                                  |
                         global / local / dual
                                  |
                              prediction
```

设 `P=softmax(Z)`，训练节点使用真实 one-hot 标签替换预测概率：

```text
E = normalize(A P_seed C^T)
Q = center(log E)
B = center(log P)

global: Z_g = Z + s Q

r_i = [B_i || Q_i || |Q_i-B_i| || B_i⊙Q_i]
u_i = tanh(MLP(r_i))
δ_i = λ/2 * (u_i - mean_nodes(u))
local:  Z_l = Z + δ_i⊙Q_i
dual:   Z_d = Z + (s+δ_i)⊙Q_i
```

`δ_i` 是 node-wise、class-wise、有符号且有界的局部强度，并在节点维度上均值为零。因此全局分支负责整体平移，局部分支只学习节点相对于全局平均的偏差，不会退化成任意第二分类器。局部 MLP 末层零初始化，训练开始时 `dual` 严格等于 `global`。

公式、设计动机、完整消融和结论边界见 [docs/DUAL_SCALE_METHOD_AND_ABLATION.md](docs/DUAL_SCALE_METHOD_AND_ABLATION.md)。历史演变见 [docs/METHOD_EVOLUTION.md](docs/METHOD_EVOLUTION.md)。

## 严格消融

所有对照使用同一 SFD checkpoint、split、seed、训练轮数、early stopping 和 validation checkpoint 规则。`classifier-only` 用于排除单纯再训练分类器的增益。

| 模型 | 宏平均增益 (pp) | 去极值增益 (pp) |
|---|---:|---:|
| SFD | 0.0000 | 0.0000 |
| classifier-only | +0.2208 | +0.0060 |
| 仅全局修正 | +0.2774 | +0.0815 |
| 仅节点局部修正 | +0.1910 | +0.0169 |
| 固定全局+局部 | +0.2823 | +0.0806 |
| **RC-DSCC** | **+0.3443** | **+0.1050** |

消融支持的结论是：全局修正提供主要稳定收益；节点局部修正提供额外容量，但必须通过 validation 可靠性控制避免在不适合的 split 上造成负作用。

## 10-split 结果

| Dataset | SFD | RC-DSCC | Paired gain (pp) | W/T/L |
|---|---:|---:|---:|---:|
| Actor | 38.447±1.084 | 38.487±1.082 | +0.039±0.447 | 6/1/3 |
| Amazon-Ratings | 53.146±1.081 | 53.158±1.032 | +0.012±0.144 | 6/0/4 |
| Chameleon | 49.045±4.884 | 49.326±4.477 | +0.281±0.662 | 5/3/2 |
| Photo | 95.575±0.345 | 95.614±0.400 | +0.039±0.155 | 5/2/3 |
| PubMed | 90.198±0.531 | 90.218±0.494 | +0.020±0.060 | 5/1/4 |
| Roman-Empire | 90.982±0.376 | 91.125±0.403 | +0.143±0.213 | 7/0/3 |
| Squirrel | 38.921±2.034 | 41.034±1.645 | +2.112±1.434 | 10/0/0 |
| WikiCS | 86.087±0.457 | 86.194±0.470 | +0.107±0.070 | 9/1/0 |

八个数据集的平均增益都为正，但 Actor、Amazon-Ratings、Photo 和 PubMed 只是弱增益，本仓库不宣称每个 split 都提升。

- 最终汇总：[results/final_summary.csv](results/final_summary.csv)
- validation 候选筛选：[experiments/evidence_residual_screen.json](experiments/evidence_residual_screen.json)
- 400 条严格消融记录：[experiments/evidence_residual_ablation_10splits.json](experiments/evidence_residual_ablation_10splits.json)
- 80 条最终选择记录：[experiments/reliability_controlled_dual_scale_10splits.json](experiments/reliability_controlled_dual_scale_10splits.json)

## 关键代码

| 文件 | 作用 |
|---|---|
| `ignn/modules/SpectralFeatureDecoupling.py` | SFD hop 差分与带符号逐通道门控 |
| `ignn/modules/compatibility_propagation.py` | CMGNN 兼容矩阵估计 |
| `ignn/modules/EvidenceResidualDualScaleAdapter.py` | 可解释的全局/局部/双尺度修正 |
| `scripts/run_evidence_residual_ablation.py` | 去极值 validation 筛选与严格消融 |
| `scripts/summarize_scale_reliability.py` | validation-only 尺度选择与最终汇总 |

## 复现

安装方式见 `.ci/install.sh`；项目实验使用 `DL_softgnn` 环境。八个数据集的固定 split 已收录在 `data/random_splits/fixed_splits/`。

```bash
# 1. 训练 10-split SFD backbone
python -m scripts.train_sfd_backbone --device cuda:0

# 2. 前 3 splits validation 上选择一个跨数据集全局配置
python -m scripts.run_evidence_residual_ablation --stage screen

# 3. 运行 8×10 splits 严格消融
python -m scripts.run_evidence_residual_ablation --stage evaluate

# 4. validation-only 选择尺度状态并生成最终结果
python -m scripts.summarize_scale_reliability
```

## 测试

```bash
python -m pytest tests/test_compatibility_propagation.py \
  tests/test_compatibility_reliability_selector.py \
  tests/test_evidence_residual_dual_scale.py -q
```

已验证 CPU/CUDA 前向与反向、局部系数零均值与幅度上界、兼容矩阵行归一化、validation-only 选择与 split 一致性。

## 来源

- IGNN: [galogm/IGNN](https://github.com/galogm/IGNN)
- CMGNN: [Understanding and Enhancing Message Passing on Heterophilic Graphs via Compatibility Matrix](https://proceedings.neurips.cc/paper_files/paper/2025/hash/4e519b18b04c31adc546738e5e836e6f-Abstract-Conference.html)
- 用于估计器核查的 CMGNN 修正实现：[commit 580dcb4](https://github.com/zzn-promax/CMGNN/commit/580dcb42542ba52119187501beeb7b5879cd05b0)

原始 IGNN 许可证见 [LICENSE](LICENSE)。
