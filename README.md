# IGNN-CMGNN：SFD 与固定全局兼容矩阵修正

本仓库在 [IGNN](https://github.com/galogm/IGNN) 主干上加入两个连续模块，同时保留 IGNN 原有的 SN、IN 和 NR 三阶段结构：

1. **SFD（Signed Frequency Decoupling）**：对相邻 hop 表示的差分进行带符号、逐通道调制，再沿用原来的多 hop 融合路径。
2. **GCC（Global Compatibility Correction）**：估计 CMGNN 风格的类别兼容矩阵，根据邻域类别证据对 SFD 输出 logits 做一次固定形式的全局修正。

最终模型固定为同一个结构和同一组参数规则，不针对数据集或 split 选择 `global / local / dual`，也不增加第二分类器、虚拟节点或图重连。

## 模型结构

```text
Graph (X, A)
    -> IGNN backbone (SN + IN + NR)
    -> SFD
    -> classifier logits Z
    -> fixed global compatibility correction
    -> prediction
```

设 `P = softmax(Z)`。训练节点使用真实 one-hot 标签替换预测概率，得到 `P_seed`；兼容矩阵 `C` 由模型软预测、训练标签和图结构估计，不读取 validation 或 test 标签。修正过程为：

```text
E = normalize(A P_seed C^T)
Q = center(log(E))
s = s_max * sigmoid(theta)
Z_out = Z + s Q
```

`Q` 表示根据类别兼容关系得到的全局传播证据，`s` 是所有节点共享的有界修正强度。该模块只修正原分类 logits，不改变 IGNN 主干结构。

公式、训练边界和消融见 [docs/GLOBAL_COMPATIBILITY_METHOD_AND_ABLATION.md](docs/GLOBAL_COMPATIBILITY_METHOD_AND_ABLATION.md)，方法演变见 [docs/METHOD_EVOLUTION.md](docs/METHOD_EVOLUTION.md)。

## 实验协议

- 8 个数据集，每个数据集 10 个固定 split。
- SFD 与 GCC 使用相同数据划分和同一组 SFD checkpoint。
- 候选配置只在各数据集前 3 个 split 的 validation accuracy 上进行一次跨数据集统一筛选；确定 `classwise_h8_bound05` 后，在全部 10 个 split 上固定使用。
- 每个 split 只按 validation accuracy 选择 checkpoint，test 只用于最终报告。
- 主汇总先计算每个数据集的 10-split 配对增益，再删除最高和最低的数据集增益，最后对其余 6 个数据集求平均。

## 10-split 最终结果

主比较是完整模型 `SFD + GCC` 相对官方 IGNN。表中 `±` 为 10 个 split 的样本标准差，配对增益按每个 split 的 `最终模型 − 官方 IGNN` 计算。

| Dataset | 官方 IGNN | SFD | SFD + GCC | 总配对增益 (pp) | W/T/L |
|---|---:|---:|---:|---:|---:|
| Chameleon | 48.202±3.790 | 49.045±4.884 | 48.933±4.770 | +0.730±3.112 | 6/0/4 |
| Actor | 38.289±1.254 | 38.447±1.084 | 38.507±0.958 | +0.217±0.890 | 6/0/4 |
| PubMed | 90.213±0.572 | 90.198±0.531 | 90.304±0.558 | +0.091±0.361 | 6/0/4 |
| Roman-Empire | 90.940±0.711 | 90.982±0.376 | 91.090±0.409 | +0.150±0.648 | 5/0/5 |
| Squirrel | 38.742±2.140 | 38.921±2.034 | 40.764±1.732 | +2.022±1.919 | 9/0/1 |
| Photo | 95.490±0.419 | 95.575±0.345 | 95.686±0.376 | +0.196±0.195 | 8/2/0 |
| Amazon-Ratings | 53.376±1.012 | 53.146±1.081 | 53.164±1.036 | -0.212±0.397 | 3/0/7 |
| WikiCS | 86.245±0.576 | 86.087±0.457 | 86.173±0.469 | -0.073±0.332 | 4/2/4 |

- 完整模型相对官方 IGNN 的宏平均配对增益：`+0.3903 pp`。
- 去掉最高的 Squirrel 和最低的 Amazon-Ratings 后：`+0.2187 pp`。
- 6/8 个数据集的平均结果为正。

结果表明完整模型有总体正信号，但仍没有达到“去极值后平均提升 0.5 pp”，也不能表述为所有数据集普遍提升。

## GCC 增量消融

下表全部以同一组 SFD checkpoint 为基线，因此衡量的是兼容矩阵阶段的额外贡献，不是完整模型相对官方 IGNN 的总增益。

| 修正方式 | 宏平均增益 (pp) | 去极值增益 (pp) | 最终采用 |
|---|---:|---:|---:|
| classifier-only | +0.2208 | +0.0060 | 否 |
| **固定全局修正** | **+0.2774** | **+0.0815** | **是** |
| 仅节点局部修正 | +0.1910 | +0.0169 | 否 |
| 固定全局+局部 | +0.2823 | +0.0806 | 否 |

局部修正没有改善去极值主指标，固定双尺度也没有优于固定全局。因此最终模型只保留全局兼容矩阵修正；局部和双尺度结构仅作为消融，不参与最终预测。

## 关键文件

| 文件 | 作用 |
|---|---|
| `ignn/modules/SpectralFeatureDecoupling.py` | SFD 的 hop 差分与带符号逐通道门控 |
| `ignn/modules/compatibility_propagation.py` | CMGNN 风格兼容矩阵估计 |
| `ignn/modules/EvidenceResidualDualScaleAdapter.py` | 全局修正实现及局部/双尺度消融 |
| `scripts/run_evidence_residual_ablation.py` | 统一 validation 筛选和 10-split 消融 |
| `scripts/summarize_final_model.py` | 按 split hash 配对并汇总完整模型相对官方 IGNN 的结果 |
| `experiments/evidence_residual_ablation_10splits.json` | 逐 split 原始结果、验证指标与测试指标 |

## 复现

安装方式见 `.ci/install.sh`；项目实验使用 `DL_softgnn` 环境。固定 split 位于 `data/random_splits/fixed_splits/`。

```bash
# 1. 训练 10-split SFD backbone
python -m scripts.train_sfd_backbone --device cuda:0

# 2. 前 3 个固定 split 上，以跨数据集 validation 主指标筛选一个统一配置
python -m scripts.run_evidence_residual_ablation --stage screen

# 3. 固定配置后运行 8×10 splits 消融
python -m scripts.run_evidence_residual_ablation --stage evaluate

# 4. 核对 split hash 并生成完整模型主表
python -m scripts.summarize_final_model
```

## 测试

```bash
python -m pytest tests/test_compatibility_propagation.py \
  tests/test_evidence_residual_dual_scale.py -q
```

已验证兼容矩阵行归一化、全局修正前向与反向以及 CPU/CUDA 路径。

## 来源

- IGNN: [galogm/IGNN](https://github.com/galogm/IGNN)
- CMGNN: [Understanding and Enhancing Message Passing on Heterophilic Graphs via Compatibility Matrix](https://proceedings.neurips.cc/paper_files/paper/2025/hash/4e519b18b04c31adc546738e5e836e6f-Abstract-Conference.html)
- 用于估计器核查的 CMGNN 修正实现：[commit 580dcb4](https://github.com/zzn-promax/CMGNN/commit/580dcb42542ba52119187501beeb7b5879cd05b0)

原始 IGNN 许可证见 [LICENSE](LICENSE)。
