# IGNN-CMGNN：SFD 与可靠性选择的双尺度兼容微调

本仓库在 [IGNN](https://github.com/galogm/IGNN) 主干上实现两个连续的模型改造：

1. **SFD（Signed Frequency Decoupling）**：对相邻 hop 的表示差分进行带符号、逐通道调制，再保留 IGNN 的完整 hop 拼接路径。
2. **RS-DCFT（Reliability-Selected Dual-Scale Compatibility Fine-Tuning）**：使用 CMGNN 风格兼容矩阵建立全局兼容头和双尺度节点关系头，并仅依据 validation accuracy 选择可靠输出头。

最终方法在 8 个数据集、每个数据集 10 个官方固定划分上，相对同一组 SFD checkpoint 获得 **+0.5298 percentage points** 的宏平均配对 test 增益。兼容模块对所有数据集使用相同结构、训练候选和选择规则；数据集名称不会输入兼容模块或可靠性选择器。

## 最终结构

```text
Graph (X, A)
    │
    ▼
IGNN + SFD backbone ───────────────► logits Z
                                          │
                         CMGNN compatibility estimator
                                          │ C
                       ┌──────────────────┴──────────────────┐
                       ▼                                     ▼
             Global compatibility head             Dual-scale head
             Z + s · log(E)                  Z + s · log(E) + g · R(Z,E)
                       │                                     │
                       └──────── validation reliability ─────┘
                                          │
                                          ▼
                                     prediction
```

其中 `E = normalize(A P_seed C^T)`；`P_seed` 将训练节点替换为真实 one-hot 标签，其余节点使用 SFD 分类器概率。双尺度关系特征为：

```text
R input = [center(log P), center(log E),
           |center(log P) - center(log E)|,
           center(log P) ⊙ center(log E)]
```

详细演变过程、保留/淘汰理由和公式见 [docs/METHOD_EVOLUTION.md](docs/METHOD_EVOLUTION.md)。

## 10-split 结果

| Dataset | SFD | RS-DCFT | Paired gain (pp) | W/T/L |
|---|---:|---:|---:|---:|
| Actor | 38.447±1.084 | 38.592±0.968 | +0.145±0.788 | 6/0/4 |
| Amazon-Ratings | 53.146±1.081 | 53.195±1.077 | +0.049±0.151 | 6/0/4 |
| Chameleon | 49.045±4.884 | 48.820±3.941 | -0.225±1.526 | 2/5/3 |
| Photo | 95.575±0.345 | 95.686±0.371 | +0.111±0.134 | 7/2/1 |
| PubMed | 90.198±0.531 | 90.216±0.534 | +0.018±0.104 | 4/2/4 |
| Roman-Empire | 90.982±0.376 | 91.178±0.350 | +0.196±0.234 | 8/0/2 |
| Squirrel | 38.921±2.034 | 42.742±2.344 | +3.820±2.350 | 10/0/0 |
| WikiCS | 86.087±0.457 | 86.211±0.479 | +0.124±0.055 | 9/1/0 |

该结果支持“8 个基准上的平均提升”和“7/8 个数据集平均为正”。Chameleon 的平均结果仍下降，因此本仓库不宣称每个数据集都提升。

- 汇总 CSV：[results/final_summary.csv](results/final_summary.csv)
- 最终逐 split JSON：[experiments/reliability_selected_compatibility_10splits.json](experiments/reliability_selected_compatibility_10splits.json)
- 全局兼容头 JSON：[experiments/compatibility_finetuning_10splits.json](experiments/compatibility_finetuning_10splits.json)
- 节点关系头 JSON：[experiments/compatibility_relation_adapter_10splits.json](experiments/compatibility_relation_adapter_10splits.json)
- 双尺度头 JSON：[experiments/dual_scale_10splits.json](experiments/dual_scale_10splits.json)

## 关键代码

| 文件 | 作用 |
|---|---|
| `ignn/modules/SpectralFeatureDecoupling.py` | SFD hop 差分与带符号逐通道门控 |
| `ignn/modules/compatibility_propagation.py` | CMGNN 兼容矩阵估计及诊断 |
| `ignn/modules/CompatibilityGuidedFineTuning.py` | 低方差全局兼容头 |
| `ignn/modules/CompatibilityRelationAdapter.py` | 演变中的节点关系残差头 |
| `ignn/modules/DualScaleCompatibilityAdapter.py` | 最终双尺度兼容头 |
| `ignn/modules/CompatibilityReliabilitySelector.py` | validation-only 可靠性选择器 |

## 环境与数据

项目继承官方 IGNN 的依赖。实验使用 `DL_softgnn` 环境；基础安装方式见 `.ci/install.sh`。

```bash
pip install -r requirements.txt
```

固定划分位于 `data/random_splits/fixed_splits/`，采用 48%/32%/20% 的 train/validation/test 比例。WikiCS 复现实验还需要 `data/wikics_dgl.pt`。

## 复现实验

所有选择只使用 validation accuracy。建议从仓库根目录用模块方式运行脚本。

```bash
# 1. 训练 SFD backbone
python -m scripts.train_sfd_backbone --device cuda:0

# 2. 全局兼容头：先用 splits 0-2 做统一候选筛选，再跑 10 splits
python -m scripts.screen_compatibility_finetuning --split-end 3
python -m scripts.evaluate_compatibility_finetuning

# 3. 节点关系残差阶段（保留用于展示演变）
python -m scripts.screen_compatibility_relation_adapter --split-end 3
python -m scripts.evaluate_compatibility_relation_adapter

# 4. 双尺度兼容头
python -m scripts.screen_dual_scale_compatibility --split-end 3
python -m scripts.evaluate_dual_scale_compatibility \
  --candidate dual_h8_joint --output experiments/dual_scale_10splits.json

# 5. 同一 validation-only 规则选择可靠头并生成最终报告
python -m scripts.summarize_reliability_selected_compatibility
```

## 测试

```bash
python -m pytest tests/test_compatibility_propagation.py \
  tests/test_compatibility_reliability_selector.py -q
```

已验证 CPU/CUDA 前向与反向、兼容矩阵行归一化、可靠性选择与状态恢复。不同 PyTorch/CUDA/GPU 组合可能产生小幅数值差异，最终比较应始终使用相同环境和固定 split 做 paired evaluation。

## 来源与致谢

- IGNN: [galogm/IGNN](https://github.com/galogm/IGNN)
- CMGNN paper: [Understanding and Enhancing Message Passing on Heterophilic Graphs via Compatibility Matrix](https://proceedings.neurips.cc/paper_files/paper/2025/hash/4e519b18b04c31adc546738e5e836e6f-Abstract-Conference.html)
- CMGNN corrected implementation used for estimator audit: [commit 580dcb4](https://github.com/zzn-promax/CMGNN/commit/580dcb42542ba52119187501beeb7b5879cd05b0)

原始 IGNN 许可证见 [LICENSE](LICENSE)。
