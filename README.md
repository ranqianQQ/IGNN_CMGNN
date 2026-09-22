# IGNN + SFD + Global Compatibility Correction

本项目保留一个最终扩展：在官方 c-IGNN 上加入频谱特征解耦（SFD），再使用一个固定结构的全局类别兼容矩阵修正分类 logits。

```text
IGNN multi-hop representations
  -> SFD signed hop-detail modulation
  -> original concat/raw NR path
  -> classifier
  -> fixed global compatibility correction
```

SFD 对连续 hop 表示构造差分，并用零初始化的通道门控调节差分：

```text
delta_k = h_k - h_(k-1)
h'_k = h_k + tanh(g_k) * delta_k
```

门控为零时严格退化为原 c-IGNN。`spectral_decoupling` 保留原 concat + MLP，`spectral_decoupling_raw` 对原本不使用 NR MLP 的配置保留完整 hop 拼接。

全局兼容修正只使用训练标签和模型软预测估计一个类别转移矩阵 `C`：

```text
E = normalize(A P_seed C^T)
Z_out = Z + s * center(log(E))
```

`s` 是所有节点共享的有界标量。模型不包含节点局部兼容分支，也不按数据集或 split 选择不同结构。

## 环境

项目沿用官方 IGNN 依赖，推荐使用本机 `DL_softgnn` 环境：

```powershell
conda activate DL_softgnn
pip install -r requirements.txt
```

## 复现实验

先训练官方 IGNN 与 SFD 的 8 数据集 × 10 固定 split 配对模型：

```powershell
python -m scripts.train_sfd_backbones
```

再从保存的 SFD checkpoint 训练固定全局兼容修正，只按 validation accuracy 选择 checkpoint：

```powershell
python -m scripts.run_final_global_correction
```

生成最终汇总：

```powershell
python -m scripts.summarize_final_model
```

保留的正式结果位于：

- `experiments/sfd_backbone_10splits.json`
- `experiments/final_global_correction_10splits.json`
- `results/final_summary.csv`

方法、消融和 10-split 结论见 [docs/GLOBAL_COMPATIBILITY_METHOD_AND_ABLATION.md](docs/GLOBAL_COMPATIBILITY_METHOD_AND_ABLATION.md)。

## 原项目

本项目基于 [galogm/IGNN](https://github.com/galogm/IGNN)。官方论文：*Making Classic GNNs Strong Baselines Across Varying Homophily: A Smoothness–Generalization Perspective*（NeurIPS 2025）。
