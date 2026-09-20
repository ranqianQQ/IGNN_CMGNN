# 实验结果索引

## 最终采用

- `evidence_residual_screen.json`：只使用前 3 个固定 split 的 validation 指标，对跨数据集统一候选配置进行一次筛选。
- `evidence_residual_ablation_10splits.json`：固定 `classwise_h8_bound05` 后的 8 数据集 × 10 splits 完整消融。最终模型读取其中 `global` 模式的结果。

## 历史对照与否决实验

- `compatibility_finetuning_10splits.json`
- `compatibility_relation_adapter_10splits.json`
- `dual_scale_10splits.json`
- `reliability_controlled_dual_scale_10splits.json`
- `reliability_selected_compatibility_10splits.json`

这些文件用于记录方法演变和复核旧结论，不定义最终模型。其中 `reliability_*` 文件包含根据 validation 指标在多个预测头之间选择结果的历史实验；该规则已被否决，最终结果和复现流程都不使用它。
