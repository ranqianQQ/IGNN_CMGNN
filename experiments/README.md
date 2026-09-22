# 正式实验文件

- `sfd_backbone_10splits.json`：官方 IGNN 与 SFD 在 8 个数据集、10 个固定 split 上的配对结果。
- `final_global_correction_10splits.json`：从相同 SFD checkpoint 出发训练固定全局兼容修正的逐 split 结果。
- `sfd_10split_<dataset>_<split>_tuned_sfd.pt`：最终兼容修正复现所需的 SFD warm-start checkpoint。

其余调参与失败路线的中间结果已经清理。最终汇总由 `python -m scripts.summarize_final_model` 生成。
