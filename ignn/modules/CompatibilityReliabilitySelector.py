"""Validation-only reliability selection for compatibility correction heads."""
import torch
from torch import nn


class CompatibilityReliabilitySelector(nn.Module):
    """Route predictions through the compatibility head that generalizes best.

    The selector has one global rule for every graph and split: choose the head
    with higher validation accuracy.  Dataset identities are never inputs to
    the module.  The selected head is stored in the state dict for exact test
    and deployment reproduction.
    """

    GLOBAL_HEAD = 0
    DUAL_SCALE_HEAD = 1

    def __init__(self):
        super().__init__()
        self.register_buffer(
            "selected_head", torch.tensor(self.DUAL_SCALE_HEAD, dtype=torch.long))

    @torch.no_grad()
    def select(self, global_logits, dual_scale_logits, labels, validation_mask):
        global_accuracy = (
            global_logits[validation_mask].argmax(dim=-1)
            == labels[validation_mask]).float().mean()
        dual_accuracy = (
            dual_scale_logits[validation_mask].argmax(dim=-1)
            == labels[validation_mask]).float().mean()
        # A tie keeps the more expressive dual-scale head.  This is fixed and
        # independent of dataset identity.
        choice = (dual_accuracy >= global_accuracy).to(torch.long)
        self.selected_head.copy_(choice)
        return {
            "global_validation_accuracy": float(global_accuracy),
            "dual_scale_validation_accuracy": float(dual_accuracy),
            "selected_head": (
                "dual_scale" if int(choice) == self.DUAL_SCALE_HEAD
                else "global"),
        }

    def forward(self, global_logits, dual_scale_logits):
        if int(self.selected_head) == self.DUAL_SCALE_HEAD:
            return dual_scale_logits
        return global_logits
