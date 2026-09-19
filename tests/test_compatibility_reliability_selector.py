import torch

from ignn.modules.CompatibilityReliabilitySelector import (
    CompatibilityReliabilitySelector,
)


def test_selects_head_using_validation_accuracy_and_routes_logits():
    selector = CompatibilityReliabilitySelector()
    labels = torch.tensor([0, 1, 0, 1])
    mask = torch.tensor([True, True, False, False])
    global_logits = torch.tensor([[3., 0.], [0., 3.], [2., 0.], [2., 0.]])
    dual_logits = torch.tensor([[0., 3.], [0., 3.], [0., 2.], [0., 2.]])
    diagnostics = selector.select(
        global_logits, dual_logits, labels, mask)
    assert diagnostics["selected_head"] == "global"
    assert torch.equal(selector(global_logits, dual_logits), global_logits)


def test_tie_rule_and_state_dict_are_reproducible():
    selector = CompatibilityReliabilitySelector()
    labels = torch.tensor([0, 1])
    mask = torch.tensor([True, True])
    first = torch.tensor([[2., 0.], [0., 2.]])
    second = torch.tensor([[4., 0.], [0., 4.]])
    diagnostics = selector.select(first, second, labels, mask)
    assert diagnostics["selected_head"] == "dual_scale"
    restored = CompatibilityReliabilitySelector()
    restored.load_state_dict(selector.state_dict())
    assert torch.equal(restored(first, second), second)
