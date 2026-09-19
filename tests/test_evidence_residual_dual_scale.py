import torch

from ignn.modules.EvidenceResidualDualScaleAdapter import (
    EvidenceResidualDualScaleAdapter,
)


def inputs(device="cpu"):
    torch.manual_seed(7)
    nodes, classes = 11, 4
    logits = torch.randn(nodes, classes, device=device, requires_grad=True)
    adjacency = torch.eye(nodes, device=device)
    labels = torch.arange(nodes, device=device) % classes
    train_mask = torch.arange(nodes, device=device) < 4
    return logits, adjacency, labels, train_mask


def test_zero_initialization_has_exact_nested_degenerations():
    logits, adjacency, labels, train_mask = inputs()
    adapter = EvidenceResidualDualScaleAdapter(torch.eye(4), hidden=8)
    parts = adapter.components(logits, adjacency, labels, train_mask)
    assert torch.equal(parts["local"], logits)
    assert torch.equal(parts["dual"], parts["global"])
    assert torch.count_nonzero(parts["local_coefficient"]) == 0


def test_node_local_coefficient_is_centered_and_bounded():
    logits, adjacency, labels, train_mask = inputs()
    adapter = EvidenceResidualDualScaleAdapter(
        torch.eye(4), hidden=8, max_local_strength=.5,
        classwise_gate=True)
    torch.nn.init.normal_(adapter.local_scorer[-1].weight)
    parts = adapter.components(logits, adjacency, labels, train_mask)
    coefficient = parts["local_coefficient"]
    assert coefficient.shape == logits.shape
    assert torch.allclose(coefficient.mean(dim=0), torch.zeros(4), atol=1e-7)
    assert coefficient.abs().max() <= .5 + 1e-7


def test_all_ablation_modes_backpropagate_and_matrix_is_stochastic():
    for mode in EvidenceResidualDualScaleAdapter.MODES:
        logits, adjacency, labels, train_mask = inputs()
        adapter = EvidenceResidualDualScaleAdapter(
            torch.eye(4), hidden=8, classwise_gate=True)
        scores = adapter(logits, adjacency, labels, train_mask, mode=mode)
        assert scores.shape == logits.shape
        scores.square().mean().backward()
        assert logits.grad is not None
        assert torch.allclose(adapter.compatibility().sum(dim=1),
                              torch.ones(4), atol=1e-7)
