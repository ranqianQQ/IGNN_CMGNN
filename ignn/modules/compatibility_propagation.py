"""CMGNN-style fixed global compatibility estimation utilities."""
import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor


def _edge_rows_and_columns(edge_index, device):
    if isinstance(edge_index, SparseTensor):
        row, column, _ = edge_index.coo()
    else:
        row, column = edge_index
    row = row.to(device).long()
    column = column.to(device).long()
    keep = row != column
    return row[keep], column[keep]


def _row_normalized_adjacency(edge_index, num_nodes, device):
    row, column = _edge_rows_and_columns(edge_index, device)
    adjacency = SparseTensor(
        row=row, col=column, sparse_sizes=(num_nodes, num_nodes)).coalesce()
    degree = adjacency.sum(dim=1).float()
    normalized = degree.clamp_min(1.).reciprocal().view(-1, 1) * adjacency
    return normalized, degree


def _row_l1_normalize(value):
    return value / value.abs().sum(dim=1, keepdim=True).clamp_min(1e-12)


@torch.no_grad()
def estimate_cmgnn_compatibility(logits, edge_index, labels, train_mask):
    """Estimate one global class transition matrix without val/test labels."""
    nodes, classes = logits.shape
    adjacency, degree = _row_normalized_adjacency(
        edge_index, nodes, logits.device)
    probability = logits.softmax(dim=-1)
    pseudo_label = probability.clone()
    pseudo_label[train_mask] = F.one_hot(
        labels[train_mask], classes).to(probability.dtype)

    confidence = (
        torch.log(torch.tensor(float(classes), device=logits.device))
        + (pseudo_label * pseudo_label.clamp_min(1e-8).log()).sum(dim=1)
    ).clamp_min(0.)
    degree_weight = torch.ones_like(degree)
    low = degree <= classes
    middle = (degree > classes) & (degree <= 3 * classes)
    degree_weight[low] = degree[low] / (2. * classes)
    degree_weight[middle] = .25 + degree[middle] / (4. * classes)

    weighted_label = confidence.unsqueeze(1) * pseudo_label
    neighbourhood = _row_l1_normalize(adjacency.matmul(weighted_label))
    centre = _row_l1_normalize(
        (pseudo_label * degree_weight.unsqueeze(1)
         * confidence.unsqueeze(1)).t())
    compatibility = _row_l1_normalize(centre.matmul(neighbourhood))
    diagnostics = {
        "mean_confidence": float(confidence.mean()),
        "mean_degree_weight": float(degree_weight.mean()),
        "zero_confidence_nodes": int((confidence <= 1e-12).sum()),
    }
    return compatibility, diagnostics
