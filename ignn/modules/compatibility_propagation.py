"""Validation-selected compatibility propagation for heterophilous graphs.

Two compatibility estimators are available: direct counts over training-only
edges and CMGNN-style confidence-weighted pseudo labels. Validation labels are
used solely to select propagation strength and iteration count; test labels are
never read by the propagation or model-selection code.
"""
import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor


def _edge_rows_and_columns(edge_index, device):
    if isinstance(edge_index, SparseTensor):
        row, col, _ = edge_index.coo()
    else:
        row, col = edge_index
    row = row.to(device).long()
    col = col.to(device).long()
    keep = row != col
    return row[keep], col[keep]


def _row_normalized_adjacency(edge_index, num_nodes, device):
    row, col = _edge_rows_and_columns(edge_index, device)
    adjacency = SparseTensor(
        row=row, col=col, sparse_sizes=(num_nodes, num_nodes)).coalesce()
    degree = adjacency.sum(dim=1).float()
    normalized = degree.clamp_min(1.0).reciprocal().view(-1, 1) * adjacency
    return normalized, degree


def _row_l1_normalize(matrix):
    return matrix / matrix.abs().sum(dim=1, keepdim=True).clamp_min(1e-12)


@torch.no_grad()
def estimate_compatibility(edge_index, labels, train_mask, num_classes, smoothing=1.0):
    if isinstance(edge_index, SparseTensor):
        row, col, _ = edge_index.coo()
    else:
        row, col = edge_index
    row = row.to(labels.device).long()
    col = col.to(labels.device).long()
    observed = train_mask[row] & train_mask[col] & (row != col)
    pairs = labels[row[observed]] * num_classes + labels[col[observed]]
    counts = torch.bincount(pairs, minlength=num_classes * num_classes).float()
    counts = counts.reshape(num_classes, num_classes) + float(smoothing)
    return counts / counts.sum(dim=1, keepdim=True)


@torch.no_grad()
def estimate_cmgnn_compatibility(logits, edge_index, labels, train_mask,
                                 num_classes=None):
    """Estimate CM with the corrected confidence/degree scheme from CMGNN.

    This ports ``update_connect_preference`` from the current official CMGNN
    implementation.  Model probabilities provide soft pseudo labels for
    unlabelled nodes, while training nodes are replaced by exact one-hot labels.
    Validation and test labels are never read.
    """
    num_nodes, inferred_classes = logits.shape
    classes = inferred_classes if num_classes is None else int(num_classes)
    if classes != inferred_classes:
        raise ValueError(
            f'num_classes={classes} does not match logits width={inferred_classes}')

    adjacency, degree = _row_normalized_adjacency(
        edge_index, num_nodes, logits.device)
    probabilities = logits.softmax(dim=-1)
    pseudo_labels = probabilities.clone()
    pseudo_labels[train_mask] = F.one_hot(
        labels[train_mask], num_classes=classes).to(probabilities.dtype)

    # CMGNN Eq. (10): g_i = log(K) - entropy(C_i).
    confidence = (
        torch.log(torch.tensor(float(classes), device=logits.device))
        + (pseudo_labels * pseudo_labels.clamp_min(1e-8).log()).sum(dim=1)
    ).clamp_min(0.0)

    # Corrected official implementation (commit 580dcb4): mutually exclusive
    # degree ranges.  The earlier code accidentally overwrote the low range.
    degree_weight = torch.ones_like(degree)
    low = degree <= classes
    middle = (degree > classes) & (degree <= 3 * classes)
    degree_weight[low] = degree[low] / (2.0 * classes)
    degree_weight[middle] = 0.25 + degree[middle] / (4.0 * classes)

    confidence_labels = confidence.unsqueeze(1) * pseudo_labels
    semantic_neighborhood = _row_l1_normalize(
        adjacency.matmul(confidence_labels))
    central_distribution = _row_l1_normalize(
        (pseudo_labels * degree_weight.unsqueeze(1)
         * confidence.unsqueeze(1)).t())
    compatibility = _row_l1_normalize(
        central_distribution.matmul(semantic_neighborhood))
    diagnostics = {
        'mean_confidence': float(confidence.mean().item()),
        'mean_degree_weight': float(degree_weight.mean().item()),
        'zero_confidence_nodes': int((confidence <= 1e-12).sum().item()),
    }
    return compatibility, diagnostics


@torch.no_grad()
def propagate_with_compatibility(logits, edge_index, labels, train_mask,
                                 alpha=0.0, steps=1, smoothing=1.0):
    n, classes = logits.shape
    if isinstance(edge_index, SparseTensor):
        row, col, _ = edge_index.coo()
    else:
        row, col = edge_index
    row = row.to(logits.device).long()
    col = col.to(logits.device).long()
    keep = row != col
    row, col = row[keep], col[keep]
    adj = SparseTensor(row=row, col=col, sparse_sizes=(n, n)).coalesce()
    degree = adj.sum(dim=1).float().clamp_min(1.0)
    adj = degree.reciprocal().view(-1, 1) * adj
    compatibility = estimate_compatibility(
        edge_index, labels, train_mask, classes, smoothing=smoothing)
    base = logits.softmax(dim=-1)
    scores = base.clone()
    targets = F.one_hot(labels[train_mask], classes).float()
    for _ in range(int(steps)):
        neighbor_scores = adj.matmul(scores)
        messages = neighbor_scores.matmul(compatibility.t())
        messages = messages / messages.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        scores = (1.0 - float(alpha)) * base + float(alpha) * messages
        scores[train_mask] = targets
    return scores, compatibility


@torch.no_grad()
def select_compatibility_propagation(logits, edge_index, labels, train_mask,
                                     val_mask, alphas=(0.0, .05, .1, .2, .3, .4),
                                     steps=(1, 2, 5, 10, 20), smoothing=1.0):
    best = None
    for alpha in alphas:
        trial_steps = (1,) if alpha == 0 else steps
        for count in trial_steps:
            scores, compatibility = propagate_with_compatibility(
                logits, edge_index, labels, train_mask, alpha, count, smoothing)
            val = (scores[val_mask].argmax(dim=-1) == labels[val_mask]).float().mean().item()
            key = (val, -float(alpha), -int(count))
            if best is None or key > best[0]:
                best = (key, scores, compatibility, float(alpha), int(count))
    _, scores, compatibility, alpha, count = best
    stats = {
        'method': 'validation_selected_training_edge_compatibility_propagation',
        'alpha': alpha,
        'steps': count,
        'validation_accuracy': best[0][0],
        'compatibility': compatibility.cpu().tolist(),
        'uses_train_labels_for_compatibility': True,
        'uses_validation_labels_for_model_selection': True,
        'uses_test_labels': False,
    }
    return scores, stats


@torch.no_grad()
def centered_compatibility_propagation(logits, edge_index, labels, train_mask,
                                       strength=0.0, steps=1, smoothing=1.0,
                                       estimator='training_edges',
                                       return_diagnostics=False):
    """Linearized belief propagation with centered class compatibilities."""
    n, classes = logits.shape
    adj, _ = _row_normalized_adjacency(edge_index, n, logits.device)
    diagnostics = {}
    if estimator == 'training_edges':
        probability_h = estimate_compatibility(
            edge_index, labels, train_mask, classes, smoothing)
    elif estimator == 'cmgnn':
        probability_h, diagnostics = estimate_cmgnn_compatibility(
            logits, edge_index, labels, train_mask, classes)
    else:
        raise ValueError(f'Unknown compatibility estimator: {estimator}')
    probability_h = .5 * (probability_h + probability_h.t())
    centered_h = probability_h - probability_h.mean(dim=1,keepdim=True)
    base_probability = logits.softmax(dim=-1)
    base = base_probability - base_probability.mean(dim=-1,keepdim=True)
    state = base.clone()
    for _ in range(int(steps)):
        state = base + float(strength) * adj.matmul(state).matmul(centered_h.t())
    if return_diagnostics:
        return state, probability_h, diagnostics
    return state, probability_h


@torch.no_grad()
def select_centered_compatibility(logits,edge_index,labels,train_mask,val_mask,
                                  strengths=(0.0,.1,.25,.5,1.0,2.0,4.0),
                                  steps=(1,2,5,10),smoothing=1.0,
                                  estimator='training_edges'):
    best=None
    for strength in strengths:
        trial_steps=(1,) if strength==0 else steps
        for count in trial_steps:
            scores,compatibility,diagnostics=centered_compatibility_propagation(
                logits,edge_index,labels,train_mask,strength,count,smoothing,
                estimator=estimator,return_diagnostics=True)
            val=(scores[val_mask].argmax(-1)==labels[val_mask]).float().mean().item()
            key=(val,-float(strength),-int(count))
            if best is None or key>best[0]:best=(key,scores,compatibility,float(strength),int(count),diagnostics)
    _,scores,compatibility,strength,count,diagnostics=best
    return scores,{
        'method':'validation_selected_centered_compatibility_propagation',
        'compatibility_estimator':estimator,
        'strength':strength,
        'steps':count,
        'validation_accuracy':best[0][0],
        'compatibility':compatibility.cpu().tolist(),
        'compatibility_diagnostics':diagnostics,
        'uses_train_labels_for_compatibility':True,
        'uses_validation_labels_for_model_selection':True,
        'uses_test_labels':False,
    }
