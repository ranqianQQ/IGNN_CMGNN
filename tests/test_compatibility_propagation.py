import torch
from torch_sparse import SparseTensor
from ignn.modules.CompatibilityGuidedFineTuning import CompatibilityGuidedFineTuning
from ignn.modules.compatibility_propagation import (
    estimate_compatibility, estimate_cmgnn_compatibility,
    centered_compatibility_propagation,
    select_centered_compatibility)


def _case():
    # Bipartite class pattern: every observed edge crosses class 0/1.
    edge_index=torch.tensor([[0,1,1,2,2,3,3,0],[1,0,2,1,3,2,0,3]])
    labels=torch.tensor([0,1,0,1])
    train=torch.tensor([True,True,True,False])
    logits=torch.tensor([[3.,0.],[0.,3.],[3.,0.],[.2,.1]])
    return edge_index,labels,train,logits


def test_compatibility_is_stochastic_and_captures_heterophily():
    edge_index,labels,train,_=_case()
    h=estimate_compatibility(edge_index,labels,train,2,smoothing=0.)
    assert torch.allclose(h.sum(1),torch.ones(2))
    assert h[0,1]>h[0,0] and h[1,0]>h[1,1]


def test_zero_strength_preserves_base_predictions():
    edge_index,labels,train,logits=_case()
    scores,_=centered_compatibility_propagation(logits,edge_index,labels,train,0.,5)
    assert torch.equal(scores.argmax(-1),logits.argmax(-1))


def test_selection_returns_valid_scores_without_test_labels():
    edge_index,labels,train,logits=_case();val=torch.tensor([False,False,False,True])
    scores,stats=select_centered_compatibility(logits,edge_index,labels,train,val,
                                               strengths=(0.,.5),steps=(1,2))
    assert scores.shape==logits.shape
    assert stats['uses_test_labels'] is False
    assert stats['strength'] in (0.,.5)


def test_cmgnn_estimator_is_stochastic_and_does_not_read_unlabelled_labels():
    edge_index,labels,train,logits=_case()
    first,diagnostics=estimate_cmgnn_compatibility(
        logits,edge_index,labels,train)
    changed=labels.clone();changed[~train]=1-changed[~train]
    second,_=estimate_cmgnn_compatibility(
        logits,edge_index,changed,train)
    assert torch.allclose(first,second)
    assert torch.allclose(first.sum(1),torch.ones(2),atol=1e-6)
    assert diagnostics['mean_confidence'] >= 0.


def test_cmgnn_centered_selector_zero_strength_preserves_predictions():
    edge_index,labels,train,logits=_case()
    val=torch.tensor([False,False,False,True])
    scores,stats=select_centered_compatibility(
        logits,edge_index,labels,train,val,
        strengths=(0.,),steps=(1,2),estimator='cmgnn')
    assert torch.equal(scores.argmax(-1),logits.argmax(-1))
    assert stats['compatibility_estimator']=='cmgnn'


def test_compatibility_finetuner_matrix_and_gradients():
    edge_index,labels,train,logits=_case()
    row,col=edge_index
    adj=SparseTensor(row=row,col=col,sparse_sizes=(4,4)).coalesce()
    degree=adj.sum(dim=1).clamp_min(1.)
    adj=degree.reciprocal().view(-1,1)*adj
    module=CompatibilityGuidedFineTuning(
        torch.tensor([[.2,.8],[.7,.3]]),learnable_matrix=True)
    variable_logits=logits.clone().requires_grad_(True)
    corrected=module(variable_logits,adj,labels,train)
    corrected.sum().backward()
    assert torch.allclose(module.compatibility().sum(1),torch.ones(2))
    assert variable_logits.grad is not None
    assert module.matrix_residual.grad is not None
    assert module.raw_strength.grad is not None


def test_compatibility_finetuner_ema_stays_stochastic():
    module=CompatibilityGuidedFineTuning(torch.eye(3),learnable_matrix=False)
    module.update_compatibility(torch.ones(3,3),momentum=.5)
    assert torch.allclose(
        module.base_compatibility.sum(1),torch.ones(3),atol=1e-6)
