"""Node-wise class residuals driven by a global compatibility matrix."""
import torch
from torch import nn
import torch.nn.functional as F


class CompatibilityRelationAdapter(nn.Module):
    def __init__(self, compatibility, hidden=16, dropout=.1,
                 residual_bound=None):
        super().__init__();compatibility=self._normalize(compatibility.detach().float())
        self.register_buffer('base_compatibility',compatibility.clone());self.matrix_residual=nn.Parameter(torch.zeros_like(compatibility));classes=compatibility.shape[0]
        self.residual_bound = (None if residual_bound is None
                               else float(residual_bound))
        self.relation=nn.Sequential(nn.Linear(4*classes,int(hidden)),nn.ReLU(),nn.Dropout(float(dropout)),nn.Linear(int(hidden),classes))
        nn.init.zeros_(self.relation[-1].weight);nn.init.zeros_(self.relation[-1].bias)
    @staticmethod
    def _normalize(x):x=x.clamp_min(0);return x/x.sum(1,keepdim=True).clamp_min(1e-12)
    def compatibility(self):return (self.base_compatibility.clamp_min(1e-8).log()+self.matrix_residual).softmax(1)
    def regularization(self):return self.matrix_residual.square().mean()
    def forward(self,logits,adjacency,labels,train_mask):
        classes=logits.shape[1];prob=logits.softmax(-1);seeded=prob.clone();seeded[train_mask]=F.one_hot(labels[train_mask],classes).to(prob.dtype)
        evidence=adjacency.matmul(seeded).matmul(self.compatibility().t());evidence=self._normalize(evidence)
        base_log=prob.clamp_min(1e-8).log();base_log=base_log-base_log.mean(1,keepdim=True)
        evidence_log=evidence.clamp_min(1e-8).log();evidence_log=evidence_log-evidence_log.mean(1,keepdim=True)
        features=torch.cat([base_log,evidence_log,(base_log-evidence_log).abs(),base_log*evidence_log],1)
        residual = self.relation(features)
        if self.residual_bound is not None:
            residual = self.residual_bound * torch.tanh(
                residual / self.residual_bound)
        return logits + residual
