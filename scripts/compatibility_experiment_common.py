"""Shared, dataset-name-independent experiment plumbing and fixed settings."""
import random

import numpy as np
import torch
from graph_datasets import load_data
from torch_geometric.data import Data


DATASETS = (
    "chameleon", "actor", "pubmed", "roman-empire",
    "squirrel", "photo", "amazon-ratings", "wikics",
)

# These are the fixed SFD settings used to create the released checkpoints.
# Compatibility heads use one shared architecture/search protocol on top.
SETTINGS = {
    "chameleon": dict(source="critical", candidate_rn="spectral_decoupling",
        h_feats=64, lr=.0005, l2_coef=0., n_hops=1, n_layers=5,
        early_stop=150, norm_type="none", act_type="none", pre_lin=True,
        fast=False, pre_dropout=.8, hid_dropout=.3, clf_dropout=.3,
        gate_lr_multiplier=.25),
    "actor": dict(source="pyg", candidate_rn="spectral_decoupling",
        h_feats=512, lr=.0005, l2_coef=0., n_hops=1, n_layers=1,
        early_stop=100, norm_type="ln", act_type="relu", pre_lin=False,
        fast=False, pre_dropout=0., hid_dropout=.8, clf_dropout=.9,
        gate_lr_multiplier=.25),
    "pubmed": dict(source="pyg", candidate_rn="spectral_decoupling",
        h_feats=128, lr=.01, l2_coef=.0005, n_hops=6, n_layers=1,
        early_stop=100, norm_type="none", act_type="relu", pre_lin=False,
        fast=True, pre_dropout=.2, hid_dropout=.5, clf_dropout=.6,
        gate_lr_multiplier=1.),
    "roman-empire": dict(source="critical", candidate_rn="spectral_decoupling_raw",
        h_feats=256, lr=.01, l2_coef=5e-5, n_hops=1, n_layers=5,
        early_stop=200, norm_type="bn", act_type="relu", pre_lin=True,
        fast=False, pre_dropout=.5, hid_dropout=.2, clf_dropout=.4,
        gate_lr_multiplier=4.),
    "squirrel": dict(source="critical", candidate_rn="spectral_decoupling_raw",
        h_feats=128, lr=.0025, l2_coef=0., n_hops=1, n_layers=3,
        early_stop=200, norm_type="none", act_type="relu", pre_lin=False,
        fast=False, pre_dropout=.8, hid_dropout=.2, clf_dropout=.8,
        gate_lr_multiplier=.25),
    "photo": dict(source="pyg", candidate_rn="spectral_decoupling_raw",
        h_feats=512, lr=.001, l2_coef=0., n_hops=3, n_layers=1,
        early_stop=150, norm_type="bn", act_type="relu", pre_lin=False,
        fast=False, pre_dropout=.5, hid_dropout=.6, clf_dropout=.2,
        gate_lr_multiplier=1.),
    "amazon-ratings": dict(source="critical", candidate_rn="spectral_decoupling_raw",
        h_feats=512, lr=.005, l2_coef=1e-5, n_hops=16, n_layers=1,
        early_stop=200, norm_type="ln", act_type="relu", pre_lin=True,
        fast=True, pre_dropout=0., hid_dropout=.8, clf_dropout=.8,
        gate_lr_multiplier=4.),
    "wikics": dict(source="local_dgl", candidate_rn="spectral_decoupling",
        h_feats=512, lr=.001, l2_coef=5e-5, n_hops=8, n_layers=1,
        early_stop=200, norm_type="ln", act_type="relu", pre_lin=True,
        fast=True, pre_dropout=.2, hid_dropout=.5, clf_dropout=.6,
        gate_lr_multiplier=1.),
}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setting_for(name):
    return dict(SETTINGS[name])


def load_dataset(name, source, row_norm=False):
    if source != "local_dgl":
        return load_data(
            name, "data", 0, source, "pyg", row_norm, False, True, True)
    import dgl
    graphs, _ = dgl.load_graphs("data/wikics_dgl.pt")
    graph = graphs[0]
    src, dst = graph.edges()
    data = Data(x=graph.ndata["feat"], y=graph.ndata["label"].long(),
                edge_index=torch.stack((src, dst)))
    data.name = "wikics_pyg"
    return data


def set_sfd_optimizer(model, base_lr, gate_multiplier, weight_decay):
    gates, backbone = [], []
    for name, parameter in model.named_parameters():
        (gates if name.endswith("detail_gates") else backbone).append(parameter)
    model.optimizer = torch.optim.Adam([
        {"params": backbone, "lr": base_lr},
        {"params": gates, "lr": base_lr * gate_multiplier},
    ], weight_decay=weight_decay)
