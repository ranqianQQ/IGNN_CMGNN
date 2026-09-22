"""Shared settings and utilities for the retained SFD + GCC model."""
import hashlib
import random

import dgl
import numpy as np
import torch
from graph_datasets import load_data
from torch_geometric.data import Data

from ignn.configs import DataConf, INConf
from ignn.models import IGNN
from utils import get_splits


DATASETS = (
    "chameleon", "actor", "pubmed", "roman-empire",
    "squirrel", "photo", "amazon-ratings", "wikics",
)

SETTINGS = {
    "chameleon": dict(source="critical", baseline_rn="concat",
        candidate_rn="spectral_decoupling", h_feats=64, lr=.001,
        candidate_lr=.0005, gate_lr_multiplier=.25, l2_coef=0., n_hops=1,
        n_layers=5, early_stop=150, norm_type="none", act_type="none",
        pre_lin=True, fast=False, pre_dropout=.8, hid_dropout=.3,
        clf_dropout=.3),
    "actor": dict(source="pyg", baseline_rn="concat",
        candidate_rn="spectral_decoupling", h_feats=512, lr=.001,
        candidate_lr=.0005, gate_lr_multiplier=.25, l2_coef=0., n_hops=1,
        n_layers=1, early_stop=100, norm_type="ln", act_type="relu",
        pre_lin=False, fast=False, pre_dropout=0., hid_dropout=.8,
        clf_dropout=.9),
    "pubmed": dict(source="pyg", baseline_rn="concat",
        candidate_rn="spectral_decoupling", h_feats=128, lr=.01,
        candidate_lr=.01, gate_lr_multiplier=1., l2_coef=.0005, n_hops=6,
        n_layers=1, early_stop=100, norm_type="none", act_type="relu",
        pre_lin=False, fast=True, pre_dropout=.2, hid_dropout=.5,
        clf_dropout=.6),
    "roman-empire": dict(source="critical", baseline_rn="none",
        candidate_rn="spectral_decoupling_raw", h_feats=256, lr=.01,
        candidate_lr=.01, gate_lr_multiplier=4., l2_coef=5e-5, n_hops=1,
        n_layers=5, early_stop=200, norm_type="bn", act_type="relu",
        pre_lin=True, fast=False, pre_dropout=.5, hid_dropout=.2,
        clf_dropout=.4),
    "squirrel": dict(source="critical", baseline_rn="none",
        candidate_rn="spectral_decoupling_raw", h_feats=128, lr=.005,
        candidate_lr=.0025, gate_lr_multiplier=.25, l2_coef=0., n_hops=1,
        n_layers=3, early_stop=200, norm_type="none", act_type="relu",
        pre_lin=False, fast=False, pre_dropout=.8, hid_dropout=.2,
        clf_dropout=.8),
    "photo": dict(source="pyg", baseline_rn="none",
        candidate_rn="spectral_decoupling_raw", h_feats=512, lr=.001,
        candidate_lr=.001, gate_lr_multiplier=1., l2_coef=0., n_hops=3,
        n_layers=1, early_stop=150, norm_type="bn", act_type="relu",
        pre_lin=False, fast=False, pre_dropout=.5, hid_dropout=.6,
        clf_dropout=.2),
    "amazon-ratings": dict(source="critical", baseline_rn="none",
        candidate_rn="spectral_decoupling_raw", h_feats=512, lr=.005,
        candidate_lr=.005, gate_lr_multiplier=4., l2_coef=1e-5, n_hops=16,
        n_layers=1, early_stop=200, norm_type="ln", act_type="relu",
        pre_lin=True, fast=True, pre_dropout=0., hid_dropout=.8,
        clf_dropout=.8),
    "wikics": dict(source="local_dgl", baseline_rn="concat",
        candidate_rn="spectral_decoupling", h_feats=512, lr=.001,
        candidate_lr=.001, gate_lr_multiplier=1., l2_coef=5e-5, n_hops=8,
        n_layers=1, early_stop=200, norm_type="ln", act_type="relu",
        pre_lin=True, fast=True, pre_dropout=.2, hid_dropout=.5,
        clf_dropout=.6),
}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_dataset(name, source):
    if source != "local_dgl":
        return load_data(
            name, "data", 0, source, "pyg", False, False, True, True)
    graphs, _ = dgl.load_graphs("data/wikics_dgl.pt")
    graph = graphs[0]
    source_node, target_node = graph.edges()
    data = Data(
        x=graph.ndata["feat"], y=graph.ndata["label"].long(),
        edge_index=torch.stack((source_node, target_node)))
    data.name = "wikics_pyg"
    return data


def masks_for(data, split, device):
    masks = get_splits(
        data, data.name, data.num_nodes, split, 10, 48, 32,
        DataConf("data", "data/random_splits"), public=False)
    return tuple(mask.to(device) for mask in masks)


def split_hash(masks):
    return hashlib.sha256(
        b"".join(mask.cpu().numpy().tobytes() for mask in masks)).hexdigest()


def set_gate_optimizer(model, base_lr, gate_multiplier, weight_decay):
    gates, backbone = [], []
    for name, parameter in model.named_parameters():
        (gates if name.endswith("detail_gates") else backbone).append(parameter)
    model.optimizer = torch.optim.Adam([
        {"params": backbone, "lr": base_lr},
        {"params": gates, "lr": base_lr * gate_multiplier},
    ], weight_decay=weight_decay)


def shared_initialization(base, enhanced):
    state = enhanced.state_dict()
    for key, value in base.state_dict().items():
        target = key.replace(".nei_rel_learn.", ".nei_rel_learn.base.")
        if target not in state or state[target].shape != value.shape:
            raise ValueError(f"Cannot map shared parameter: {key}")
        state[target] = value.clone()
    enhanced.load_state_dict(state)


def model_configuration(name, official=False):
    setting = dict(SETTINGS[name])
    source = setting.pop("source")
    baseline_rn = setting.pop("baseline_rn")
    candidate_rn = setting.pop("candidate_rn")
    candidate_lr = setting.pop("candidate_lr")
    gate_multiplier = setting.pop("gate_lr_multiplier")
    setting["lr"] = setting["lr"] if official else candidate_lr
    return setting, source, baseline_rn if official else candidate_rn, \
        gate_multiplier


def build_model(name, data, device, official=False):
    setting, _, rn, gate_multiplier = model_configuration(
        name, official)
    model = IGNN(
        data.num_features, n_clusters=int(data.y.max()) + 1,
        IN="IN-SN", RN=rn, agg_type="gcn_incep", **setting).to(device)
    if not official:
        set_gate_optimizer(
            model, setting["lr"], gate_multiplier, setting["l2_coef"])
    return setting, model


def data_setup(name, device):
    setting, source, _, _ = model_configuration(name)
    data = load_dataset(name, source).to(device)
    config = INConf(
        data.name, setting["n_hops"], symm_norm=True, fast=setting["fast"])
    return data, config


def model_setup(name, device, official=False):
    data, config = data_setup(name, device)
    setting, _, _, _ = model_configuration(name, official)
    _, model = build_model(name, data, device, official)
    return setting, data, config, model


def accuracy(logits, labels, mask):
    return (logits[mask].argmax(-1) == labels[mask]).float().mean().item()
