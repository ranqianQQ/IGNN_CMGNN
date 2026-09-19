"""Train the fixed SFD backbone checkpoints required by compatibility heads."""
import argparse
import copy
from pathlib import Path

import torch

from ignn.configs import DataConf, INConf
from ignn.models import IGNN
from scripts.compatibility_experiment_common import (
    DATASETS, load_dataset, seed_all, set_sfd_optimizer, setting_for)
from utils import get_splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--split-start", type=int, default=0)
    parser.add_argument("--split-end", type=int, default=10)
    parser.add_argument("--max-epochs", type=int, default=3000)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    Path("experiments").mkdir(exist_ok=True)
    for dataset in args.datasets:
        params = setting_for(dataset)
        source = params.pop("source")
        rn = params.pop("candidate_rn")
        gate_multiplier = params.pop("gate_lr_multiplier")
        data = load_dataset(dataset, source).to(device)
        config = INConf(data.name, params["n_hops"], True, False, True,
                        False, params["fast"])
        for split in range(args.split_start, args.split_end):
            masks = tuple(mask.to(device) for mask in get_splits(
                data, data.name, data.num_nodes, split, 10, 48, 32,
                DataConf("data", "data/random_splits"), public=False))
            seed_all(42 + split)
            model = IGNN(
                data.num_features, n_clusters=int(data.y.max()) + 1,
                IN="IN-SN", RN=rn, agg_type="gcn_incep", **params).to(device)
            set_sfd_optimizer(
                model, params["lr"], gate_multiplier, params["l2_coef"])
            best_val, best_epoch, best_state = -1., -1, None
            for epoch in range(args.max_epochs):
                model.train(); model.optimizer.zero_grad()
                logits = model.classifier(
                    model(data.edge_index, data.x, config, device))
                loss = model.criterion(logits[masks[0]], data.y[masks[0]])
                loss.backward(); model.optimizer.step()
                model.eval()
                with torch.no_grad():
                    logits = model.classifier(
                        model(data.edge_index, data.x, config, device))
                    val = (logits[masks[1]].argmax(-1)
                           == data.y[masks[1]]).float().mean().item()
                if val >= best_val:
                    best_val, best_epoch = val, epoch
                    best_state = copy.deepcopy(model.state_dict())
                if epoch - best_epoch >= model.estop_steps:
                    break
            model.load_state_dict(best_state)
            path = Path("experiments") / (
                f"sfd_10split_{dataset}_{split}_tuned_sfd.pt")
            torch.save(model.state_dict(), path)
            print(dataset, split, "best_epoch", best_epoch + 1,
                  "best_val", best_val, "checkpoint", path, flush=True)


if __name__ == "__main__":
    main()
