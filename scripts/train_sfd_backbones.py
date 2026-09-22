"""Train official IGNN and the retained SFD backbone on fixed splits."""
import argparse
import copy
import json
import time
from pathlib import Path

import torch

from scripts.final_common import (
    DATASETS, SETTINGS, accuracy, build_model, data_setup, masks_for, seed_all,
    shared_initialization, split_hash,
)


def fit(model, data, masks, config, seed, device, max_epochs):
    seed_all(seed)
    best = (-1., -1, None)
    started = time.perf_counter()
    for epoch in range(max_epochs):
        model.train()
        model.optimizer.zero_grad()
        logits = model.classifier(model(data.edge_index, data.x, config, device))
        loss = model.criterion(logits[masks[0]], data.y[masks[0]])
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite training loss")
        loss.backward()
        model.optimizer.step()
        model.eval()
        with torch.no_grad():
            logits = model.classifier(
                model(data.edge_index, data.x, config, device))
            value = accuracy(logits, data.y, masks[1])
        if value >= best[0]:
            best = (value, epoch, copy.deepcopy(model.state_dict()))
        if epoch - best[1] >= model.estop_steps:
            break
    model.load_state_dict(best[2])
    return best[1] + 1, best[0], epoch + 1, time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS,
                        default=list(DATASETS))
    parser.add_argument("--split-start", type=int, default=0)
    parser.add_argument("--split-end", type=int, default=10)
    parser.add_argument("--max-epochs", type=int, default=3000)
    parser.add_argument("--output",
                        default="experiments/sfd_backbone_10splits.json")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(4)
    output = Path(args.output)
    result = {
        "protocol": "official_ignn_vs_sfd_fixed_splits",
        "selection": "validation accuracy only; latest epoch on ties",
        "test_labels_used_for_training_or_selection": False,
        "settings": SETTINGS,
        "records": [],
    }
    for dataset in args.datasets:
        data, config = data_setup(dataset, device)
        for split in range(args.split_start, args.split_end):
            masks = masks_for(data, split, device)
            seed = 42 + split
            seed_all(seed)
            _, official = build_model(dataset, data, device, official=True)
            _, sfd = build_model(dataset, data, device, official=False)
            shared_initialization(official, sfd)
            with torch.no_grad():
                official.eval()
                sfd.eval()
                base = official.classifier(
                    official(data.edge_index, data.x, config, device))
                enhanced = sfd.classifier(
                    sfd(data.edge_index, data.x, config, device))
                torch.testing.assert_close(base, enhanced, rtol=0, atol=0)
            for name, model in (("official_ignn", official),
                                ("tuned_sfd", sfd)):
                best_epoch, val, epochs, seconds = fit(
                    model, data, masks, config, seed, device, args.max_epochs)
                model.eval()
                with torch.no_grad():
                    logits = model.classifier(
                        model(data.edge_index, data.x, config, device))
                    test = accuracy(logits, data.y, masks[2])
                record = {
                    "dataset": dataset,
                    "split": split,
                    "seed": seed,
                    "model": name,
                    "split_hash": split_hash(masks),
                    "best_epoch": best_epoch,
                    "best_val_accuracy": val,
                    "test_accuracy": test,
                    "epochs": epochs,
                    "training_seconds": seconds,
                    "parameters": sum(p.numel() for p in model.parameters()),
                }
                if name == "tuned_sfd":
                    checkpoint = output.parent / (
                        f"sfd_10split_{dataset}_{split}_{name}.pt")
                    torch.save(model.state_dict(), checkpoint)
                    record["checkpoint"] = checkpoint.name
                result["records"].append(record)
                output.write_text(
                    json.dumps(result, indent=2), encoding="utf-8")
                print(dataset, split, name, f"test={test:.4f}", flush=True)


if __name__ == "__main__":
    main()
