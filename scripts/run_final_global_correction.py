"""Train the retained fixed global compatibility correction on SFD models."""
import argparse
import copy
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from ignn.modules.GlobalCompatibilityCorrection import (
    GlobalCompatibilityCorrection,
)
from ignn.modules.compatibility_propagation import (
    _row_normalized_adjacency, estimate_cmgnn_compatibility,
)
from scripts.final_common import (
    DATASETS, accuracy, masks_for, model_setup, seed_all, split_hash,
)


def mean_std(values):
    return {"mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.}


def train(model, embeddings, adjacency, data, masks, params, seed,
          max_epochs=300, patience=70):
    seed_all(seed)
    model.classifier.eval()
    with torch.no_grad():
        initial = model.classifier(embeddings)
        matrix, diagnostics = estimate_cmgnn_compatibility(
            initial, data.edge_index, data.y, masks[0])
    correction = GlobalCompatibilityCorrection(matrix).to(embeddings.device)
    optimizer = torch.optim.Adam([
        {"params": model.classifier.parameters(), "lr": params["lr"] * .1},
        {"params": correction.parameters(), "lr": .01},
    ], weight_decay=params["l2_coef"])
    best = (-1., -1, None, None)
    started = time.perf_counter()
    for epoch in range(max_epochs):
        model.classifier.train()
        correction.train()
        optimizer.zero_grad()
        raw = model.classifier(embeddings)
        corrected = correction(raw, adjacency, data.y, masks[0])
        loss = (F.cross_entropy(corrected[masks[0]], data.y[masks[0]])
                + .25 * F.cross_entropy(raw[masks[0]], data.y[masks[0]])
                + 1e-3 * correction.regularization())
        loss.backward()
        optimizer.step()
        model.classifier.eval()
        correction.eval()
        with torch.no_grad():
            raw = model.classifier(embeddings)
            value = accuracy(
                correction(raw, adjacency, data.y, masks[0]),
                data.y, masks[1])
        if value >= best[0]:
            best = (value, epoch, copy.deepcopy(model.classifier.state_dict()),
                    copy.deepcopy(correction.state_dict()))
        if epoch - best[1] >= patience:
            break
    model.classifier.load_state_dict(best[2])
    correction.load_state_dict(best[3])
    row_sum_error = (correction.compatibility().sum(dim=1) - 1.).abs().max()
    return correction, {
        "best_epoch": best[1] + 1,
        "best_val_accuracy": best[0],
        "epochs": epoch + 1,
        "training_seconds": time.perf_counter() - started,
        "trainable_parameters": (
            sum(parameter.numel() for parameter in model.classifier.parameters())
            + sum(parameter.numel() for parameter in correction.parameters())
        ),
        "strength": float(correction.strength().detach()),
        "compatibility_diagnostics": diagnostics,
        "compatibility_row_sum_max_error": float(row_sum_error.detach()),
    }


def summarize(records):
    datasets = {}
    for dataset in DATASETS:
        rows = [row for row in records if row["dataset"] == dataset]
        gains = [row["test_gain_pp"] for row in rows]
        datasets[dataset] = {
            "test_accuracy": mean_std(
                [100 * row["test_accuracy"] for row in rows]),
            "paired_test_gain_pp": mean_std(gains),
            "wins_ties_losses": [
                sum(value > 1e-7 for value in gains),
                sum(abs(value) <= 1e-7 for value in gains),
                sum(value < -1e-7 for value in gains),
            ],
        }
    means = {name: value["paired_test_gain_pp"]["mean"]
             for name, value in datasets.items()}
    ordered = sorted(means.items(), key=lambda item: item[1])
    kept = ordered[1:-1]
    return {
        "datasets": datasets,
        "macro_mean_paired_gain_pp": statistics.mean(means.values()),
        "trimmed_macro_mean_paired_gain_pp": {
            "mean": statistics.mean(value for _, value in kept),
            "included": [name for name, _ in kept],
            "dropped_low": ordered[0][0],
            "dropped_high": ordered[-1][0],
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS,
                        default=list(DATASETS))
    parser.add_argument("--split-start", type=int, default=0)
    parser.add_argument("--split-end", type=int, default=10)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=70)
    parser.add_argument(
        "--output",
        default="experiments/final_global_correction_10splits.json")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(4)
    output = Path(args.output)
    result = {
        "protocol": "fixed_global_compatibility_8_datasets_10_splits",
        "selection": "validation accuracy only; latest epoch on ties",
        "test_labels_used_for_training_or_selection": False,
        "configuration": {
            "initial_strength": .1,
            "max_strength": 4.,
            "classifier_lr_multiplier": .1,
            "correction_lr": .01,
            "matrix_regularization": 1e-3,
        },
        "records": [],
    }
    for dataset in args.datasets:
        params, data, config, model = model_setup(dataset, device)
        adjacency, _ = _row_normalized_adjacency(
            data.edge_index, data.num_nodes, device)
        for split in range(args.split_start, args.split_end):
            masks = masks_for(data, split, device)
            seed = 42 + split
            checkpoint = Path(
                f"experiments/sfd_10split_{dataset}_{split}_tuned_sfd.pt")
            model.load_state_dict(torch.load(checkpoint, map_location=device))
            model.eval()
            with torch.no_grad():
                embeddings = model(
                    data.edge_index, data.x, config, device).detach()
                raw = model.classifier(embeddings)
                baseline_val = accuracy(raw, data.y, masks[1])
                baseline_test = accuracy(raw, data.y, masks[2])
            correction, row = train(
                model, embeddings, adjacency, data, masks, params, seed,
                args.max_epochs, args.patience)
            model.eval()
            correction.eval()
            with torch.no_grad():
                raw = model.classifier(embeddings)
                corrected = correction(raw, adjacency, data.y, masks[0])
                test = accuracy(corrected, data.y, masks[2])
            row.update(
                dataset=dataset,
                split=split,
                seed=seed,
                split_hash=split_hash(masks),
                baseline_val_accuracy=baseline_val,
                baseline_test_accuracy=baseline_test,
                test_accuracy=test,
                test_gain_pp=100 * (test - baseline_test),
                warmup_checkpoint=checkpoint.name,
            )
            result["records"].append(row)
            output.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(dataset, split, f"gain={row['test_gain_pp']:+.4f}",
                  flush=True)
    if set(args.datasets) == set(DATASETS) and args.split_start == 0 \
            and args.split_end == 10:
        result["summary"] = summarize(result["records"])
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
