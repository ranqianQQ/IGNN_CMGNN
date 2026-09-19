"""Ten-split evaluation of the selected dual-scale compatibility adapter."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch

from ignn.models import IGNN
from ignn.modules.compatibility_propagation import _row_normalized_adjacency
from scripts.compatibility_experiment_common import seed_all
from scripts.screen_dual_scale_compatibility import CANDIDATES, fine_tune
from scripts.screen_compatibility_finetuning import (
    DATASETS, accuracy, masks_for, model_setup)


def mean_std(values):
    return {"mean": statistics.mean(values), "std": statistics.stdev(values)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--candidate", choices=tuple(CANDIDATES),
                        default="dual_h8_joint")
    parser.add_argument("--output", default="experiments/dual_scale_10splits.json")
    args = parser.parse_args()
    selected = CANDIDATES[args.candidate]
    device = torch.device("cuda:0")
    torch.set_num_threads(4)
    output = Path(args.output)
    result = {
        "protocol": "fixed_global_configuration_official_10_splits",
        "selection": (
            f"{args.candidate} was included in the global validation screen "
            "on splits 0-2 across all eight datasets; test unseen"),
        "selected_name": args.candidate,
        "selected": selected,
        "records": [],
    }
    for name in args.datasets:
        _, params, rn, data, config = model_setup(name, device)
        adjacency, _ = _row_normalized_adjacency(
            data.edge_index, data.num_nodes, device)
        for split in range(10):
            masks = masks_for(data, split, device)
            seed = 42 + split
            seed_all(seed)
            model = IGNN(
                data.num_features, n_clusters=int(data.y.max()) + 1,
                IN="IN-SN", RN=rn, agg_type="gcn_incep", **params,
            ).to(device)
            warm = Path(f"experiments/sfd_10split_{name}_{split}_tuned_sfd.pt")
            model.load_state_dict(torch.load(warm, map_location=device))
            model.eval()
            with torch.no_grad():
                embeddings = model(data.edge_index, data.x, config, device).detach()
                raw = model.classifier(embeddings)
                base_val = accuracy(raw, data.y, masks[1])
                base_test = accuracy(raw, data.y, masks[2])
            adapter, fit = fine_tune(
                model, embeddings, adjacency, data, masks, selected, params,
                seed, args.max_epochs)
            model.classifier.eval()
            adapter.eval()
            with torch.no_grad():
                raw = model.classifier(embeddings)
                scores = adapter(raw, adjacency, data.y, masks[0])
                val = accuracy(scores, data.y, masks[1])
                test = accuracy(scores, data.y, masks[2])
            checkpoint = output.parent / (
                f"dsca_{args.candidate}_10split_{name}_{split}.pt")
            torch.save({"classifier": model.classifier.state_dict(),
                        "adapter": adapter.state_dict(),
                        "warmup_checkpoint": warm.name,
                        "selected": selected}, checkpoint)
            record = {
                "dataset": name, "split": split, "seed": seed,
                "split_hash": hashlib.sha256(b"".join(
                    m.cpu().numpy().tobytes() for m in masks)).hexdigest(),
                "warmup_checkpoint": warm.name,
                "checkpoint": checkpoint.name,
                "baseline_val_accuracy": base_val,
                "best_val_accuracy": val,
                "val_gain_pp": 100 * (val - base_val),
                "baseline_test_accuracy": base_test,
                "test_accuracy": test,
                "test_gain_pp": 100 * (test - base_test),
                "adapter_parameters": sum(p.numel() for p in adapter.parameters()),
                **fit,
            }
            result["records"].append(record)
            output.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print({k: v for k, v in record.items()
                   if k != "initial_cm_diagnostics"}, flush=True)
            del model, adapter, embeddings
    summary = {}
    for name in args.datasets:
        rows = sorted((r for r in result["records"] if r["dataset"] == name),
                      key=lambda r: r["split"])
        gains = [r["test_gain_pp"] for r in rows]
        summary[name] = {
            "baseline_test": mean_std(
                [100 * r["baseline_test_accuracy"] for r in rows]),
            "dual_scale_test": mean_std(
                [100 * r["test_accuracy"] for r in rows]),
            "paired_test_gain_pp": mean_std(gains),
            "wins_ties_losses": [sum(x > 1e-8 for x in gains),
                                  sum(abs(x) <= 1e-8 for x in gains),
                                  sum(x < -1e-8 for x in gains)],
            "mean_val_gain_pp": statistics.mean(
                r["val_gain_pp"] for r in rows),
            "mean_gate": statistics.mean(r["gate"] for r in rows),
            "mean_strength": statistics.mean(r["strength"] for r in rows),
        }
    result["summary"] = summary
    result["macro_average_paired_test_gain_pp"] = statistics.mean(
        item["paired_test_gain_pp"]["mean"] for item in summary.values())
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary,
                      "macro_average_paired_test_gain_pp":
                          result["macro_average_paired_test_gain_pp"]},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
