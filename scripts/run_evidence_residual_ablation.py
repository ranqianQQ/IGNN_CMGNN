"""Validation screen and ten-split ablation for evidence-residual dual scale."""
import argparse
import copy
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from ignn.models import IGNN
from ignn.modules.EvidenceResidualDualScaleAdapter import (
    EvidenceResidualDualScaleAdapter,
)
from ignn.modules.compatibility_propagation import (
    _row_normalized_adjacency,
    estimate_cmgnn_compatibility,
)
from scripts.compatibility_experiment_common import seed_all
from scripts.screen_compatibility_finetuning import (
    DATASETS, accuracy, masks_for, model_setup,
)


CANDIDATES = {
    "scalar_h8_bound025": dict(hidden=8, classwise_gate=False, local_scale=.25,
                      anchor=.1, classifier_lr=.1),
    "scalar_h8_bound05": dict(hidden=8, classwise_gate=False, local_scale=.5,
                       anchor=.1, classifier_lr=.1),
    "scalar_h8_bound1": dict(hidden=8, classwise_gate=False, local_scale=1.,
                              anchor=.1, classifier_lr=.1),
    "scalar_h16_bound05": dict(hidden=16, classwise_gate=False, local_scale=.5,
                           anchor=.1, classifier_lr=.05),
    "classwise_h8_bound05": dict(hidden=8, classwise_gate=True, local_scale=.5,
                         anchor=.1, classifier_lr=.1),
}
ABLATIONS = ("classifier_only", "global", "local", "dual")


def mean_std(values):
    return {"mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.}


def trimmed_dataset_mean(dataset_values):
    """Drop the lowest and highest dataset means before averaging."""
    ordered = sorted(dataset_values.items(), key=lambda item: item[1])
    kept = ordered[1:-1] if len(ordered) > 2 else ordered
    return {
        "mean": statistics.mean(value for _, value in kept),
        "included": [name for name, _ in kept],
        "dropped_low": ordered[0][0] if len(ordered) > 2 else None,
        "dropped_high": ordered[-1][0] if len(ordered) > 2 else None,
    }


def train_variant(model, embeddings, adjacency, data, masks, candidate,
                  params, seed, mode, max_epochs=300, patience=70):
    seed_all(seed)
    model.classifier.eval()
    with torch.no_grad():
        initial_logits = model.classifier(embeddings)
        compatibility, diagnostics = estimate_cmgnn_compatibility(
            initial_logits, data.edge_index, data.y, masks[0])

    adapter = None
    if mode != "classifier_only":
        adapter = EvidenceResidualDualScaleAdapter(
            compatibility,
            hidden=candidate["hidden"],
            max_local_strength=candidate["local_scale"],
            classwise_gate=candidate["classwise_gate"],
        ).to(embeddings.device)
        if mode == "global":
            for parameter in adapter.local_scorer.parameters():
                parameter.requires_grad_(False)
        elif mode == "local":
            adapter.raw_global_strength.requires_grad_(False)

    classifier_parameters = list(model.classifier.parameters())
    trainable = list(classifier_parameters)
    groups = [{"params": classifier_parameters,
               "lr": params["lr"] * candidate["classifier_lr"]}]
    if adapter is not None:
        adapter_parameters = [p for p in adapter.parameters() if p.requires_grad]
        groups.append({"params": adapter_parameters, "lr": .01})
        trainable += adapter_parameters
    optimizer = torch.optim.Adam(groups, weight_decay=params["l2_coef"])

    best_val, best_epoch, best_state = -1., -1, None
    started = time.perf_counter()
    for epoch in range(max_epochs):
        model.classifier.train()
        if adapter is not None:
            adapter.train()
        optimizer.zero_grad()
        raw = model.classifier(embeddings)
        if adapter is None:
            corrected, reference = raw, raw
        else:
            components = adapter.components(raw, adjacency, data.y, masks[0])
            corrected = components[mode]
            reference = raw if mode == "local" else components["global"]
        loss = F.cross_entropy(corrected[masks[0]], data.y[masks[0]])
        if mode != "classifier_only":
            loss = loss + .25 * F.cross_entropy(
                raw[masks[0]], data.y[masks[0]])
            if mode in ("local", "dual"):
                loss = loss + candidate["anchor"] * F.kl_div(
                    corrected.log_softmax(dim=-1),
                    reference.softmax(dim=-1).detach(), reduction="batchmean")
            loss = loss + 1e-3 * adapter.regularization()
        loss.backward()
        optimizer.step()

        model.classifier.eval()
        if adapter is not None:
            adapter.eval()
        with torch.no_grad():
            raw = model.classifier(embeddings)
            scores = (raw if adapter is None else adapter(
                raw, adjacency, data.y, masks[0], mode=mode))
            val = accuracy(scores, data.y, masks[1])
        if val >= best_val:
            best_val, best_epoch = val, epoch
            best_state = {
                "classifier": copy.deepcopy(model.classifier.state_dict()),
                "adapter": (None if adapter is None else
                            copy.deepcopy(adapter.state_dict())),
            }
        if epoch - best_epoch >= patience:
            break

    model.classifier.load_state_dict(best_state["classifier"])
    if adapter is not None:
        adapter.load_state_dict(best_state["adapter"])
    statistics_out = {
        "best_epoch": best_epoch + 1,
        "best_val_accuracy": best_val,
        "epochs": epoch + 1,
        "seconds": time.perf_counter() - started,
        "trainable_parameters": sum(p.numel() for p in trainable),
        "initial_cm_diagnostics": diagnostics,
    }
    if adapter is not None:
        adapter.eval()
        with torch.no_grad():
            raw = model.classifier(embeddings)
            components = adapter.components(raw, adjacency, data.y, masks[0])
            coefficient = components["local_coefficient"]
        statistics_out.update({
            "global_strength": float(adapter.global_strength().detach()),
            "local_coefficient_mean": float(coefficient.mean()),
            "local_coefficient_abs_mean": float(coefficient.abs().mean()),
            "local_coefficient_std": float(coefficient.std()),
            "compatibility_row_sum_max_error": float(
                (adapter.compatibility().sum(dim=1) - 1.).abs().max()),
        })
    return adapter, statistics_out


def initialize_split(name, split, device):
    _, params, rn, data, config = model_setup(name, device)
    masks = masks_for(data, split, device)
    seed = 42 + split
    seed_all(seed)
    model = IGNN(
        data.num_features, n_clusters=int(data.y.max()) + 1,
        IN="IN-SN", RN=rn, agg_type="gcn_incep", **params,
    ).to(device)
    checkpoint = Path(
        f"experiments/sfd_10split_{name}_{split}_tuned_sfd.pt")
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    with torch.no_grad():
        embeddings = model(data.edge_index, data.x, config, device).detach()
        raw = model.classifier(embeddings)
        baseline_val = accuracy(raw, data.y, masks[1])
        baseline_test = accuracy(raw, data.y, masks[2])
    adjacency, _ = _row_normalized_adjacency(
        data.edge_index, data.num_nodes, device)
    return (params, data, masks, seed, model, embeddings, adjacency,
            checkpoint, baseline_val, baseline_test)


def screen(args, device):
    result = {
        "protocol": "cross_dataset_validation_only_screen",
        "selection_metric": (
            "mean validation gain across splits 0-2 per dataset, then drop "
            "the highest and lowest dataset means and average the remaining"),
        "test_labels_evaluated": False,
        "candidates": CANDIDATES,
        "records": [],
    }
    output = Path(args.output)
    for name in args.datasets:
        for split in range(args.split_end):
            state = initialize_split(name, split, device)
            (params, data, masks, seed, model, embeddings, adjacency,
             _, baseline_val, _) = state
            classifier_state = copy.deepcopy(model.classifier.state_dict())
            for candidate_name in args.candidates:
                model.classifier.load_state_dict(classifier_state)
                for parameter in model.classifier.parameters():
                    parameter.requires_grad_(True)
                _, fit = train_variant(
                    model, embeddings, adjacency, data, masks,
                    CANDIDATES[candidate_name], params, seed, "dual",
                    args.max_epochs, args.patience)
                record = {
                    "dataset": name, "split": split,
                    "candidate": candidate_name,
                    "baseline_val_accuracy": baseline_val,
                    "val_gain_pp": 100 * (
                        fit["best_val_accuracy"] - baseline_val),
                    **fit,
                }
                result["records"].append(record)
                output.write_text(json.dumps(result, indent=2), encoding="utf-8")
                print("SCREEN", name, split, candidate_name,
                      f"val_gain={record['val_gain_pp']:.4f}", flush=True)
            del model, embeddings, adjacency

    ranking = []
    for candidate_name in args.candidates:
        rows = [r for r in result["records"]
                if r["candidate"] == candidate_name]
        dataset_means = {
            name: statistics.mean(r["val_gain_pp"] for r in rows
                                  if r["dataset"] == name)
            for name in args.datasets
        }
        trimmed = trimmed_dataset_mean(dataset_means)
        ranking.append({
            "candidate": candidate_name,
            "dataset_mean_val_gain_pp": dataset_means,
            "macro_mean_val_gain_pp": statistics.mean(dataset_means.values()),
            "trimmed_macro_mean_val_gain_pp": trimmed["mean"],
            "trimmed_details": trimmed,
        })
    ranking.sort(key=lambda row: row["trimmed_macro_mean_val_gain_pp"],
                 reverse=True)
    result["ranking"] = ranking
    result["selected_candidate"] = ranking[0]["candidate"]
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(ranking, indent=2), flush=True)


def evaluate(args, device):
    screen_result = json.loads(Path(args.screen_result).read_text(encoding="utf-8"))
    selected_name = args.selected or screen_result["selected_candidate"]
    candidate = CANDIDATES[selected_name]
    output = Path(args.output)
    result = {
        "protocol": "same_backbone_same_split_same_training_protocol_ablation",
        "selection": (
            f"{selected_name} selected only by trimmed cross-dataset mean "
            "validation gain on splits 0-2"),
        "selected_candidate": selected_name,
        "candidate": candidate,
        "modes": ["sfd", *ABLATIONS],
        "test_labels_used_for_training_or_selection": False,
        "records": [],
    }
    for name in args.datasets:
        for split in range(10):
            state = initialize_split(name, split, device)
            (params, data, masks, seed, model, embeddings, adjacency,
             checkpoint, baseline_val, baseline_test) = state
            classifier_state = copy.deepcopy(model.classifier.state_dict())
            split_hash = hashlib.sha256(b"".join(
                mask.cpu().numpy().tobytes() for mask in masks)).hexdigest()
            result["records"].append({
                "dataset": name, "split": split, "seed": seed,
                "split_hash": split_hash, "mode": "sfd",
                "warmup_checkpoint": checkpoint.name,
                "best_val_accuracy": baseline_val,
                "test_accuracy": baseline_test,
                "test_gain_pp": 0., "best_epoch": 0,
            })
            for mode in ABLATIONS:
                model.classifier.load_state_dict(classifier_state)
                for parameter in model.classifier.parameters():
                    parameter.requires_grad_(True)
                adapter, fit = train_variant(
                    model, embeddings, adjacency, data, masks, candidate,
                    params, seed, mode, args.max_epochs, args.patience)
                model.classifier.eval()
                if adapter is not None:
                    adapter.eval()
                with torch.no_grad():
                    raw = model.classifier(embeddings)
                    scores = (raw if adapter is None else adapter(
                        raw, adjacency, data.y, masks[0], mode=mode))
                    test_accuracy = accuracy(scores, data.y, masks[2])
                record = {
                    "dataset": name, "split": split, "seed": seed,
                    "split_hash": split_hash, "mode": mode,
                    "warmup_checkpoint": checkpoint.name,
                    "baseline_val_accuracy": baseline_val,
                    "baseline_test_accuracy": baseline_test,
                    "test_accuracy": test_accuracy,
                    "test_gain_pp": 100 * (test_accuracy - baseline_test),
                    **fit,
                }
                result["records"].append(record)
                output.write_text(json.dumps(result, indent=2), encoding="utf-8")
                print("EVAL", name, split, mode,
                      f"gain={record['test_gain_pp']:.4f}", flush=True)
            del model, embeddings, adjacency

    summary = {}
    for mode in ABLATIONS:
        by_dataset = {}
        for name in args.datasets:
            rows = [r for r in result["records"]
                    if r["mode"] == mode and r["dataset"] == name]
            gains = [r["test_gain_pp"] for r in rows]
            by_dataset[name] = {
                "test_accuracy": mean_std(
                    [100 * r["test_accuracy"] for r in rows]),
                "paired_test_gain_pp": mean_std(gains),
                "wins_ties_losses": [
                    sum(x > 1e-8 for x in gains),
                    sum(abs(x) <= 1e-8 for x in gains),
                    sum(x < -1e-8 for x in gains),
                ],
            }
        dataset_gains = {
            name: values["paired_test_gain_pp"]["mean"]
            for name, values in by_dataset.items()
        }
        summary[mode] = {
            "datasets": by_dataset,
            "macro_mean_paired_gain_pp": statistics.mean(dataset_gains.values()),
            "trimmed_macro_mean_paired_gain_pp":
                trimmed_dataset_mean(dataset_gains),
        }
    result["summary"] = summary
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("screen", "evaluate"), required=True)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=70)
    parser.add_argument("--split-end", type=int, default=3)
    parser.add_argument("--candidates", nargs="+", default=list(CANDIDATES))
    parser.add_argument("--selected", choices=tuple(CANDIDATES))
    parser.add_argument("--screen-result",
                        default="experiments/evidence_residual_screen.json")
    parser.add_argument("--output")
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if args.output is None:
        args.output = ("experiments/evidence_residual_screen.json"
                       if args.stage == "screen" else
                       "experiments/evidence_residual_ablation_10splits.json")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if args.stage == "screen":
        screen(args, device)
    else:
        evaluate(args, device)


if __name__ == "__main__":
    main()
