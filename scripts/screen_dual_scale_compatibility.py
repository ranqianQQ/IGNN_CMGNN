"""Global validation screen for dual-scale compatibility correction."""
import argparse
import copy
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from ignn.models import IGNN
from ignn.modules.DualScaleCompatibilityAdapter import DualScaleCompatibilityAdapter
from ignn.modules.compatibility_propagation import (
    _row_normalized_adjacency,
    estimate_cmgnn_compatibility,
)
from scripts.compatibility_experiment_common import seed_all
from scripts.screen_compatibility_finetuning import (
    DATASETS, accuracy, masks_for, model_setup)


CANDIDATES = {
    "dual_h8_frozen": dict(hidden=8, tune_classifier=False, anchor=.1, gate=.1, classifier_lr=.1),
    "dual_h8_joint": dict(hidden=8, tune_classifier=True, anchor=.1, gate=.1, classifier_lr=.1),
    "dual_h16_frozen": dict(hidden=16, tune_classifier=False, anchor=.1, gate=.1, classifier_lr=.1),
    "dual_h8_anchor1": dict(hidden=8, tune_classifier=False, anchor=1., gate=.1, classifier_lr=.1),
    "dual_h8_gate02": dict(hidden=8, tune_classifier=False, anchor=.1, gate=.2, classifier_lr=.1),
    "dual_h8_joint_lr05": dict(hidden=8, tune_classifier=True, anchor=.1, gate=.1, classifier_lr=.05),
    "dual_h8_joint_lr02": dict(hidden=8, tune_classifier=True, anchor=.1, gate=.1, classifier_lr=.02),
    "dual_h8_joint_gate02": dict(hidden=8, tune_classifier=True, anchor=.1, gate=.2, classifier_lr=.1),
    "dual_h16_joint": dict(hidden=16, tune_classifier=True, anchor=.1, gate=.1, classifier_lr=.1),
}


def fine_tune(model, embeddings, adjacency, data, masks, candidate, params,
              seed, max_epochs=300, patience=70):
    seed_all(seed)
    model.classifier.eval()
    with torch.no_grad():
        raw = model.classifier(embeddings)
        cm, diagnostics = estimate_cmgnn_compatibility(
            raw, data.edge_index, data.y, masks[0])
    adapter = DualScaleCompatibilityAdapter(
        cm, hidden=candidate["hidden"], initial_gate=candidate["gate"]
    ).to(embeddings.device)
    groups = [{"params": adapter.parameters(), "lr": .01}]
    if candidate["tune_classifier"]:
        groups.insert(0, {
            "params": model.classifier.parameters(),
            "lr": params["lr"] * candidate["classifier_lr"]})
    else:
        for parameter in model.classifier.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(groups, weight_decay=params["l2_coef"])
    best_val, best_epoch, best_state = -1., -1, None
    started = time.perf_counter()
    for epoch in range(max_epochs):
        model.classifier.train(candidate["tune_classifier"])
        adapter.train()
        optimizer.zero_grad()
        raw = model.classifier(embeddings)
        corrected, global_logits, _ = adapter.components(
            raw, adjacency, data.y, masks[0])
        loss = F.cross_entropy(corrected[masks[0]], data.y[masks[0]])
        loss = loss + candidate["anchor"] * F.kl_div(
            corrected.log_softmax(dim=-1),
            global_logits.softmax(dim=-1).detach(), reduction="batchmean")
        if candidate["tune_classifier"]:
            loss = loss + .25 * F.cross_entropy(
                raw[masks[0]], data.y[masks[0]])
        loss = loss + 1e-3 * adapter.regularization()
        loss.backward()
        optimizer.step()
        model.classifier.eval()
        adapter.eval()
        with torch.no_grad():
            val = accuracy(adapter(
                model.classifier(embeddings), adjacency, data.y, masks[0]),
                data.y, masks[1])
        if val >= best_val:
            best_val, best_epoch = val, epoch
            best_state = {
                "classifier": copy.deepcopy(model.classifier.state_dict()),
                "adapter": copy.deepcopy(adapter.state_dict()),
            }
        if epoch - best_epoch >= patience:
            break
    model.classifier.load_state_dict(best_state["classifier"])
    adapter.load_state_dict(best_state["adapter"])
    return adapter, {
        "best_epoch": best_epoch + 1,
        "best_val_accuracy": best_val,
        "epochs": epoch + 1,
        "seconds": time.perf_counter() - started,
        "strength": float(adapter.strength().detach()),
        "gate": float(adapter.gate().detach()),
        "initial_cm_diagnostics": diagnostics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-end", type=int, default=3)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--candidates", nargs="+", default=list(CANDIDATES))
    parser.add_argument("--output", default="experiments/dual_scale_screen.json")
    args = parser.parse_args()
    device = torch.device("cuda:0")
    torch.set_num_threads(4)
    output = Path(args.output)
    result = {"protocol": "global_validation_only_dual_scale_screen",
              "test_labels_evaluated": False,
              "candidates": CANDIDATES, "records": []}
    for name in args.datasets:
        _, params, rn, data, config = model_setup(name, device)
        adjacency, _ = _row_normalized_adjacency(
            data.edge_index, data.num_nodes, device)
        for split in range(args.split_end):
            masks = masks_for(data, split, device)
            seed = 42 + split
            for candidate_name in args.candidates:
                candidate = CANDIDATES[candidate_name]
                seed_all(seed)
                model = IGNN(
                    data.num_features, n_clusters=int(data.y.max()) + 1,
                    IN="IN-SN", RN=rn, agg_type="gcn_incep", **params,
                ).to(device)
                model.load_state_dict(torch.load(
                    f"experiments/sfd_10split_{name}_{split}_tuned_sfd.pt",
                    map_location=device))
                model.eval()
                with torch.no_grad():
                    embeddings = model(data.edge_index, data.x, config, device).detach()
                    baseline = accuracy(
                        model.classifier(embeddings), data.y, masks[1])
                adapter, fit = fine_tune(
                    model, embeddings, adjacency, data, masks, candidate,
                    params, seed, args.max_epochs)
                record = {"dataset": name, "split": split,
                          "candidate": candidate_name,
                          "baseline_val_accuracy": baseline,
                          "val_gain_pp": 100 * (
                              fit["best_val_accuracy"] - baseline), **fit}
                result["records"].append(record)
                output.write_text(json.dumps(result, indent=2), encoding="utf-8")
                print({k: v for k, v in record.items()
                       if k != "initial_cm_diagnostics"}, flush=True)
                del model, adapter, embeddings
    ranking = []
    for candidate_name in args.candidates:
        rows = [r for r in result["records"]
                if r["candidate"] == candidate_name]
        ranking.append({
            "candidate": candidate_name,
            "mean_val_gain_pp": statistics.mean(
                r["val_gain_pp"] for r in rows),
            "positive_splits": sum(r["val_gain_pp"] > 1e-8 for r in rows),
            "total_splits": len(rows),
        })
    ranking.sort(key=lambda item: item["mean_val_gain_pp"], reverse=True)
    result["ranking"] = ranking
    result["selected_global_candidate"] = ranking[0]["candidate"]
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(ranking, indent=2), flush=True)


if __name__ == "__main__":
    main()
