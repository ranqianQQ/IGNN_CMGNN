"""Select global/local/dual compatibility scales using validation only."""
import argparse
import csv
import json
import statistics
from pathlib import Path


MODES = ("global", "local", "dual")


def mean_std(values):
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.,
    }


def trimmed(dataset_values):
    ordered = sorted(dataset_values.items(), key=lambda item: item[1])
    kept = ordered[1:-1]
    return {
        "mean": statistics.mean(value for _, value in kept),
        "included": [name for name, _ in kept],
        "dropped_low": {"dataset": ordered[0][0], "value": ordered[0][1]},
        "dropped_high": {"dataset": ordered[-1][0], "value": ordered[-1][1]},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="experiments/evidence_residual_ablation_10splits.json")
    parser.add_argument(
        "--output",
        default="experiments/reliability_controlled_dual_scale_10splits.json")
    parser.add_argument(
        "--csv", default="results/reliability_controlled_ablation.csv")
    args = parser.parse_args()
    source = json.loads(Path(args.input).read_text(encoding="utf-8"))
    groups = {}
    for row in source["records"]:
        if row["mode"] in MODES:
            groups.setdefault((row["dataset"], row["split"]), {})[
                row["mode"]] = row

    records = []
    for (dataset, split), variants in sorted(groups.items()):
        if set(variants) != set(MODES):
            raise RuntimeError(f"missing scale ablation: {dataset} split {split}")
        hashes = {row["split_hash"] for row in variants.values()}
        checkpoints = {row["warmup_checkpoint"] for row in variants.values()}
        baselines = {row["baseline_test_accuracy"]
                     for row in variants.values()}
        if len(hashes) != 1 or len(checkpoints) != 1 or len(baselines) != 1:
            raise RuntimeError(f"unpaired scale ablation: {dataset} split {split}")
        # Conservative deterministic tie rule: global, then local, then dual.
        selected_mode = max(
            MODES,
            key=lambda mode: (variants[mode]["best_val_accuracy"],
                              -MODES.index(mode)))
        selected = variants[selected_mode]
        records.append({
            "dataset": dataset,
            "split": split,
            "split_hash": selected["split_hash"],
            "warmup_checkpoint": selected["warmup_checkpoint"],
            "selected_mode": selected_mode,
            "validation_accuracy": {
                mode: variants[mode]["best_val_accuracy"] for mode in MODES
            },
            "baseline_test_accuracy": selected["baseline_test_accuracy"],
            "test_accuracy": selected["test_accuracy"],
            "paired_test_gain_pp": selected["test_gain_pp"],
            "best_epoch": selected["best_epoch"],
            "local_coefficient_abs_mean": selected.get(
                "local_coefficient_abs_mean"),
        })

    datasets = sorted({row["dataset"] for row in records})
    summary = {}
    for dataset in datasets:
        rows = [row for row in records if row["dataset"] == dataset]
        gains = [row["paired_test_gain_pp"] for row in rows]
        summary[dataset] = {
            "sfd_test_accuracy": mean_std(
                [100 * row["baseline_test_accuracy"] for row in rows]),
            "selected_test_accuracy": mean_std(
                [100 * row["test_accuracy"] for row in rows]),
            "paired_test_gain_pp": mean_std(gains),
            "wins_ties_losses": [
                sum(value > 1e-8 for value in gains),
                sum(abs(value) <= 1e-8 for value in gains),
                sum(value < -1e-8 for value in gains),
            ],
            "selected_mode_counts": {
                mode: sum(row["selected_mode"] == mode for row in rows)
                for mode in MODES
            },
        }

    dataset_gains = {
        name: value["paired_test_gain_pp"]["mean"]
        for name, value in summary.items()
    }
    result = {
        "protocol": "validation_only_reliability_controlled_dual_scale",
        "selection_rule": (
            "select the highest-validation state among global, local and dual; "
            "ties prefer global, then local, then dual"),
        "dataset_identity_used_for_selection": False,
        "test_labels_used_for_selection": False,
        "source_ablation": args.input,
        "selected_candidate": source["selected_candidate"],
        "candidate": source["candidate"],
        "records": records,
        "summary": summary,
        "macro_mean_paired_test_gain_pp": statistics.mean(
            dataset_gains.values()),
        "trimmed_macro_mean_paired_test_gain_pp": trimmed(dataset_gains),
        "ablation": source["summary"],
    }
    output = Path(args.output)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "dataset", "sfd_mean", "selected_mean", "gain_pp",
            "gain_std_pp", "wins", "ties", "losses",
            "global_count", "local_count", "dual_count",
        ])
        for dataset in datasets:
            item = summary[dataset]
            writer.writerow([
                dataset,
                item["sfd_test_accuracy"]["mean"],
                item["selected_test_accuracy"]["mean"],
                item["paired_test_gain_pp"]["mean"],
                item["paired_test_gain_pp"]["std"],
                *item["wins_ties_losses"],
                *(item["selected_mode_counts"][mode] for mode in MODES),
            ])
        writer.writerow([
            "macro_mean_paired_gain_pp", "", "",
            result["macro_mean_paired_test_gain_pp"],
        ])
        writer.writerow([
            "trimmed_macro_mean_paired_gain_pp", "", "",
            result["trimmed_macro_mean_paired_test_gain_pp"]["mean"],
        ])
    print(json.dumps({
        "summary": summary,
        "macro": result["macro_mean_paired_test_gain_pp"],
        "trimmed_macro": result["trimmed_macro_mean_paired_test_gain_pp"],
    }, indent=2))


if __name__ == "__main__":
    main()
