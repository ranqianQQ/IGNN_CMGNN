"""Build the blind-test report for validation-selected compatibility heads."""
import argparse
import json
import statistics
from pathlib import Path


def mean_std(values):
    return {"mean": statistics.mean(values), "std": statistics.stdev(values)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--global-results",
        default="experiments/compatibility_finetuning_10splits.json")
    parser.add_argument(
        "--dual-results", default="experiments/dual_scale_10splits.json")
    parser.add_argument(
        "--output",
        default="experiments/reliability_selected_compatibility_10splits.json")
    args = parser.parse_args()
    global_result = json.loads(Path(args.global_results).read_text(encoding="utf-8"))
    dual_result = json.loads(Path(args.dual_results).read_text(encoding="utf-8"))
    global_rows = {(r["dataset"], r["split"]): r
                   for r in global_result["records"]}
    dual_rows = {(r["dataset"], r["split"]): r
                 for r in dual_result["records"]}
    if global_rows.keys() != dual_rows.keys():
        raise RuntimeError("compatibility heads do not cover identical splits")
    records = []
    for key in sorted(global_rows):
        global_row, dual_row = global_rows[key], dual_rows[key]
        if global_row["split_hash"] != dual_row["split_hash"]:
            raise RuntimeError(f"split mismatch: {key}")
        if abs(global_row["baseline_test_accuracy"]
               - dual_row["baseline_test_accuracy"]) > 1e-10:
            raise RuntimeError(f"baseline mismatch: {key}")
        use_dual = (dual_row["best_val_accuracy"]
                    >= global_row["best_val_accuracy"])
        selected = dual_row if use_dual else global_row
        records.append({
            "dataset": key[0],
            "split": key[1],
            "split_hash": selected["split_hash"],
            "selected_head": "dual_scale" if use_dual else "global",
            "global_val_accuracy": global_row["best_val_accuracy"],
            "dual_scale_val_accuracy": dual_row["best_val_accuracy"],
            "baseline_test_accuracy": selected["baseline_test_accuracy"],
            "test_accuracy": selected["test_accuracy"],
            "test_gain_pp": 100 * (
                selected["test_accuracy"]
                - selected["baseline_test_accuracy"]),
            "selected_checkpoint": selected["checkpoint"],
        })
    datasets = sorted({r["dataset"] for r in records})
    summary = {}
    for dataset in datasets:
        rows = [r for r in records if r["dataset"] == dataset]
        gains = [r["test_gain_pp"] for r in rows]
        summary[dataset] = {
            "baseline_test": mean_std(
                [100 * r["baseline_test_accuracy"] for r in rows]),
            "selected_test": mean_std(
                [100 * r["test_accuracy"] for r in rows]),
            "paired_test_gain_pp": mean_std(gains),
            "wins_ties_losses": [sum(g > 1e-8 for g in gains),
                                  sum(abs(g) <= 1e-8 for g in gains),
                                  sum(g < -1e-8 for g in gains)],
            "head_counts": {
                "global": sum(r["selected_head"] == "global" for r in rows),
                "dual_scale": sum(
                    r["selected_head"] == "dual_scale" for r in rows),
            },
        }
    result = {
        "protocol": "one_validation_only_head_selection_rule_all_datasets",
        "selection_rule": (
            "select dual_scale when its validation accuracy is greater than "
            "or equal to the global head; otherwise select global"),
        "dataset_identity_used_for_selection": False,
        "test_labels_used_for_selection": False,
        "global_results": args.global_results,
        "dual_results": args.dual_results,
        "records": records,
        "summary": summary,
        "macro_average_paired_test_gain_pp": statistics.mean(
            item["paired_test_gain_pp"]["mean"]
            for item in summary.values()),
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
