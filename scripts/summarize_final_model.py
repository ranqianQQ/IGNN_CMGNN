"""Summarize fixed SFD+global-compatibility results against official IGNN."""
import argparse
import csv
import json
import statistics
from pathlib import Path


SFD_FILES = (
    "experiments/sfd_three_datasets_10splits.json",
    "experiments/sfd_actor_10splits.json",
    "experiments/sfd_four_more_10splits.json",
)
DATASET_ORDER = (
    "chameleon", "actor", "pubmed", "roman-empire",
    "squirrel", "photo", "amazon-ratings", "wikics",
)


def mean_std(values):
    return statistics.mean(values), statistics.stdev(values)


def load_records(compatibility_path):
    official, sfd = {}, {}
    for file_name in SFD_FILES:
        payload = json.loads(Path(file_name).read_text(encoding="utf-8"))
        for record in payload["records"]:
            key = (record["dataset"], int(record["split"]))
            item = (100.0 * record["test_accuracy"], record["split_hash"])
            if record["model"] == "official_ignn":
                official[key] = item
            elif record["model"] == "tuned_sfd":
                sfd[key] = item

    payload = json.loads(Path(compatibility_path).read_text(encoding="utf-8"))
    final = {
        (record["dataset"], int(record["split"])):
        (100.0 * record["test_accuracy"], record["split_hash"])
        for record in payload["records"] if record["mode"] == "global"
    }
    if not (set(official) == set(sfd) == set(final)):
        raise ValueError("Official IGNN, SFD and final record keys differ")
    for key in official:
        hashes = {official[key][1], sfd[key][1], final[key][1]}
        if len(hashes) != 1:
            raise ValueError(f"Split hash mismatch for {key}")
    return official, sfd, final


def summarize(official, sfd, final):
    rows = []
    for dataset in DATASET_ORDER:
        keys = sorted(key for key in official if key[0] == dataset)
        if len(keys) != 10:
            raise ValueError(f"Expected 10 splits for {dataset}, got {len(keys)}")
        base_values = [official[key][0] for key in keys]
        sfd_values = [sfd[key][0] for key in keys]
        final_values = [final[key][0] for key in keys]
        sfd_gains = [new - base for base, new in zip(base_values, sfd_values)]
        gcc_gains = [new - base for base, new in zip(sfd_values, final_values)]
        total_gains = [new - base for base, new in zip(base_values, final_values)]
        base_mean, base_std = mean_std(base_values)
        sfd_mean, sfd_std = mean_std(sfd_values)
        final_mean, final_std = mean_std(final_values)
        rows.append({
            "dataset": dataset,
            "official_mean": base_mean,
            "official_std": base_std,
            "sfd_mean": sfd_mean,
            "sfd_std": sfd_std,
            "final_mean": final_mean,
            "final_std": final_std,
            "sfd_gain_pp": statistics.mean(sfd_gains),
            "gcc_increment_pp": statistics.mean(gcc_gains),
            "total_gain_pp": statistics.mean(total_gains),
            "total_gain_std_pp": statistics.stdev(total_gains),
            "wins": sum(value > 1e-7 for value in total_gains),
            "ties": sum(abs(value) <= 1e-7 for value in total_gains),
            "losses": sum(value < -1e-7 for value in total_gains),
        })
    gains = [row["total_gain_pp"] for row in rows]
    ordered = sorted(gains)
    return rows, statistics.mean(gains), statistics.mean(ordered[1:-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compatibility",
        default="experiments/evidence_residual_ablation_10splits.json")
    parser.add_argument("--output", default="results/final_summary.csv")
    args = parser.parse_args()

    rows, macro, trimmed = summarize(*load_records(args.compatibility))
    fields = list(rows[0])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        for label, value in (
                ("macro_mean_paired_gain_pp", macro),
                ("trimmed_macro_mean_paired_gain_pp", trimmed)):
            summary = {field: "" for field in fields}
            summary["dataset"] = label
            summary["total_gain_pp"] = value
            writer.writerow(summary)
    print(f"macro gain: {macro:+.6f} pp")
    print(f"trimmed gain: {trimmed:+.6f} pp")


if __name__ == "__main__":
    main()
