"""Ten-split evaluation of the globally selected CM-guided fine-tuner."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch

from ignn.models import IGNN
from ignn.modules.compatibility_propagation import _row_normalized_adjacency
from scripts.compatibility_experiment_common import seed_all
from scripts.screen_compatibility_finetuning import (
    DATASETS, CANDIDATES, accuracy, fine_tune, masks_for, model_setup)


SELECTED = 'learnable_cm'


def mean_std(values):
    return {'mean': statistics.mean(values), 'std': statistics.stdev(values)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=list(DATASETS))
    parser.add_argument('--max-epochs', type=int, default=200)
    parser.add_argument(
        '--output', default='experiments/compatibility_finetuning_10splits.json')
    args = parser.parse_args()
    device = torch.device('cuda:0')
    torch.set_num_threads(4)
    output = Path(args.output)
    result = {
        'protocol': 'fixed_global_configuration_official_10_splits',
        'selection': 'learnable_cm selected by mean validation gain on splits 0-2; test unseen',
        'selected_candidate': SELECTED,
        'candidate': CANDIDATES[SELECTED],
        'cmgnn_source_commit': '580dcb42542ba52119187501beeb7b5879cd05b0',
        'max_fine_tuning_epochs': args.max_epochs,
        'records': [],
    }
    candidate = CANDIDATES[SELECTED]
    for name in args.datasets:
        _, params, rn, data, config = model_setup(name, device)
        adjacency, _ = _row_normalized_adjacency(
            data.edge_index, data.num_nodes, device)
        for split in range(10):
            masks = masks_for(data, split, device)
            split_hash = hashlib.sha256(
                b''.join(m.cpu().numpy().tobytes() for m in masks)).hexdigest()
            seed = 42 + split
            seed_all(seed)
            model = IGNN(
                data.num_features, n_clusters=int(data.y.max()) + 1,
                IN='IN-SN', RN=rn, agg_type='gcn_incep', **params).to(device)
            warmup_checkpoint = Path(
                f'experiments/sfd_10split_{name}_{split}_tuned_sfd.pt')
            model.load_state_dict(torch.load(warmup_checkpoint, map_location=device))
            model.eval()
            with torch.no_grad():
                embeddings = model(
                    data.edge_index, data.x, config, device).detach()
                baseline_logits = model.classifier(embeddings)
                baseline_val = accuracy(baseline_logits, data.y, masks[1])
                baseline_test = accuracy(baseline_logits, data.y, masks[2])
            print(
                f'dataset={name} split={split} baseline_val={baseline_val:.4f} '
                f'baseline_test={baseline_test:.4f}', flush=True)
            adapter, fit = fine_tune(
                model, embeddings, adjacency, data, masks, candidate, params,
                seed, max_epochs=args.max_epochs)
            model.classifier.eval(); adapter.eval()
            with torch.no_grad():
                raw_logits = model.classifier(embeddings)
                corrected = adapter(
                    raw_logits, adjacency, data.y, masks[0], steps=1)
                test = accuracy(corrected, data.y, masks[2])
                val = accuracy(corrected, data.y, masks[1])
                compatibility = adapter.compatibility()
                row_sum_error = float(
                    (compatibility.sum(1) - 1).abs().max().item())
            checkpoint = output.parent / (
                f'cmft_10split_{name}_{split}_{SELECTED}.pt')
            torch.save({
                'classifier': model.classifier.state_dict(),
                'adapter': adapter.state_dict(),
                'warmup_checkpoint': warmup_checkpoint.name,
            }, checkpoint)
            adapter_parameters = sum(p.numel() for p in adapter.parameters())
            record = {
                'dataset': name,
                'split': split,
                'seed': seed,
                'split_hash': split_hash,
                'warmup_checkpoint': warmup_checkpoint.name,
                'checkpoint': checkpoint.name,
                'baseline_val_accuracy': baseline_val,
                'best_val_accuracy': val,
                'val_gain_pp': 100 * (val - baseline_val),
                'baseline_test_accuracy': baseline_test,
                'test_accuracy': test,
                'test_gain_pp': 100 * (test - baseline_test),
                'backbone_parameters': sum(p.numel() for p in model.parameters()),
                'adapter_parameters': adapter_parameters,
                'fine_tuned_parameters': (
                    sum(p.numel() for p in model.classifier.parameters())
                    + sum(p.numel() for p in adapter.parameters() if p.requires_grad)),
                'compatibility_row_sum_max_error': row_sum_error,
                **fit,
            }
            result['records'].append(record)
            output.write_text(json.dumps(result, indent=2), encoding='utf-8')
            print({k: v for k, v in record.items()
                   if k not in ('val_accuracy_history', 'initial_cm_diagnostics')}, flush=True)
            del model, adapter, embeddings
    summary = {}
    for name in args.datasets:
        rows = sorted(
            (r for r in result['records'] if r['dataset'] == name),
            key=lambda r: r['split'])
        summary[name] = {
            'baseline_test': mean_std(
                [100 * r['baseline_test_accuracy'] for r in rows]),
            'cmft_test': mean_std([100 * r['test_accuracy'] for r in rows]),
            'paired_test_gain_pp': mean_std(
                [r['test_gain_pp'] for r in rows]),
            'wins_ties_losses': [
                sum(r['test_gain_pp'] > 1e-8 for r in rows),
                sum(abs(r['test_gain_pp']) <= 1e-8 for r in rows),
                sum(r['test_gain_pp'] < -1e-8 for r in rows),
            ],
            'mean_val_gain_pp': statistics.mean(r['val_gain_pp'] for r in rows),
            'mean_fine_tuning_seconds': statistics.mean(
                r['fine_tuning_seconds'] for r in rows),
            'mean_strength': statistics.mean(r['strength'] for r in rows),
        }
    result['summary'] = summary
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
