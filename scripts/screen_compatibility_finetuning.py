"""Validation-only screen for compatibility-guided IGNN fine-tuning."""
import argparse
import copy
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch

from ignn.configs import DataConf, INConf
from ignn.models import IGNN
from ignn.modules.CompatibilityGuidedFineTuning import CompatibilityGuidedFineTuning
from ignn.modules.compatibility_propagation import (
    _row_normalized_adjacency, estimate_cmgnn_compatibility)
from scripts.compatibility_experiment_common import (
    DATASETS, load_dataset, seed_all, setting_for)
from utils import get_splits


CANDIDATES = {
    'dynamic_cm': dict(learnable_matrix=False, update_cm=True),
    'learnable_cm': dict(learnable_matrix=True, update_cm=False),
    'dynamic_learnable_cm': dict(learnable_matrix=True, update_cm=True),
}
def model_setup(name, device):
    setting = setting_for(name)
    params = dict(setting)
    source = params.pop('source')
    row_norm = params.pop('row_norm', False)
    rn = params.pop('candidate_rn')
    params.pop('gate_lr_multiplier', None)
    data = load_dataset(name, source, row_norm).to(device)
    config = INConf(
        data.name, params['n_hops'], True, False, True, row_norm, params['fast'])
    return setting, params, rn, data, config


def masks_for(data, split, device):
    return tuple(m.to(device) for m in get_splits(
        data, data.name, data.num_nodes, split, 10, 48, 32,
        DataConf('data', 'data/random_splits'), public=False))


def accuracy(logits, labels, mask):
    return (logits[mask].argmax(-1) == labels[mask]).float().mean().item()


def fine_tune(model, embeddings, adjacency, data, masks, candidate, params,
              seed, max_epochs=200, patience=50):
    seed_all(seed)
    model.classifier.eval()
    with torch.no_grad():
        initial_logits = model.classifier(embeddings)
        initial_cm, initial_diagnostics = estimate_cmgnn_compatibility(
            initial_logits, data.edge_index, data.y, masks[0])
    adapter = CompatibilityGuidedFineTuning(
        initial_cm, initial_strength=0.1,
        learnable_matrix=candidate['learnable_matrix']).to(embeddings.device)
    trainable = list(model.classifier.parameters()) + [
        p for p in adapter.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        [
            {'params': model.classifier.parameters(), 'lr': params['lr'] * 0.1},
            {'params': [p for p in adapter.parameters() if p.requires_grad], 'lr': 0.01},
        ], weight_decay=params['l2_coef'])
    del trainable
    best_val, best_epoch, best_state = -1.0, -1, None
    history = []
    started = time.perf_counter()
    for epoch in range(max_epochs):
        model.classifier.train(); adapter.train(); optimizer.zero_grad()
        raw_logits = model.classifier(embeddings)
        corrected = adapter(raw_logits, adjacency, data.y, masks[0], steps=1)
        loss = (
            model.criterion(corrected[masks[0]], data.y[masks[0]])
            + 0.25 * model.criterion(raw_logits[masks[0]], data.y[masks[0]])
            + 1e-3 * adapter.regularization())
        if not torch.isfinite(loss):
            raise RuntimeError('non-finite fine-tuning loss')
        loss.backward(); optimizer.step()

        model.classifier.eval(); adapter.eval()
        with torch.no_grad():
            raw_logits = model.classifier(embeddings)
            corrected = adapter(raw_logits, adjacency, data.y, masks[0], steps=1)
            val = accuracy(corrected, data.y, masks[1])
        history.append(val)
        if val >= best_val:
            best_val, best_epoch = val, epoch
            best_state = {
                'classifier': copy.deepcopy(model.classifier.state_dict()),
                'adapter': copy.deepcopy(adapter.state_dict()),
            }
        if candidate['update_cm'] and (epoch + 1) % 10 == 0:
            with torch.no_grad():
                estimate, _ = estimate_cmgnn_compatibility(
                    corrected, data.edge_index, data.y, masks[0])
                adapter.update_compatibility(estimate, momentum=0.8)
        if epoch - best_epoch >= patience:
            break
    elapsed = time.perf_counter() - started
    model.classifier.load_state_dict(best_state['classifier'])
    adapter.load_state_dict(best_state['adapter'])
    return adapter, {
        'best_epoch': best_epoch + 1,
        'best_val_accuracy': best_val,
        'epochs': epoch + 1,
        'fine_tuning_seconds': elapsed,
        'strength': float(adapter.strength().detach().item()),
        'matrix_shift_l2': float(adapter.matrix_residual.detach().norm().item()),
        'initial_cm_diagnostics': initial_diagnostics,
        'val_accuracy_history': history,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split-end', type=int, default=3)
    parser.add_argument('--max-epochs', type=int, default=200)
    parser.add_argument('--datasets', nargs='+', default=list(DATASETS))
    parser.add_argument('--output', default='experiments/compatibility_finetuning_screen.json')
    args = parser.parse_args()
    device = torch.device('cuda:0')
    torch.set_num_threads(4)
    output = Path(args.output)
    result = {
        'protocol': 'shared_validation_only_compatibility_finetuning_screen',
        'test_labels_evaluated': False,
        'cmgnn_source_commit': '580dcb42542ba52119187501beeb7b5879cd05b0',
        'candidates': CANDIDATES,
        'datasets': args.datasets,
        'split_range': [0, args.split_end],
        'max_epochs': args.max_epochs,
        'records': [],
    }
    for name in args.datasets:
        _, params, rn, data, config = model_setup(name, device)
        adjacency, _ = _row_normalized_adjacency(
            data.edge_index, data.num_nodes, device)
        for split in range(args.split_end):
            masks = masks_for(data, split, device)
            split_hash = hashlib.sha256(
                b''.join(m.cpu().numpy().tobytes() for m in masks)).hexdigest()
            seed = 42 + split
            for candidate_name, candidate in CANDIDATES.items():
                seed_all(seed)
                model = IGNN(
                    data.num_features, n_clusters=int(data.y.max()) + 1,
                    IN='IN-SN', RN=rn, agg_type='gcn_incep', **params).to(device)
                checkpoint = f'experiments/sfd_10split_{name}_{split}_tuned_sfd.pt'
                model.load_state_dict(torch.load(checkpoint, map_location=device))
                model.eval()
                with torch.no_grad():
                    embeddings = model(data.edge_index, data.x, config, device).detach()
                    baseline_logits = model.classifier(embeddings)
                    baseline_val = accuracy(baseline_logits, data.y, masks[1])
                print(
                    f'dataset={name} split={split} candidate={candidate_name} '
                    f'baseline_val={baseline_val:.4f}', flush=True)
                adapter, fit = fine_tune(
                    model, embeddings, adjacency, data, masks, candidate, params,
                    seed, max_epochs=args.max_epochs)
                saved = output.parent / (
                    f'{output.stem}_{name}_{split}_{candidate_name}.pt')
                torch.save({
                    'classifier': model.classifier.state_dict(),
                    'adapter': adapter.state_dict(),
                }, saved)
                record = {
                    'dataset': name,
                    'split': split,
                    'seed': seed,
                    'split_hash': split_hash,
                    'candidate': candidate_name,
                    'baseline_val_accuracy': baseline_val,
                    'val_gain_pp': 100 * (fit['best_val_accuracy'] - baseline_val),
                    'checkpoint': saved.name,
                    **fit,
                }
                result['records'].append(record)
                output.write_text(json.dumps(result, indent=2), encoding='utf-8')
                print({k: v for k, v in record.items()
                       if k not in ('val_accuracy_history', 'initial_cm_diagnostics')}, flush=True)
                del model, adapter, embeddings
    ranking = []
    for candidate_name in CANDIDATES:
        rows = [r for r in result['records'] if r['candidate'] == candidate_name]
        ranking.append({
            'candidate': candidate_name,
            'mean_val_gain_pp': statistics.mean(r['val_gain_pp'] for r in rows),
            'positive_splits': sum(r['val_gain_pp'] > 1e-8 for r in rows),
            'total_splits': len(rows),
        })
    ranking.sort(key=lambda row: row['mean_val_gain_pp'], reverse=True)
    result['ranking'] = ranking
    result['selected_global_candidate'] = ranking[0]['candidate']
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(ranking, indent=2), flush=True)


if __name__ == '__main__':
    main()
