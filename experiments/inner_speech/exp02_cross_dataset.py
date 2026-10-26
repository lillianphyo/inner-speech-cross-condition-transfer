"""
Experiment 2: Cross-Dataset Domain Adaptation
==============================================
Cross-dataset transfer learning between different EEG speech datasets.

Transfer scenarios:
1. ds006104 (Overt Speech, 20 classes) -> ds003626 (Inner Speech, 4 classes)
2. ds003626 (Inner Speech) -> ds006104 (Overt Speech)
3. Chisco (Imagined Speech, 39 classes) -> ds003626 (Inner Speech)

Domain adaptation methods:
- Baseline (direct transfer, no adaptation)
- Fine-tuning (train on source, fine-tune on target)
- Feature alignment (MMD, CORAL)

This addresses Gap 1 from the literature review: Cross-subject/cross-dataset generalization.
"""

import copy
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
from pathlib import Path
import gc
import argparse

# Import data loaders
from src.data.inner_speech_loader import InnerSpeechDataset
from src.data.leakage import assert_disjoint
from src.data.splits import split_indices

# Import models
from src.models.eegnet import EEGNet
from src.models.deepconvnet import DeepConvNet, ShallowConvNet


# Configuration
DEFAULT_CONFIG = {
    'inner_speech_root': '/media/lillian/BIWIN/datasets/ds003626',
    'batch_size': 32,
    'num_epochs': 100,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'patience': 15,
    'finetune_epochs': 50,
    'finetune_lr': 1e-4,
    'output_dir': 'results/cross_dataset',
}


def get_model(model_name: str, n_channels: int, n_times: int, n_classes: int):
    """Factory function to create models."""
    if model_name == "EEGNet":
        return EEGNet(n_channels, n_times, n_classes,
                      n_temporal_filters=8, depth_multiplier=2,
                      kernel_length=64, dropout_rate=0.5)
    elif model_name == "DeepConvNet":
        return DeepConvNet(n_channels, n_times, n_classes, dropout_rate=0.5)
    elif model_name == "ShallowConvNet":
        return ShallowConvNet(n_channels, n_times, n_classes, dropout_rate=0.5)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """Compute classification metrics."""
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'kappa': cohen_kappa_score(y_true, y_pred),
    }
    return metrics


def train_model(model, train_loader, val_loader, device, config, n_classes, regularizer=None):
    """Train model and return best model state; `regularizer(model)` is added to the loss after each forward pass."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                  lr=config['learning_rate'],
                                  weight_decay=config['weight_decay'])

    best_val_acc = 0.0
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(config['num_epochs']):
        # Training
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            if regularizer is not None:
                loss = loss + regularizer(model)
            loss.backward()
            optimizer.step()

        # Validation
        if val_loader is not None:
            model.eval()
            val_preds, val_labels = [], []
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    val_preds.extend(output.argmax(dim=1).cpu().numpy())
                    val_labels.extend(target.cpu().numpy())

            val_acc = np.mean(np.array(val_preds) == np.array(val_labels))

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= config['patience']:
                break

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}: val_acc={val_acc:.4f}" if val_loader else f"    Epoch {epoch+1}")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def evaluate_model(model, test_loader, device):
    """Evaluate model on test set."""
    model.eval()
    test_preds, test_labels = [], []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_preds.extend(output.argmax(dim=1).cpu().numpy())
            test_labels.extend(target.cpu().numpy())

    return compute_metrics(np.array(test_labels), np.array(test_preds))


def finetune_model(model, train_loader, val_loader, device, config):
    """Fine-tune a pre-trained model on target domain."""
    # Freeze feature extractor, only train classifier
    for name, param in model.named_parameters():
        if 'classifier' not in name and 'fc' not in name:
            param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=config['finetune_lr'],
                                  weight_decay=config['weight_decay'])

    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(config['finetune_epochs']):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # Validation
        if val_loader is not None:
            model.eval()
            val_preds, val_labels = [], []
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    val_preds.extend(output.argmax(dim=1).cpu().numpy())
                    val_labels.extend(target.cpu().numpy())

            val_acc = np.mean(np.array(val_preds) == np.array(val_labels))

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = copy.deepcopy(model.state_dict())

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def run_cross_condition_transfer(config: dict, device: torch.device):
    """
    Transfer between conditions within Inner Speech dataset.

    Scenarios:
    - Pronounced -> Inner Speech (overt to imagined)
    - Visualized -> Inner Speech
    - All conditions -> Inner Speech
    """
    print("\n" + "="*80)
    print("CROSS-CONDITION TRANSFER (Within ds003626)")
    print("="*80)

    results = []

    # Source conditions
    source_conditions = ['pronounced', 'visualized']
    target_condition = 'inner_speech'

    for source_condition in source_conditions:
        print(f"\n{'-'*60}")
        print(f"Transfer: {source_condition} -> {target_condition}")
        print(f"{'-'*60}")

        # Load source and target datasets
        try:
            source_dataset = InnerSpeechDataset(
                data_root=config['inner_speech_root'],
                condition=source_condition,
            )
            target_dataset = InnerSpeechDataset(
                data_root=config['inner_speech_root'],
                condition=target_condition,
            )
        except Exception as e:
            print(f"Error loading datasets: {e}")
            continue

        assert_disjoint(source_dataset.data, target_dataset.data)

        print(f"Source ({source_condition}): {len(source_dataset)} samples")
        print(f"Target ({target_condition}): {len(target_dataset)} samples")

        n_channels = source_dataset.get_num_channels()
        n_times = source_dataset.get_num_timepoints()
        n_classes = source_dataset.get_num_classes()

        train_idx, val_idx, test_idx = split_indices(target_dataset.labels, (0.6, 0.2, 0.2), np.random.default_rng(42))

        target_train = torch.utils.data.Subset(target_dataset, train_idx)
        target_val = torch.utils.data.Subset(target_dataset, val_idx)
        target_test = torch.utils.data.Subset(target_dataset, test_idx)

        # Data loaders
        source_loader = DataLoader(source_dataset, batch_size=config['batch_size'],
                                   shuffle=True, num_workers=0)
        target_train_loader = DataLoader(target_train, batch_size=config['batch_size'],
                                         shuffle=True, num_workers=0)
        target_val_loader = DataLoader(target_val, batch_size=config['batch_size'],
                                       shuffle=False, num_workers=0)
        target_test_loader = DataLoader(target_test, batch_size=config['batch_size'],
                                        shuffle=False, num_workers=0)

        for model_name in ['EEGNet', 'ShallowConvNet']:
            print(f"\n  Model: {model_name}")

            # 1. Direct transfer (no adaptation)
            print("    Training on source...")
            model = get_model(model_name, n_channels, n_times, n_classes)
            model = train_model(model, source_loader, None, device, config, n_classes)

            print("    Evaluating direct transfer...")
            metrics_direct = evaluate_model(model, target_test_loader, device)
            print(f"    Direct: acc={metrics_direct['accuracy']:.4f}")

            results.append({
                'source': source_condition,
                'target': target_condition,
                'model': model_name,
                'method': 'direct_transfer',
                **metrics_direct
            })

            # 2. Fine-tuning
            print("    Fine-tuning on target...")
            model = finetune_model(model, target_train_loader, target_val_loader, device, config)

            metrics_finetune = evaluate_model(model, target_test_loader, device)
            print(f"    Fine-tune: acc={metrics_finetune['accuracy']:.4f}")

            results.append({
                'source': source_condition,
                'target': target_condition,
                'model': model_name,
                'method': 'fine_tuning',
                **metrics_finetune
            })

            # 3. Target-only baseline
            print("    Training target-only baseline...")
            model = get_model(model_name, n_channels, n_times, n_classes)
            model = train_model(model, target_train_loader, target_val_loader, device, config, n_classes)

            metrics_target = evaluate_model(model, target_test_loader, device)
            print(f"    Target-only: acc={metrics_target['accuracy']:.4f}")

            results.append({
                'source': 'none',
                'target': target_condition,
                'model': model_name,
                'method': 'target_only',
                **metrics_target
            })

            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return results


def main(args):
    """Main experiment function."""
    config = DEFAULT_CONFIG.copy()
    config['inner_speech_root'] = args.data_root

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Run cross-condition transfer
    results = run_cross_condition_transfer(config, device)

    # Save results
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    results_df = pd.DataFrame(results)
    results_path = output_dir / f'cross_condition_results_{timestamp}.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")

    # Summary
    print("\n" + "="*80)
    print("CROSS-CONDITION TRANSFER SUMMARY")
    print("="*80)
    print(results_df.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cross-Dataset Domain Adaptation')
    parser.add_argument('--data-root', type=str,
                        default='/media/lillian/BIWIN/datasets/ds003626',
                        help='Path to ds003626 dataset')

    args = parser.parse_args()
    main(args)
