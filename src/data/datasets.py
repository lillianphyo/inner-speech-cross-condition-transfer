"""
PyTorch Dataset classes for EEG speech decoding.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import pandas as pd
from pathlib import Path
import gc

from .loaders import EEGDataLoader
from .preprocessing import EEGPreprocessor, preprocess_subject, create_phoneme_labels, create_category_labels, create_articulatory_labels
from .articulatory_features import ArticulatoryFeatureEncoder
from ..utils.paths import resolve_data_root


class EEGSpeechDataset(Dataset):
    """PyTorch Dataset for EEG speech decoding."""
    
    def __init__(self,
                 data: np.ndarray,
                 labels: np.ndarray,
                 trial_info: pd.DataFrame,
                 transform: Optional[callable] = None,
                 n_classes: Optional[int] = None):
        """
        Initialize dataset.
        
        Args:
            data: EEG data array (n_trials, n_channels, n_times)
            labels: Label array (n_trials,)
            trial_info: Trial information DataFrame
            transform: Optional data augmentation transform
            n_classes: Total number of classes (if known). If None, will be inferred from labels.
        """
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        self.trial_info = trial_info
        self.transform = transform
        self._n_classes = n_classes
        
        # Add channel dimension if not present
        if self.data.ndim == 2:
            self.data = self.data.unsqueeze(1)  # (n_trials, 1, n_times)
        elif self.data.ndim == 3:
            # Data is already (n_trials, n_channels, n_times)
            pass
        else:
            raise ValueError(f"Unexpected data shape: {self.data.shape}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx]
        y = self.labels[idx]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y
    
    def get_num_classes(self) -> int:
        """Get number of unique classes."""
        if self._n_classes is not None:
            return self._n_classes
        return len(torch.unique(self.labels))
    
    def get_num_channels(self) -> int:
        """Get number of EEG channels."""
        return self.data.shape[1]
    
    def get_num_timepoints(self) -> int:
        """Get number of time points per trial."""
        return self.data.shape[2]


class SubjectDataset:
    """Dataset for a single subject."""
    
    def __init__(self,
                 subject_id: str,
                 data: np.ndarray,
                 trial_info: pd.DataFrame,
                 task_type: str = "phonemes",
                 label_type: str = "phoneme",
                 feature_type: Optional[str] = None):
        """
        Initialize subject dataset.
        
        Args:
            subject_id: Subject identifier
            data: Preprocessed EEG data
            trial_info: Trial information
            task_type: 'single-phoneme' or 'phonemes'
            label_type: 'phoneme' or 'category'
            feature_type: If set, use articulatory features ('manner', 'place', 'voicing', 'consonant_vowel')
        """
        self.subject_id = subject_id
        self.data = data
        self.trial_info = trial_info
        self.task_type = task_type
        self.label_type = label_type
        self.feature_type = feature_type
        
        # Initialize articulatory feature encoder if needed
        if feature_type is not None:
            self.feature_encoder = ArticulatoryFeatureEncoder(feature_type)
        else:
            self.feature_encoder = None
        
        # Create labels
        if feature_type is not None:
            # Use articulatory features (for backward compatibility with articulatory_features.py)
            phoneme_labels, _ = create_phoneme_labels(trial_info, task_type)
            phoneme_names = [trial_info.iloc[i]['Phoneme1'] if task_type == 'phonemes' else trial_info.iloc[i]['Phoneme'] for i in range(len(trial_info))]
            self.labels = self.feature_encoder.encode_phonemes(phoneme_names)
            self.label_names = self.feature_encoder.get_class_names()
        elif label_type == "phoneme":
            self.labels, self.label_names = create_phoneme_labels(trial_info, task_type)
        elif label_type == "category":
            self.labels, self.label_names = create_category_labels(trial_info)
        elif label_type in ["place", "manner", "voicing"]:
            # Use articulatory features directly from trial_info columns
            self.labels, self.label_names = create_articulatory_labels(trial_info, label_type)
        else:
            raise ValueError(f"Unknown label_type: {label_type}. Must be one of: 'phoneme', 'category', 'place', 'manner', 'voicing'")
    
    def to_pytorch_dataset(self, transform: Optional[callable] = None) -> EEGSpeechDataset:
        """Convert to PyTorch Dataset."""
        n_classes = self.feature_encoder.n_classes if self.feature_encoder is not None else None
        return EEGSpeechDataset(self.data, self.labels, self.trial_info, transform, n_classes)


class StudyDataset:
    """Dataset for entire study (multiple subjects)."""
    
    def __init__(self,
                 data_root: str,
                 study: str = "Study2",
                 preprocessor: Optional[EEGPreprocessor] = None,
                 task_type: str = "phonemes",
                 label_type: str = "phoneme",
                 condition: str = "all",
                 load_on_init: bool = False,
                 max_subjects: Optional[int] = None,
                 feature_type: Optional[str] = None):
        """
        Initialize study dataset.
        
        Args:
            data_root: Root directory containing data (or 'auto' for auto-detection)
            study: 'Study1' or 'Study2'
            preprocessor: EEGPreprocessor instance
            task_type: 'single-phoneme' or 'phonemes'
            label_type: 'phoneme' or 'category'
            condition: 'control', 'tongue', 'lip', or 'all'
            load_on_init: Whether to load all data on initialization
            max_subjects: Maximum number of subjects to load (None = all)
            feature_type: If set, use articulatory features ('manner', 'place', 'voicing', 'consonant_vowel')
        """
        # Resolve data_root path (handles 'auto' keyword)
        self.data_root = resolve_data_root(data_root)
        self.study = study
        self.preprocessor = preprocessor if preprocessor else EEGPreprocessor()
        self.task_type = task_type
        self.label_type = label_type
        self.condition = condition
        self.feature_type = feature_type
        
        # Initialize articulatory feature encoder if needed
        if feature_type is not None:
            self.feature_encoder = ArticulatoryFeatureEncoder(feature_type)
            print(f"Using articulatory features: {feature_type}")
            print(f"  Classes ({self.feature_encoder.n_classes}): {self.feature_encoder.get_class_names()}")
        else:
            self.feature_encoder = None
        
        self.loader = EEGDataLoader(data_root, study)
        self.subject_ids = self.loader.get_subject_ids()
        self.max_subjects = max_subjects
        
        # Limit number of subjects if specified
        if max_subjects is not None and max_subjects > 0:
            self.subject_ids = self.subject_ids[:max_subjects]
            print(f"Limiting to first {max_subjects} subjects: {self.subject_ids}")
        
        self.subject_datasets: Dict[str, SubjectDataset] = {}
        
        if load_on_init:
            self.load_all_subjects()
    
    def load_subject(self, subject_id: str) -> SubjectDataset:
        """Load and preprocess a single subject with memory-efficient handling."""
        if subject_id in self.subject_datasets:
            return self.subject_datasets[subject_id]
        
        # Load raw data - DO NOT PRELOAD to save memory
        raw, trial_info = self.loader.load_subject(subject_id, preload=False)
        
        # Filter by task type
        if self.task_type == "single-phoneme":
            trial_info = trial_info[trial_info['Task'] == 'single-phoneme'].reset_index(drop=True)
        elif self.task_type == "phonemes":
            trial_info = trial_info[trial_info['Task'] == 'phonemes'].reset_index(drop=True)
        
        # Filter by condition
        if self.condition != "all":
            if self.condition == "control":
                trial_info = trial_info[trial_info['TMS'] == 0].reset_index(drop=True)
            elif self.condition == "tongue":
                trial_info = trial_info[
                    (trial_info['TMS'] == 1) & 
                    (trial_info['TMStarget'].str.contains('tongue', case=False, na=False))
                ].reset_index(drop=True)
            elif self.condition == "lip":
                trial_info = trial_info[
                    (trial_info['TMS'] == 1) & 
                    (trial_info['TMStarget'].str.contains('lip', case=False, na=False))
                ].reset_index(drop=True)
        
        # Preprocess - this is memory intensive
        data, trial_info, epochs = preprocess_subject(raw, trial_info, self.preprocessor)
        
        # IMMEDIATELY free raw data and epochs memory
        del raw
        del epochs
        gc.collect()
        
        # Ensure data is a numpy array to avoid keeping MNE objects
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        gc.collect()
        
        # Create subject dataset
        subject_dataset = SubjectDataset(
            subject_id, data, trial_info, self.task_type, self.label_type, self.feature_type
        )
        
        self.subject_datasets[subject_id] = subject_dataset
        return subject_dataset
    
    def load_all_subjects(self):
        """Load all subjects in the study with aggressive memory management."""
        import time
        
        for idx, subject_id in enumerate(self.subject_ids):
            try:
                # Aggressive garbage collection before loading
                gc.collect()
                
                print(f"\n[{idx+1}/{len(self.subject_ids)}] Loading {subject_id}...")
                self.load_subject(subject_id)
                trials_count = len(self.subject_datasets[subject_id].data)
                print(f"✓ Loaded {subject_id}: {trials_count} trials")
                
                # Skip subjects with 0 trials (all epochs rejected)
                if trials_count == 0:
                    print(f"  WARNING: {subject_id} has 0 trials (all epochs rejected). Removing from dataset.")
                    del self.subject_datasets[subject_id]
                
                # AGGRESSIVE memory cleanup after each subject
                gc.collect()
                
                # Add small delay every 3 subjects to let OS reclaim memory
                if (idx + 1) % 3 == 0:
                    print(f"  Pausing for memory cleanup...")
                    time.sleep(2)
                    gc.collect()
                    
            except MemoryError as e:
                print(f"\n{'!'*60}")
                print(f"ERROR: Out of memory loading {subject_id}. Skipping this subject.")
                print(f"{'!'*60}")
                print(f"  Solutions: 1) Close other programs, 2) Restart Python")
                # Aggressive memory cleanup
                if subject_id in self.subject_datasets:
                    del self.subject_datasets[subject_id]
                gc.collect()
                time.sleep(3)
            except Exception as e:
                print(f"Error loading {subject_id}: {str(e)[:200]}")
                # Try to free memory even on error
                if subject_id in self.subject_datasets:
                    del self.subject_datasets[subject_id]
                gc.collect()
    
    def get_combined_dataset(self, transform: Optional[callable] = None) -> EEGSpeechDataset:
        """
        Combine all subjects into a single PyTorch dataset.
        Memory-efficient: extracts data then frees subject objects.
        
        Args:
            transform: Optional data augmentation transform
            
        Returns:
            Combined EEGSpeechDataset
        """
        if not self.subject_datasets:
            self.load_all_subjects()
        
        if not self.subject_datasets:
            raise ValueError("No subjects loaded successfully. Check data paths and file formats.")
        
        all_data = []
        all_labels = []
        all_trial_info = []
        expected_n_channels = None
        
        # Extract data from each subject and free memory as we go
        for subject_id in list(self.subject_datasets.keys()):
            subject_dataset = self.subject_datasets[subject_id]
            
            # Check channel consistency
            n_channels = subject_dataset.data.shape[1]
            if expected_n_channels is None:
                expected_n_channels = n_channels
                print(f"Expected channels: {expected_n_channels}")
            
            if n_channels != expected_n_channels:
                print(f"[WARNING] {subject_id} has {n_channels} channels, expected {expected_n_channels}. Skipping.")
                continue
            
            # Extract arrays (numpy will maintain references, no copy needed)
            all_data.append(subject_dataset.data)
            all_labels.append(subject_dataset.labels)
            
            trial_info = subject_dataset.trial_info.copy()
            trial_info['subject_id'] = subject_id
            all_trial_info.append(trial_info)
            
            # Free this subject's memory immediately
            del self.subject_datasets[subject_id]
            del subject_dataset
            gc.collect()
        
        if not all_data:
            raise ValueError("No subjects with consistent channel counts. Cannot combine datasets.")
        
        combined_data = np.concatenate(all_data, axis=0)
        combined_labels = np.concatenate(all_labels, axis=0)
        combined_trial_info = pd.concat(all_trial_info, ignore_index=True)
        
        # Free the lists of individual arrays
        del all_data, all_labels, all_trial_info
        gc.collect()
        
        # Determine number of classes
        n_classes = self.feature_encoder.n_classes if self.feature_encoder is not None else None
        
        return EEGSpeechDataset(combined_data, combined_labels, combined_trial_info, transform, n_classes)
    
    def get_subject_dataset(self, subject_id: str, transform: Optional[callable] = None) -> EEGSpeechDataset:
        """Get dataset for a specific subject."""
        if subject_id not in self.subject_datasets:
            self.load_subject(subject_id)
        
        return self.subject_datasets[subject_id].to_pytorch_dataset(transform)
    
    def get_leave_one_subject_out_datasets(self, test_subject_id: str, 
                                          transform: Optional[callable] = None) -> Tuple[EEGSpeechDataset, EEGSpeechDataset]:
        """
        Get train/test split for leave-one-subject-out cross-validation.
        
        Args:
            test_subject_id: Subject to hold out for testing
            transform: Optional data augmentation
            
        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        if not self.subject_datasets:
            self.load_all_subjects()
        
        # Collect train subjects
        train_data = []
        train_labels = []
        train_trial_info = []
        expected_n_channels = None
        test_data = None
        test_labels = None
        test_trial_info = None
        
        for subject_id, subject_dataset in self.subject_datasets.items():
            n_channels = subject_dataset.data.shape[1]
            
            if subject_id == test_subject_id:
                test_data = subject_dataset.data
                test_labels = subject_dataset.labels
                test_trial_info = subject_dataset.trial_info.copy()
                test_trial_info['subject_id'] = subject_id
                if expected_n_channels is None:
                    expected_n_channels = n_channels
            else:
                if expected_n_channels is None:
                    expected_n_channels = n_channels
                
                if n_channels != expected_n_channels:
                    print(f"[WARNING] {subject_id} has {n_channels} channels, expected {expected_n_channels}. Skipping from training.")
                    continue
                
                train_data.append(subject_dataset.data)
                train_labels.append(subject_dataset.labels)
                trial_info = subject_dataset.trial_info.copy()
                trial_info['subject_id'] = subject_id
                train_trial_info.append(trial_info)
        
        if not train_data:
            raise ValueError("No training subjects with consistent channel counts.")
        
        train_data = np.concatenate(train_data, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)
        train_trial_info = pd.concat(train_trial_info, ignore_index=True)
        
        train_dataset = EEGSpeechDataset(train_data, train_labels, train_trial_info, transform)
        test_dataset = EEGSpeechDataset(test_data, test_labels, test_trial_info, None)
        
        return train_dataset, test_dataset


def create_dataloaders(train_dataset: Dataset,
                       val_dataset: Optional[Dataset] = None,
                       batch_size: int = 32,
                       num_workers: int = 4,
                       shuffle_train: bool = True) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create PyTorch DataLoaders.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset (optional)
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle_train: Whether to shuffle training data
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return train_loader, val_loader

