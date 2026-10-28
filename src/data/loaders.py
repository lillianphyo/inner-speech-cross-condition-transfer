"""
EEG data loaders for Study1 and Study2 datasets.
Loads .cnt/.evt files and trial information CSVs.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import mne
from ..utils.paths import resolve_data_root


class EEGDataLoader:
    """Load EEG data from .cnt/.evt files with trial information."""
    
    def __init__(self, data_root: str, study: str = "Study2"):
        """
        Initialize EEG data loader.
        
        Args:
            data_root: Root directory containing data folder (or 'auto' for auto-detection)
            study: Either "Study1" or "Study2"
        """
        # Resolve data_root path (handles 'auto' keyword)
        data_root = resolve_data_root(data_root)
        self.data_root = Path(data_root)
        self.study = study
        self.study_path = self.data_root / "data" / study / "Data_Descriptors"
        
        # Study1 and Study2 have different EEG path structures
        if study == "Study1":
            self.eeg_path = self.study_path / "EEG"
        else:  # Study2
            self.eeg_path = self.study_path / "Neural_decoding" / "EEG"
        
        self.trial_info_path = self.study_path / "TrialInfo"
        
        if not self.study_path.exists():
            raise ValueError(f"Study path does not exist: {self.study_path}")
    
    def get_subject_ids(self) -> List[str]:
        """Get list of available subject IDs."""
        if not self.eeg_path.exists():
            return []
        
        # Study1 uses P*.cnt, Study2 uses S*.cnt
        pattern = "P*.cnt" if self.study == "Study1" else "S*.cnt"
        cnt_files = list(self.eeg_path.glob(pattern))
        subject_ids = [f.stem for f in cnt_files]
        subject_ids.sort()
        return subject_ids
    
    def load_raw_eeg(self, subject_id: str, preload: bool = False) -> mne.io.Raw:
        """
        Load raw EEG data for a subject.
        
        Args:
            subject_id: Subject identifier (e.g., 'S01')
            preload: Whether to preload data into memory
            
        Returns:
            MNE Raw object
        """
        cnt_file = self.eeg_path / f"{subject_id}.cnt"
        
        if not cnt_file.exists():
            raise FileNotFoundError(f"EEG file not found: {cnt_file}")
        
        try:
            raw = mne.io.read_raw_ant(cnt_file, preload=preload)
            return raw
        except Exception as e_ant:
            try:
                raw = mne.io.read_raw_cnt(cnt_file, preload=preload)
                return raw
            except Exception as e_cnt:
                raise RuntimeError(f"Failed to load {cnt_file}. ANT error: {str(e_ant)[:100]}, CNT error: {str(e_cnt)[:100]}")
    
    def load_trial_info(self, subject_id: str) -> pd.DataFrame:
        """
        Load trial information CSV for a subject.
        
        Args:
            subject_id: Subject identifier (e.g., 'S01')
            
        Returns:
            DataFrame with trial information
        """
        csv_file = self.trial_info_path / f"{subject_id}_Tab.csv"
        
        if not csv_file.exists():
            raise FileNotFoundError(f"Trial info file not found: {csv_file}")
        
        df = pd.read_csv(csv_file)
        
        return df
    
    def load_subject(self, subject_id: str, preload: bool = False, load_eeg: bool = True) -> Tuple[Optional[mne.io.Raw], pd.DataFrame]:
        """
        Load both EEG data and trial information for a subject.
        
        Args:
            subject_id: Subject identifier (e.g., 'S01')
            preload: Whether to preload EEG data
            load_eeg: Whether to load EEG data (set False for metadata-only)
            
        Returns:
            Tuple of (raw EEG, trial DataFrame). raw EEG is None if load_eeg=False
        """
        raw = self.load_raw_eeg(subject_id, preload=preload) if load_eeg else None
        trial_info = self.load_trial_info(subject_id)
        
        return raw, trial_info
    
    def get_phoneme_labels(self, trial_info: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Extract phoneme labels from trial information.
        
        Args:
            trial_info: Trial information DataFrame
            
        Returns:
            Dictionary with phoneme information
        """
        labels = {
            'phoneme1': trial_info['Phoneme1'].values,
            'phoneme2': trial_info['Phoneme2'].values,
            'place': trial_info['Place'].values,
            'manner': trial_info['Manner'].values,
            'voicing': trial_info['Voicing'].values,
            'category': trial_info['Category'].values,
            'tms_condition': trial_info['TMS'].values,
            'tms_target': trial_info['TMStarget'].values,
            'task': trial_info['Task'].values,
        }
        
        return labels
    
    def get_event_times(self, trial_info: pd.DataFrame) -> np.ndarray:
        """
        Extract event onset times from trial information.
        
        Args:
            trial_info: Trial information DataFrame
            
        Returns:
            Array of event onset times (in samples)
        """
        # Use P1_TSidx as the primary stimulus onset
        event_times = trial_info['P1_TSidx'].values
        
        return event_times
    
    def filter_trials_by_condition(self, trial_info: pd.DataFrame, 
                                   condition: str = "control") -> pd.DataFrame:
        """
        Filter trials by TMS condition.
        
        Args:
            trial_info: Trial information DataFrame
            condition: One of 'control', 'tongue', 'lip', or 'all'
            
        Returns:
            Filtered DataFrame
        """
        if condition == "all":
            return trial_info
        
        condition_map = {
            "control": 0,
            "tongue": 1,  # TMS on tongue motor cortex
            "lip": 1,     # TMS on lip motor cortex (distinguished by TMStarget)
        }
        
        if condition in ["control"]:
            return trial_info[trial_info['TMS'] == condition_map[condition]]
        elif condition == "tongue":
            return trial_info[
                (trial_info['TMS'] == 1) & 
                (trial_info['TMStarget'].str.contains('tongue', case=False, na=False))
            ]
        elif condition == "lip":
            return trial_info[
                (trial_info['TMS'] == 1) & 
                (trial_info['TMStarget'].str.contains('lip', case=False, na=False))
            ]
        else:
            raise ValueError(f"Unknown condition: {condition}")
    
    def filter_single_phoneme_trials(self, trial_info: pd.DataFrame) -> pd.DataFrame:
        """
        Filter for single phoneme trials only.
        
        Args:
            trial_info: Trial information DataFrame
            
        Returns:
            Filtered DataFrame with only single-phoneme trials
        """
        return trial_info[trial_info['Task'] == 'single-phoneme']
    
    def filter_phoneme_pair_trials(self, trial_info: pd.DataFrame) -> pd.DataFrame:
        """
        Filter for phoneme pair (CV) trials.
        
        Args:
            trial_info: Trial information DataFrame
            
        Returns:
            Filtered DataFrame with only phoneme pair trials
        """
        return trial_info[trial_info['Task'] == 'phonemes']


def load_all_subjects(data_root: str, study: str = "Study2", 
                     preload: bool = True) -> Dict[str, Tuple[mne.io.Raw, pd.DataFrame]]:
    """
    Load all subjects from a study.
    
    Args:
        data_root: Root directory containing data folder
        study: Either "Study1" or "Study2"
        preload: Whether to preload EEG data
        
    Returns:
        Dictionary mapping subject_id to (raw, trial_info) tuples
    """
    loader = EEGDataLoader(data_root, study=study)
    subject_ids = loader.get_subject_ids()
    
    data = {}
    for subject_id in subject_ids:
        try:
            raw, trial_info = loader.load_subject(subject_id, preload=preload)
            data[subject_id] = (raw, trial_info)
            print(f"Loaded {subject_id}: {len(trial_info)} trials")
        except Exception as e:
            print(f"Error loading {subject_id}: {e}")
    
    return data

