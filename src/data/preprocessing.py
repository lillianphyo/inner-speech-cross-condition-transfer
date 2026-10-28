"""
EEG preprocessing pipeline: filtering, ICA, epoching, artifact rejection.
"""

import numpy as np
import mne
from typing import Tuple, Optional, List
import pandas as pd
import warnings


class EEGPreprocessor:
    """Preprocess EEG data for speech decoding."""
    
    def __init__(self, 
                 l_freq: float = 1.0,
                 h_freq: float = 40.0,
                 epoch_tmin: float = -0.2,
                 epoch_tmax: float = 1.0,
                 baseline: Tuple[Optional[float], Optional[float]] = (-0.2, 0.0),
                 reject_threshold: float = 100e-6,
                 sfreq: Optional[float] = None,
                 apply_ica: bool = False,
                 ica_n_components: Optional[int] = None,
                 ica_method: str = 'fastica',
                 ica_exclude_eog: bool = True,
                 ica_exclude_ecg: bool = False):
        """
        Initialize preprocessor.
        
        Args:
            l_freq: Low-pass filter frequency (Hz)
            h_freq: High-pass filter frequency (Hz)
            epoch_tmin: Start time of epoch relative to event (seconds)
            epoch_tmax: End time of epoch relative to event (seconds)
            baseline: Baseline correction window (start, end) in seconds
            reject_threshold: Amplitude threshold for artifact rejection (V)
            sfreq: Target sampling frequency for resampling (None = no resampling)
            apply_ica: Whether to apply ICA for artifact removal
            ica_n_components: Number of ICA components (None = 0.99 variance)
            ica_method: ICA method ('fastica', 'infomax', 'picard')
            ica_exclude_eog: Auto-detect and remove EOG artifacts
            ica_exclude_ecg: Auto-detect and remove ECG artifacts
        """
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.epoch_tmin = epoch_tmin
        self.epoch_tmax = epoch_tmax
        self.baseline = baseline
        self.reject_threshold = reject_threshold
        self.target_sfreq = sfreq
        
        # ICA parameters
        self.apply_ica = apply_ica
        self.ica_n_components = ica_n_components
        self.ica_method = ica_method
        self.ica_exclude_eog = ica_exclude_eog
        self.ica_exclude_ecg = ica_exclude_ecg
    
    def filter_raw(self, raw: mne.io.Raw) -> mne.io.Raw:
        """
        Apply bandpass filter to raw EEG data.
        
        Args:
            raw: MNE Raw object (will be filtered in-place for memory efficiency)
            
        Returns:
            Filtered Raw object
        """
        # Filter without loading all data - MNE will handle this efficiently
        # Use n_jobs=1 and no copying to minimize memory usage
        raw.filter(
            self.l_freq, 
            self.h_freq, 
            fir_design='firwin',
            verbose=False,
            n_jobs=1,  # Use single job to avoid memory overhead
            method='fir',  # Explicit method for consistency
            phase='zero'  # Use zero-phase filtering
        )
        
        return raw
    
    def apply_ica_artifact_removal(self, raw: mne.io.Raw) -> mne.io.Raw:
        """
        Apply ICA for artifact removal (EOG, ECG, muscle artifacts).
        
        This is an efficient end-to-end method that:
        1. Fits ICA on the raw data
        2. Automatically detects artifact components
        3. Removes artifact components from the data
        
        Args:
            raw: MNE Raw object (must be filtered and loaded)
            
        Returns:
            Raw object with artifacts removed
        """
        if not self.apply_ica:
            return raw
        
        # Suppress MNE verbose output
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Count only EEG channels (ICA fits only on EEG)
            eeg_picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
            n_eeg_channels = len(eeg_picks)
            
            if n_eeg_channels < 2:
                print(f"  ICA skipped: Only {n_eeg_channels} EEG channels available")
                return raw
            
            # Determine n_components - must be < n_eeg_channels
            n_components = self.ica_n_components
            if n_components is None:
                # Use 0.95 variance equivalent, but cap at n_channels - 1
                # For 64 channels: min(60, 63) = 60
                n_components = min(int(0.95 * n_eeg_channels), n_eeg_channels - 1)
            else:
                # Ensure user-specified n_components doesn't exceed limit
                n_components = min(n_components, n_eeg_channels - 1)
            
            # Safety check
            if n_components < 1:
                print(f"  ICA skipped: n_components={n_components} is too small")
                return raw
            
            print(f"  ICA: Using {n_components} components for {n_eeg_channels} EEG channels")
            
            # Create ICA object
            ica = mne.preprocessing.ICA(
                n_components=n_components,
                method=self.ica_method,
                random_state=42,
                max_iter='auto'
            )
            
            # Fit ICA (requires high-pass filtered data, ideally >= 1 Hz)
            try:
                ica.fit(raw, verbose=False)
            except Exception as e:
                print(f"  ICA fitting failed: {e}. Skipping ICA.")
                return raw
            
            # Find artifact components
            exclude_idx = []
            
            # Auto-detect EOG artifacts
            if self.ica_exclude_eog:
                try:
                    # Try to find EOG channels
                    eog_channels = mne.pick_types(raw.info, eog=True)
                    if len(eog_channels) > 0:
                        eog_indices, eog_scores = ica.find_bads_eog(raw, verbose=False)
                        exclude_idx.extend(eog_indices)
                    else:
                        # Use Fp1/Fp2 channels as proxy for EOG
                        fp_channels = [ch for ch in raw.ch_names if 'Fp' in ch or 'FP' in ch]
                        if fp_channels:
                            eog_indices, eog_scores = ica.find_bads_eog(
                                raw, ch_name=fp_channels[0], verbose=False
                            )
                            exclude_idx.extend(eog_indices)
                except Exception:
                    pass  # No EOG artifacts detected
            
            # Auto-detect ECG artifacts
            if self.ica_exclude_ecg:
                try:
                    ecg_indices, ecg_scores = ica.find_bads_ecg(raw, verbose=False)
                    exclude_idx.extend(ecg_indices)
                except Exception:
                    pass  # No ECG artifacts detected
            
            # Remove duplicate indices
            exclude_idx = list(set(exclude_idx))
            
            if len(exclude_idx) > 0:
                ica.exclude = exclude_idx
                # Apply ICA to remove artifacts (in-place for memory efficiency)
                ica.apply(raw, verbose=False)
                print(f"  ICA removed {len(exclude_idx)} artifact component(s)")
            else:
                print("  ICA: No artifact components detected")
        
        return raw
    
    def resample(self, raw: mne.io.Raw) -> mne.io.Raw:
        """
        Resample raw EEG data.
        
        Args:
            raw: MNE Raw object (will be resampled in-place for memory efficiency)
            
        Returns:
            Resampled Raw object
        """
        if self.target_sfreq is None or raw.info['sfreq'] == self.target_sfreq:
            return raw
        
        raw.resample(self.target_sfreq, verbose=False)
        
        return raw
    
    def create_epochs(self, 
                     raw: mne.io.Raw, 
                     event_times: np.ndarray,
                     event_ids: Optional[np.ndarray] = None) -> mne.Epochs:
        """
        Create epochs from raw data and event times.
        
        Args:
            raw: MNE Raw object (should already be filtered)
            event_times: Array of event onset times (in samples)
            event_ids: Array of event IDs (optional)
            
        Returns:
            MNE Epochs object
        """
        # Create events array for MNE (samples, 0, event_id)
        n_events = len(event_times)
        
        if event_ids is None:
            event_ids = np.ones(n_events, dtype=int)
        
        events = np.column_stack([
            event_times.astype(int),
            np.zeros(n_events, dtype=int),
            event_ids.astype(int)
        ])
        
        # Create event dictionary
        event_dict = {f'event_{i}': i for i in np.unique(event_ids)}
        if len(event_dict) == 1:
            event_dict = {'stimulus': 1}
        
        # Create epochs with baseline and artifact rejection
        # Handle reject threshold - None means no rejection
        reject_dict = {'eeg': self.reject_threshold} if self.reject_threshold is not None else None
        
        epochs = mne.Epochs(
            raw,
            events,
            event_id=event_dict,
            tmin=self.epoch_tmin,
            tmax=self.epoch_tmax,
            baseline=self.baseline,
            preload=True,   # Load only epoch segments, not entire raw
            verbose=False,
            reject=reject_dict
        )
        
        return epochs
    
    def preprocess(self, 
                   raw: mne.io.Raw,
                   event_times: np.ndarray,
                   event_ids: Optional[np.ndarray] = None) -> Tuple[mne.Epochs, np.ndarray]:
        """
        Full preprocessing pipeline: load -> filter -> ICA -> resample -> epoch -> reject artifacts.
        Memory note: loads full raw into memory for filtering (required by MNE).
        Caller should delete raw immediately after this returns.
        
        Args:
            raw: MNE Raw object
            event_times: Array of event onset times (in samples)
            event_ids: Array of event IDs (optional)
            
        Returns:
            Tuple of (epochs, kept_indices) where kept_indices indicates which trials were kept
        """
        # Step 1: Load data into memory (required for filtering)
        # This is memory-intensive but necessary for MNE filtering
        raw.load_data()
        
        # Step 2: Filter raw data in-place
        raw.filter(
            self.l_freq,
            self.h_freq,
            fir_design='firwin',
            verbose=False,
            n_jobs=1
        )
        
        # Step 3: Apply ICA for artifact removal (if enabled)
        if self.apply_ica:
            raw = self.apply_ica_artifact_removal(raw)
        
        # Step 4: Resample raw if needed
        original_sfreq = raw.info['sfreq']
        if self.target_sfreq is not None and original_sfreq != self.target_sfreq:
            raw.resample(self.target_sfreq, verbose=False)
            # Adjust event times for new sampling rate
            ratio = self.target_sfreq / original_sfreq
            event_times = (event_times * ratio).astype(int)
        
        # Step 5: Create epochs (loads trial segments + applies baseline + rejects artifacts)
        epochs = self.create_epochs(raw, event_times, event_ids)
        
        # Get indices of kept epochs (after artifact rejection)
        kept_indices = epochs.selection
        
        # Note: Caller MUST delete raw immediately after this function to free memory
        
        return epochs, kept_indices
    
    def get_data_array(self, epochs: mne.Epochs) -> np.ndarray:
        """
        Get epochs data as numpy array.
        
        Args:
            epochs: MNE Epochs object
            
        Returns:
            Array of shape (n_epochs, n_channels, n_times)
        """
        return epochs.get_data()


def preprocess_subject(raw: mne.io.Raw,
                       trial_info: pd.DataFrame,
                       preprocessor: Optional[EEGPreprocessor] = None) -> Tuple[np.ndarray, pd.DataFrame, mne.Epochs]:
    """
    Preprocess a single subject's data with memory-efficient handling.
    
    Args:
        raw: MNE Raw object
        trial_info: DataFrame with trial information
        preprocessor: EEGPreprocessor instance (creates default if None)
        
    Returns:
        Tuple of (data_array, filtered_trial_info, epochs)
        
    Note:
        Caller MUST delete raw and epochs immediately after this function
        to free memory.
    """
    import gc
    
    if preprocessor is None:
        preprocessor = EEGPreprocessor()
    
    # Get event times from trial info
    event_times = trial_info['P1_TSidx'].values
    
    # Preprocess - this loads data into memory
    epochs, kept_indices = preprocessor.preprocess(raw, event_times)
    
    # Filter trial info to match kept epochs
    filtered_trial_info = trial_info.iloc[kept_indices].reset_index(drop=True)
    
    # Get data array as numpy (not MNE object) to minimize memory
    data_array = preprocessor.get_data_array(epochs)
    
    # Ensure it's a standard numpy array
    data_array = np.asarray(data_array, dtype=np.float32)
    
    # Force copy to avoid keeping reference to epochs
    data_array = data_array.copy()
    
    return data_array, filtered_trial_info, epochs


def create_condition_specific_data(data: np.ndarray,
                                   trial_info: pd.DataFrame,
                                   condition: str = "control") -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Filter data and trial info for specific TMS condition.
    
    Args:
        data: Data array (n_trials, n_channels, n_times)
        trial_info: Trial information DataFrame
        condition: One of 'control', 'tongue', 'lip', or 'all'
        
    Returns:
        Tuple of (filtered_data, filtered_trial_info)
    """
    if condition == "all":
        return data, trial_info
    
    if condition == "control":
        mask = trial_info['TMS'] == 0
    elif condition == "tongue":
        mask = (trial_info['TMS'] == 1) & (trial_info['TMStarget'].str.contains('tongue', case=False, na=False))
    elif condition == "lip":
        mask = (trial_info['TMS'] == 1) & (trial_info['TMStarget'].str.contains('lip', case=False, na=False))
    else:
        raise ValueError(f"Unknown condition: {condition}")
    
    filtered_data = data[mask]
    filtered_trial_info = trial_info[mask].reset_index(drop=True)
    
    return filtered_data, filtered_trial_info


def create_phoneme_labels(trial_info: pd.DataFrame, 
                          task_type: str = "phonemes") -> Tuple[np.ndarray, List[str]]:
    """
    Create phoneme labels for classification.
    
    Args:
        trial_info: Trial information DataFrame
        task_type: 'single-phoneme' or 'phonemes' (CV pairs)
        
    Returns:
        Tuple of (label_array, unique_labels) where label_array is integer labels
    """
    if task_type == "single-phoneme":
        # Use Phoneme1 for single phonemes
        phoneme_strings = trial_info['Phoneme1'].values
    elif task_type == "phonemes":
        # Combine Phoneme1 and Phoneme2 for CV pairs
        phoneme_strings = trial_info['Phoneme1'].astype(str) + trial_info['Phoneme2'].astype(str)
    else:
        raise ValueError(f"Unknown task_type: {task_type}")
    
    # Remove NaN and create unique labels
    phoneme_strings = [str(p) if pd.notna(p) else 'unknown' for p in phoneme_strings]
    unique_labels = sorted(list(set(phoneme_strings)))
    
    # Create label to index mapping
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    
    # Convert to integer labels
    label_array = np.array([label_to_idx[p] for p in phoneme_strings])
    
    return label_array, unique_labels


def create_category_labels(trial_info: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    Create category labels (bilabial vs alveolar).
    
    Args:
        trial_info: Trial information DataFrame
        
    Returns:
        Tuple of (label_array, unique_labels)
    """
    category_strings = trial_info['Category'].values
    category_strings = [str(c) if pd.notna(c) else 'unknown' for c in category_strings]
    unique_labels = sorted(list(set(category_strings)))
    
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    label_array = np.array([label_to_idx[c] for c in category_strings])
    
    return label_array, unique_labels


def create_articulatory_labels(trial_info: pd.DataFrame, 
                               feature_type: str) -> Tuple[np.ndarray, List[str]]:
    """
    Create articulatory feature labels (place, manner, voicing).
    
    Args:
        trial_info: Trial information DataFrame
        feature_type: One of 'place', 'manner', 'voicing'
        
    Returns:
        Tuple of (label_array, unique_labels)
    """
    valid_features = ['place', 'manner', 'voicing']
    if feature_type not in valid_features:
        raise ValueError(f"Unknown feature_type: {feature_type}. Must be one of {valid_features}")
    
    # Map feature_type to column name (capitalize first letter)
    column_name = feature_type.capitalize()
    
    if column_name not in trial_info.columns:
        raise ValueError(f"Column '{column_name}' not found in trial_info")
    
    feature_strings = trial_info[column_name].values
    feature_strings = [str(f) if pd.notna(f) else 'unknown' for f in feature_strings]
    unique_labels = sorted(list(set(feature_strings)))
    
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    label_array = np.array([label_to_idx[f] for f in feature_strings])
    
    return label_array, unique_labels

