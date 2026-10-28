"""
Inner Speech Dataset Loader (ds003626)
======================================
OpenNeuro: https://openneuro.org/datasets/ds003626
Paper: https://www.nature.com/articles/s41597-022-01147-2
GitHub: https://github.com/N-Nieto/Inner_Speech_Dataset

Dataset contains:
- 10 subjects (sub-01 to sub-10)
- 3 sessions per subject
- 128 EEG channels in the derivatives epochs
- 3 conditions: Inner Speech, Pronounced Speech, Visualized
- 4 directions: Up, Down, Left, Right (Spanish: Arriba, Abajo, Derecha, Izquierda)
- Total: 5640 trials

Preprocessed data available in derivatives folder as .fif (MNE Epochs) files,
with class and condition per trial in the companion *_events.dat.

For cross-dataset validation with your conference paper methods.
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

try:
    import mne
except ImportError:
    mne = None


INNER_SPEECH_CLASSES = {
    'arriba': 0,    # Up
    'abajo': 1,     # Down
    'derecha': 2,   # Right
    'izquierda': 3, # Left
}

CONDITION_CODES = {
    'pronounced': 0,
    'inner_speech': 1,
    'visualized': 2,
}
CONDITION_NAMES = {code: name for name, code in CONDITION_CODES.items()}


class InnerSpeechDataset(Dataset):
    """
    PyTorch Dataset for Inner Speech (ds003626), read from the preprocessed
    .fif epochs in derivatives/.

    The .fif event names carry only the word. Class and condition come from the
    companion `*_events.dat` (columns: sample, class, condition, session), which
    is row-aligned with the epochs.

    Args:
        data_root: Path to ds003626 folder
        subjects: List of subjects (e.g., ['sub-01', 'sub-02']) or None for all
        sessions: List of sessions (e.g., ['ses-01']) or None for all
        condition: 'inner_speech', 'pronounced', 'visualized', or 'all'
        max_samples: Maximum samples to load
        target_length: Fixed length for padding/truncating (None = use original)
        tmin, tmax: Crop window in seconds relative to cue onset (None = epoch
            edge, -0.5 / 4.0 s). Concentration -0.5..0, cue 0..0.5,
            action 0.5..3.0, relax 3.0..4.0.
        epochs_suffix: 'eeg' for the 128 scalp channels, 'exg' for the eight
            external electrodes; both epoch files share the events table.

    Example:
        >>> dataset = InnerSpeechDataset(
        ...     data_root="/Volumes/BIWIN/datasets/ds003626",
        ...     condition="inner_speech",
        ...     tmin=0.5, tmax=3.0,
        ... )
    """

    SUBJECTS = [f"sub-{i:02d}" for i in range(1, 11)]  # sub-01 to sub-10
    SESSIONS = ["ses-01", "ses-02", "ses-03"]

    def __init__(
        self,
        data_root: str,
        subjects: Optional[List[str]] = None,
        sessions: Optional[List[str]] = None,
        condition: str = "inner_speech",
        max_samples: Optional[int] = None,
        target_length: Optional[int] = None,
        tmin: Optional[float] = None,
        tmax: Optional[float] = None,
        epochs_suffix: str = "eeg",
    ):
        if mne is None:
            raise ImportError("MNE-Python required: pip install mne")
        if condition != "all" and condition not in CONDITION_CODES:
            raise ValueError(
                f"Unknown condition '{condition}'. "
                f"Choose from {list(CONDITION_CODES)} or 'all'."
            )

        self.data_root = Path(data_root)
        self.subjects = subjects or self.SUBJECTS
        self.sessions = sessions or self.SESSIONS
        self.condition = condition
        self.max_samples = max_samples
        self.target_length = target_length
        self.tmin = tmin
        self.tmax = tmax
        self.epochs_suffix = epochs_suffix

        self.data = []
        self.labels = []
        self.conditions_list = []
        self.subject_ids = []
        self.session_ids = []
        self.ch_names: List[str] = []
        self.sfreq: float = 0.0

        self._load_data()

    def _find_derivatives_files(self) -> List[Tuple[Path, str, str]]:
        """Find preprocessed .fif epochs files in derivatives folder."""
        files = []
        derivatives_path = self.data_root / "derivatives"

        for subject in self.subjects:
            for session in self.sessions:
                session_path = derivatives_path / subject / session
                for fif_file in sorted(session_path.glob(f"*{self.epochs_suffix}-epo.fif")):
                    files.append((fif_file, subject, session))

        return files

    def _load_events(self, fif_file: Path, epochs) -> np.ndarray:
        """Load the events table row-aligned with the epochs of `fif_file`."""
        events_file = fif_file.with_name(fif_file.name.replace(f"{self.epochs_suffix}-epo.fif", "events.dat"))
        if not events_file.exists():
            raise FileNotFoundError(
                f"{events_file} not found. Condition labels live only in this file; "
                "refusing to load epochs without it."
            )
        with open(events_file, "rb") as f:
            events = np.asarray(pickle.load(f))

        if len(events) != len(epochs):
            raise ValueError(
                f"{events_file.name} has {len(events)} rows but {fif_file.name} "
                f"has {len(epochs)} epochs."
            )
        code_to_class = {
            code: next(idx for word, idx in INNER_SPEECH_CLASSES.items() if word in name.lower())
            for name, code in epochs.event_id.items()
        }
        fif_classes = np.array([code_to_class[code] for code in epochs.events[:, 2]])
        if not np.array_equal(fif_classes, events[:, 1]):
            raise ValueError(f"Class labels in {events_file.name} disagree with {fif_file.name}.")
        return events

    def _time_mask(self, times: np.ndarray, sfreq: float) -> np.ndarray:
        tolerance = 0.5 / sfreq
        mask = np.ones(len(times), dtype=bool)
        if self.tmin is not None:
            mask &= times >= self.tmin - tolerance
        if self.tmax is not None:
            mask &= times <= self.tmax + tolerance
        return mask

    def _load_data(self):
        """Load data from the derivatives .fif epochs and their events tables."""
        files = self._find_derivatives_files()
        if not files:
            raise FileNotFoundError(
                f"No derivatives epochs found in {self.data_root}. "
                "Download with: aws s3 sync --no-sign-request s3://openneuro.org/ds003626 ."
            )

        print(f"Loading from {len(files)} preprocessed .fif files")
        print(f"Condition: {self.condition}")

        samples_loaded = 0

        for fif_file, subject, session in files:
            epochs = mne.read_epochs(str(fif_file), preload=True, verbose=False)
            events = self._load_events(fif_file, epochs)
            if not self.ch_names:
                self.ch_names = list(epochs.ch_names)
                self.sfreq = float(epochs.info["sfreq"])
            elif list(epochs.ch_names) != self.ch_names:
                raise ValueError(f"{fif_file.name} has a different channel order from the first file loaded.")

            keep = np.ones(len(events), dtype=bool)
            if self.condition != "all":
                keep = events[:, 2] == CONDITION_CODES[self.condition]
            if self.max_samples:
                keep[np.flatnonzero(keep)[self.max_samples - samples_loaded:]] = False

            time_mask = self._time_mask(epochs.times, epochs.info["sfreq"])
            data = epochs.get_data()[keep][:, :, time_mask].astype(np.float32)
            if self.target_length is not None:
                data = np.stack([self._normalize_length(trial) for trial in data])

            self.data.append(data)
            self.labels.extend(events[keep, 1].astype(int).tolist())
            self.conditions_list.extend(CONDITION_NAMES[code] for code in events[keep, 2])
            self.subject_ids.extend([subject] * len(data))
            self.session_ids.extend([session] * len(data))
            samples_loaded += len(data)

            del epochs

            if self.max_samples and samples_loaded >= self.max_samples:
                break

        self._finalize_data()

    def _finalize_data(self):
        """Finalize loaded data."""
        self.data = np.concatenate(self.data)
        self.labels = np.array(self.labels)

        if len(self.data) == 0:
            raise ValueError(f"No data loaded for condition '{self.condition}'.")

        print(f"Loaded {len(self.data)} samples")
        print(f"  Shape: {self.data.shape}")
        print(f"  Classes: {np.unique(self.labels)}")
        print(f"  Distribution: {self.get_class_distribution()}")

    def _normalize_length(self, data: np.ndarray) -> np.ndarray:
        """Pad or truncate to target length."""
        n_channels, n_times = data.shape

        if n_times < self.target_length:
            padding = np.zeros((n_channels, self.target_length - n_times), dtype=data.dtype)
            data = np.concatenate([data, padding], axis=1)
        elif n_times > self.target_length:
            data = data[:, :self.target_length]

        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        data = torch.FloatTensor(self.data[idx])
        label = int(self.labels[idx])
        return data, label

    def get_num_channels(self) -> int:
        return self.data.shape[1]

    def get_num_timepoints(self) -> int:
        return self.data.shape[2]

    def get_num_classes(self) -> int:
        return len(INNER_SPEECH_CLASSES)

    def get_class_distribution(self) -> Dict[int, int]:
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))


def create_inner_speech_dataloaders(
    data_root: str,
    subjects: Optional[List[str]] = None,
    condition: str = "inner_speech",
    batch_size: int = 32,
    train_ratio: float = 0.8,
    num_workers: int = 0,
    max_samples: Optional[int] = None,
    target_length: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders for Inner Speech dataset.

    Args:
        data_root: Path to dataset
        subjects: List of subjects to include
        condition: Which condition to use
        batch_size: Batch size
        train_ratio: Ratio of data for training
        num_workers: Number of dataloader workers
        max_samples: Maximum samples to load
        target_length: Fixed length for all samples

    Returns:
        Tuple of (train_loader, val_loader)
    """
    dataset = InnerSpeechDataset(
        data_root=data_root,
        subjects=subjects,
        condition=condition,
        max_samples=max_samples,
        target_length=target_length,
    )

    # Split dataset
    n_samples = len(dataset)
    n_train = int(n_samples * train_ratio)
    n_val = n_samples - n_train

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader


def download_inner_speech(output_dir: str = "data/ds003626"):
    """Download Inner Speech dataset from OpenNeuro."""
    print("Downloading Inner Speech Dataset (ds003626)")
    print("This may take a while (~24 GB)")
    print()
    print("Run this command:")
    print(f"  aws s3 sync --no-sign-request s3://openneuro.org/ds003626 {output_dir}")
    print()
    print("Or with OpenNeuro CLI:")
    print(f"  openneuro download --snapshot 2.1.2 ds003626 {output_dir}")
    print()
    print("Or with datalad:")
    print("  datalad install https://github.com/OpenNeuroDatasets/ds003626.git")


def inspect_inner_speech(data_root: str):
    """Inspect the Inner Speech dataset structure."""
    data_root = Path(data_root)

    print("="*60)
    print("Inner Speech Dataset Inspector")
    print("="*60)
    print(f"\nData root: {data_root}")
    print(f"Exists: {data_root.exists()}")

    if not data_root.exists():
        print("\nERROR: Data root does not exist!")
        return

    # Check derivatives
    derivatives_path = data_root / "derivatives"
    if derivatives_path.exists():
        print(f"\nDerivatives found: {derivatives_path}")

        subjects = sorted([d.name for d in derivatives_path.iterdir()
                          if d.is_dir() and d.name.startswith("sub-")])
        print(f"Subjects: {subjects}")

        # Inspect first subject
        if subjects:
            sub_path = derivatives_path / subjects[0]
            sessions = sorted([d.name for d in sub_path.iterdir()
                              if d.is_dir() and d.name.startswith("ses-")])
            print(f"Sessions in {subjects[0]}: {sessions}")

            if sessions:
                ses_path = sub_path / sessions[0]
                files = list(ses_path.iterdir())
                print(f"\nFiles in {subjects[0]}/{sessions[0]}:")
                for f in files:
                    size_mb = f.stat().st_size / (1024 * 1024)
                    print(f"  {f.name} ({size_mb:.1f} MB)")

                # Try loading epochs to check event IDs
                epochs_files = [f for f in files if 'eeg-epo.fif' in f.name]
                if epochs_files and mne is not None:
                    print(f"\nInspecting: {epochs_files[0].name}")
                    try:
                        epochs = mne.read_epochs(str(epochs_files[0]),
                                                preload=False, verbose=False)
                        print(f"  Channels: {len(epochs.ch_names)}")
                        print(f"  Timepoints: {len(epochs.times)}")
                        print(f"  Sampling rate: {epochs.info['sfreq']} Hz")
                        print(f"  Events: {epochs.events.shape[0]}")
                        print("  Event IDs:")
                        for name, code in sorted(epochs.event_id.items()):
                            print(f"    {code}: {name}")
                    except Exception as e:
                        print(f"  Error: {e}")
    else:
        print("\nNo derivatives folder found. Checking for raw data...")

        # Check raw data
        raw_subjects = sorted([d.name for d in data_root.iterdir()
                               if d.is_dir() and d.name.startswith("sub-")])
        if raw_subjects:
            print(f"Raw subjects found: {raw_subjects}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str,
                        default="data/ds003626")
    parser.add_argument("--download", action="store_true",
                        help="Show download instructions")
    parser.add_argument("--inspect", action="store_true",
                        help="Inspect dataset structure")
    parser.add_argument("--test", action="store_true",
                        help="Test loading")
    parser.add_argument("--condition", type=str, default="inner_speech",
                        choices=["inner_speech", "pronounced", "visualized", "all"])
    args = parser.parse_args()

    if args.download:
        download_inner_speech(args.data_root)

    if args.inspect:
        inspect_inner_speech(args.data_root)

    if args.test:
        print("\nTesting Inner Speech Dataset loader...")
        try:
            dataset = InnerSpeechDataset(
                data_root=args.data_root,
                subjects=["sub-01"],
                sessions=["ses-01"],
                condition=args.condition,
                max_samples=100
            )
            print(f"\nSuccess! Loaded {len(dataset)} samples")
            print(f"  Channels: {dataset.get_num_channels()}")
            print(f"  Timepoints: {dataset.get_num_timepoints()}")
            print(f"  Classes: {dataset.get_num_classes()}")
            print(f"  Distribution: {dataset.get_class_distribution()}")

            # Test batch
            loader = DataLoader(dataset, batch_size=16, shuffle=True)
            batch_data, batch_labels = next(iter(loader))
            print(f"\n  Batch shape: {batch_data.shape}")
            print(f"  Labels: {batch_labels.tolist()}")

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            print("\nMake sure to download the dataset first:")
            download_inner_speech(args.data_root)
