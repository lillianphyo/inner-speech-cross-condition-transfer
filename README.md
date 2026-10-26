# Cross-condition transfer learning for EEG-based inner speech classification

Code, per-participant results and figure scripts for the manuscript *Cross-Condition Transfer Learning for EEG-Based Inner Speech Classification: A Controlled Evaluation of Neural Networks and Temporal Segmentation* (K. P. P. San, Y. Sugiura, T. Shimamura; submitted to *Sensors*).

The study asks whether a decoder trained on overt (pronounced) speech transfers to inner speech on OpenNeuro ds003626, under an evaluation in which no test trial enters training or model selection. It compares bidirectional transfer, target-domain calibration and fine-tuning, MMD alignment, two convolutional networks against CSP+LDA and a Riemannian pipeline, and four trial windows, each with shuffled-label controls.
Every number in the paper comes from a per-participant CSV under `runs/`, made by one of the commands below. The `--summarize` form of each script rebuilds the summary table from those CSVs.

## 1. Data

| Dataset | Version | Source | Used |
|---|---|---|---|
| Thinking out loud (Nieto et al. 2022) | ds003626 **v2.1.2** | <https://doi.org/10.18112/openneuro.ds003626.v2.1.2> | `derivatives/sub-XX/ses-0Y/*_eeg-epo.fif`, `*_exg-epo.fif`, `*_events.dat` |
| EEG speech articulation (Moreira et al. 2025) | ds006104 **v1.0.1** | OpenNeuro ds006104 | overt arms only (second dataset) |

Download ds003626 with the OpenNeuro CLI or `datalad` and pass its root as `--data-root`. No preprocessing is applied on top of the authors' derivatives, apart from the optional band-passes of the band-limited arms.

**Label mapping** (`src/data/inner_speech_loader.py`). Class and condition come from columns 1 and 2 of `*_events.dat`, never from the `.fif` event names, which carry only the word.

| Class code | Word | | Condition code | Condition |
|---|---|---|---|---|
| 0 | arriba (up) | | 0 | pronounced |
| 1 | abajo (down) | | 1 | inner speech |
| 2 | derecha (right) | | 2 | visualised (unused) |
| 3 | izquierda (left) | | | |

The loader raises if an events file is missing or disagrees with the `.fif` word labels (`tests/test_inner_speech_loader.py`).

## 2. Environment

Python 3.11.15, PyTorch 2.11.0 + CUDA 12.8, MNE-Python 1.12.1, scikit-learn 1.9.0, pyRiemann 0.12, NumPy 2.4.6, SciPy 1.17.1. Figures: GNU Octave 9 or MATLAB R2025b.

```
pip install -r requirements.txt
python -m pytest tests -q          # 119 tests, includes the leakage and split checks
```

## 3. Splits, seeds and models

- **Within participant** (`experiments/inner_speech/honest_transfer.py`): for seed *s*, `rng = numpy.random.default_rng(s)`; inner speech is split 60/20/20 (train/val/test) and pronounced speech 80/20 (train/val) by `src/data/splits.py:split_indices`, stratified by class. The overt-only arm splits pronounced speech 60/20/20 with `default_rng([s, 1])`. Calibration subsets of 25/50 % are drawn from the inner training split with the same generator. `torch.manual_seed(s)` sets network initialisation.
- Every arm calls `src/data/leakage.py:assert_disjoint` on its training and test indices before fitting.
- **Seeds:** networks 0–4 in the main table, 0–2 elsewhere; covariance pipelines 0–2 everywhere.
- **Models and training:** `experiments/inner_speech/exp02_cross_dataset.py:get_model` (EEGNet, ShallowConvNet) and `experiments/inner_speech/classical.py` (CSP+LDA, Riemannian). Training config `DEFAULT_CONFIG` in `honest_transfer.py`: Adam, lr 1e-3, weight decay 1e-4, batch 32, ≤100 epochs, early stopping patience 15 on the validation split. Each output directory stores its settings in `run_config.json` and refuses to resume with different ones.

## 4. Commands per table and figure

Run from the repository root with `--data-root <ds003626 root>`; every script is under `experiments/inner_speech/`. Add `--device cuda` for the networks. Rebuild a summary with the same script, `--out-dir`, and `--summarize`.

| Paper item | Output directory | Command |
|---|---|---|
| Table 3, Table A1, Figures 2–5 (networks) | `runs/honest_transfer` | `exp05_honest_transfer.py --models EEGNet ShallowConvNet --seeds 0 1 2 3 4` |
| Table 3, Table A1 (covariance) | `runs/honest_transfer` | `exp05_honest_transfer.py --models CSP+LDA Riemannian --seeds 0 1 2` |
| Held-out session table, Figure 3 | `runs/session_split` | `exp08_session_split.py --windows action --arms source_only shuffle_source target_only transfer shuffle_transfer reciprocal --models EEGNet ShallowConvNet CSP+LDA Riemannian` |
| Fine-tuning table, effect table | `runs/finetune` | `exp05_honest_transfer.py --out-dir runs/finetune --windows action --arms target_only finetune shuffle_finetune --models EEGNet ShallowConvNet --seeds 0 1 2` |
| LOSO, Figures 3–4 | `runs/loso` | `exp07_loso.py --models EEGNet ShallowConvNet CSP+LDA Riemannian` |
| 8–30 Hz table, Figure 5b | `runs/honest_transfer_8_30` | `exp05_honest_transfer.py --out-dir runs/honest_transfer_8_30 --band 8 30 --windows action --arms source_only shuffle_source target_only transfer shuffle_transfer reciprocal --models EEGNet ShallowConvNet CSP+LDA Riemannian --seeds 0 1 2` |
| 30–100 Hz table, Figure 5b | `runs/honest_transfer_30_100` | same with `--band 30 100 --out-dir runs/honest_transfer_30_100` |
| Channel subsets, Figure 5a, Table B1 | `runs/overt_channels` | `exp06_overt_channels.py` |
| External-electrode table | `runs/exg_arms` | `exp10_exg_arms.py --windows action --signals exg lips scalp_regressed --seeds 0 1 2` |
| Concentration-window table | `runs/concentration` | `exp05_honest_transfer.py --out-dir runs/concentration --windows concentration` (networks seeds 0–4, covariance 0–2) |
| No-ICA table | `runs/noica` | `raw_epochs.py --raw-root <ds003626 root> --out-root <new tree>`, then `exp05_honest_transfer.py --data-root <new tree> --out-dir runs/noica --windows action --train-fractions 1.0 --arms source_only shuffle_source target_only shuffle_target transfer shuffle_transfer reciprocal --models EEGNet ShallowConvNet CSP+LDA Riemannian --seeds 0 1 2` |
| Second-dataset table | `runs/second_dataset` | `exp09_second_dataset.py --data-root <ds006104 root>` |
| Figures | `figures/sensors/` | `octave --eval "run('make_figN.m')"` for N = 2…5; scripts read the CSVs above |

The published runs split participants across two GPU and two CPU workers; the split does not change any result, since every row is a deterministic function of participant, model, arm and seed for the covariance pipelines and of the seed for the networks up to GPU non-determinism (about 2.5 percentage points between repeated network runs).

## 5. Effect sizes and detectable effects

The paired effect table and the minimum detectable effects (manuscript Equation for the minimum detectable effect) are computed from `runs/honest_transfer` and `runs/finetune` per participant, mean over seeds, with a 95 % t-interval and MDE = (t₀.₉₅,₉ + t₀.₈₀,₉)·s_d/√10.

## 6. Repository layout

| Path | Content |
|---|---|
| `src/data/` | ds003626 loader (condition from the events table), splits, leakage assertion; ds006104 loading used by the second-dataset arms |
| `src/models/` | EEGNet and ShallowConvNet |
| `experiments/inner_speech/` | harness (`honest_transfer.py`), classical pipelines, MMD, EXG regression, raw rebuild, one script per study arm |
| `experiments/channel_selection/`, `experiments/trackb_4word_floor/floor.py`, `experiments/journal_extension/config_journal.yaml` | helpers reused unchanged from earlier studies (channel positions, ds006104 epoching, label shuffling) |
| `runs/` | per-participant result CSVs and `run_config.json` of every arm in the paper |
| `figures/sensors/` | Octave/MATLAB scripts and PNGs of the figures |
| `tests/` | unit tests |

## 7. Known gaps

- The full subject-dependent appendix table of the manuscript was compiled from `runs/honest_transfer` summaries without a committed generator script.
- The effect-table computation above is not yet a committed script.
- The independent component analysis of the derivatives was fitted by the dataset authors once per session on all its trials (verified from the rank of the released epochs). `experiments/inner_speech/raw_epochs.py` rebuilds the epochs from the raw BDF files without it, into a separate tree; the derivatives stay the default and are never modified. Select the variant only through `--data-root`.
