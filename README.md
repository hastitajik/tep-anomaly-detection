# TEP: Pre-training Enhanced Transformer for Multivariate Time-Series Anomaly Detection

A PyTorch implementation of the two-stage **unsupervised** framework in:

> Wang et al. **Pre-training Enhanced Transformer for multivariate time series anomaly detection.** *Information Fusion* 121 (2025).

## Method

1. **CANets pre-training** (Algorithm 1): temporal convolution (TCE) blocks combined with multi-head attention learn a representation of normal behaviour; the encoder is then frozen.
2. **Reconstruction Transformer** (Algorithm 2): reconstructs each window from the frozen CANets features.
3. **Anomaly Feature Knowledge + POT** (Algorithm 3): an anomaly score built from reconstruction error and a sliding anomaly-knowledge window, thresholded automatically with Peaks-Over-Threshold (generalised Pareto).

Everything is computed **per entity** (normalisation, windowing, POT threshold), so data never leaks across machines/channels.

## What is reproduced

| Paper section | Notebook |
|---|---|
| Main results on SMD, MSL, PSM (Table 2) | §13, precision / recall / F1 / AUC, plus "blind" metrics without point adjustment |
| Reconstruction vs. threshold plots (Fig. 4) | §13 |
| Ablation: CANets / Transformer / Knowledge (Table 4) | §13 |
| CANets kernel visualisation (Fig. 5) | §8 |
| Parameter sensitivity: N, layers, L (Fig. 7) | §9 |
| Robustness to Gaussian noise, 10–100 dB SNR (Fig. 8–9) | §10 |
| Knowledge transfer to a TranAD-style model (Table 3) | §11 |
| Multi-seed and all-entity runs | §13 |

All hyperparameters live in a single `TEPConfig` dataclass. Places where the paper is ambiguous or silent (e.g. the POT risk level `q`) are marked `PAPER GAP` in the code and summarised at the end of the notebook. Section 12 contains smoke tests for every module.

## Data

The real datasets are not included. Download them and set `TEP_DATA_ROOT` (default `./data`):

```
data/
  SMD/train/machine-1-1.txt   SMD/test/machine-1-1.txt   SMD/test_label/machine-1-1.txt
  MSL/MSL_train.npy           MSL/MSL_test.npy           MSL/MSL_test_label.npy
  PSM/train.csv               PSM/test.csv               PSM/test_label.csv
```

- SMD: [NetManAIOps/OmniAnomaly](https://github.com/NetManAIOps/OmniAnomaly)
- MSL: NASA telemanom (preprocessed `.npy` versions are widely mirrored, e.g. in the Anomaly-Transformer repository)
- PSM: [eBay/RANSynCoders](https://github.com/eBay/RANSynCoders)

If a file is missing, the notebook falls back to a synthetic entity so the whole pipeline still runs end to end.

## Status

The included outputs come from a **quick-test run** (`QUICK_TEST = True`, two training epochs) used to check that every stage runs; they are not comparable to the paper's numbers (AUC ≈ 0.84 on SMD and 0.86 on PSM, but F1 is still low at this training budget). Set `QUICK_TEST = False` for full-length training.

## Run it

```bash
pip install -r requirements.txt
export TEP_DATA_ROOT=./data
jupyter notebook tep_anomaly_detection.ipynb
```

## Author

Hasti Tajik — M.Sc. Data Science, University of Science and Culture
