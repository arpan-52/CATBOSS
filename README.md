# CATBOSS — Radio Astronomy RFI Flagging Suite


<p align="center">
<img width="990" height="744" alt="CATBOSS" src="https://github.com/user-attachments/assets/a453d0c6-56ab-44bf-8665-5ca99c6de2d3" />
</p>


<p align="center">
<em>Developed by Arpan Pal — National Centre for Radio Astrophysics, TIFR</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.8%2B-blue" />
  <img src="https://img.shields.io/badge/GPU-CUDA%20accelerated-green" />
  <img src="https://img.shields.io/badge/license-MIT-lightgrey" />
  <img src="https://img.shields.io/badge/status-active-brightgreen" />
</p>

---

Radio frequency interference (RFI) is one of the most persistent headaches in modern radio astronomy. Every observation is contaminated—mobile phones, satellites, radar, even your laptop—and cleaning it out before imaging is absolutely non-negotiable. I built CATBOSS because I needed something fast, flexible, and honest about what it was doing to my data. It can process a full GMRT dataset in minutes without babysitting and is clever enough to handle both time–frequency domain RFI and UV-domain outliers in a single framework.

Originally, I developed these as two separate codebases. Then I realized I have two cats—Pooh and Nimki. So this package now brings both of them together under the same hood, hunting down RFI like a pair of mischievous little predators. And yes—Pooh and Nimki are my real cats. This package is named in their honor.

CATBOSS gives you two complementary flaggers:

- **POOH** — *Parallelized Optimized Outlier Hunter* — works in the dynamic spectra domain, baseline by baseline, with GPU acceleration and multi-pass iterative flagging
- **NIMKI** — *Non-linear Interference Modeling and Korrection Interface* — works in the UV plane, fits a Gabor basis to the visibility amplitude vs UV distance, and flags deviations from the model

NIMKI can be thought of as a 1D version of [gridflag](https://github.com/skunkworks-ra/gridflag), but unlike gridflag, it performs explicit model fitting rather than relying solely on local statistics.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [POOH: Dynamic Spectra Flagging](#pooh-dynamic-spectra-flagging)
- [NIMKI: UV-Domain Flagging](#nimki-uv-domain-flagging)
- [CLI Reference](#cli-reference)
- [Python API](#python-api)
- [Algorithm Details](#algorithm-details)
- [Configuration Files](#configuration-files)
- [Diagnostic Plots](#diagnostic-plots)
- [Performance Tuning](#performance-tuning)
- [Citation](#citation)
- [License](#license)

---

## Features

### POOH — Dynamic Spectra Flagging

Three flagging methods, each with its own strengths:

| Method | Description | Best For |
|--------|-------------|----------|
| **SumThreshold** | Multi-scale combinatorial thresholding (Offringa et al. 2010) | General RFI, broadband bursts, narrow spikes |
| **IQR** | Inter-quartile range outlier detection | Symmetric, well-behaved noise distributions |
| **MAD** | Median Absolute Deviation | Skewed or asymmetric distributions, heavy tails |

**What makes it fast:**
- GPU acceleration via CUDA / Numba — SumThreshold runs entirely on-device, batched across baselines
- Numba JIT-compiled parallel CPU implementations for IQR and MAD
- Memory-aware batching: probes real data dimensions, calculates batch size from actual free VRAM, pre-allocates GPU arrays once per field and reuses them
- Async I/O prefetch with ThreadPoolExecutor while GPU is processing

**What makes it smart:**
- Iterative polynomial bandpass normalization with automatic bad channel detection before flagging
- Multi-pass processing — passes are the outer loop, so each pass sees the full flag state from the previous one across the entire field, not just within a chunk
- Time-frequency chunking for local statistics — `--timebin` and `--freqbin` define the boundaries for statistic calculation, respecting non-stationarity

### NIMKI — UV-Domain Flagging

The visibility amplitude as a function of UV distance follows a predictable structure — a sum of Gaussian-modulated oscillations from the source structure in the sky. NIMKI fits this with a Gabor basis:

```
V(r) = Σᵢ Aᵢ · exp(-(r/σᵢ)²/2) · cos(ωᵢ·r + φᵢ)
```

Residuals from this fit that exceed N×MAD_σ are flagged. This is particularly effective for RFI that contaminates specific baselines or UV ranges in a way that doesn't show up cleanly in the time-frequency domain.

**Key capabilities:**
- C++ accelerated Gabor fitting via pybind11 (Python fallback always available)
- Adaptive "roam around" mode — starts with N components and adds more until improvement drops below a threshold
- Levenberg-Marquardt optimization for robust nonlinear fitting
- MAD-based outlier detection on residuals
- Time-binned processing for time-variable RFI
- Multiprocessing across time chunks

---

## Installation

### Requirements

- Python ≥ 3.8
- NumPy ≥ 1.20
- Dask ≥ 2021.0
- dask-ms ≥ 0.2.0
- python-casacore ≥ 3.4
- Numba ≥ 0.55
- psutil ≥ 5.8

### Basic Installation

```bash
pip install catboss
```

### Installation Options

```bash
# With GPU support
pip install catboss[gpu]

# With plotting support  (PIL + Bokeh + matplotlib)
pip install catboss[plotting]

# Full installation
pip install catboss[all]

# Development
pip install catboss[dev]
```

### From Source

```bash
git clone https://github.com/arpanpal/catboss.git
cd catboss
pip install -e ".[all]"
```

### Building the NIMKI C++ Extension (Optional but Recommended)

The C++ Gabor fitter is significantly faster than the Python fallback. Build it once:

```bash
cd src/catboss/nimki
pip install pybind11
python setup_cpp.py build_ext --inplace
```

If the build fails for any reason, CATBOSS falls back to the Python implementation automatically — nothing breaks.

### GPU Setup

```bash
# Check CUDA
nvidia-smi

# Install CUDA-enabled Numba (conda is easiest)
conda install numba cudatoolkit

# Verify
python -c "from catboss.utils import is_gpu_available; print('GPU:', is_gpu_available())"
```

Without a GPU, CATBOSS runs perfectly on CPU — just slower. It detects this automatically at startup and switches modes without any configuration needed.

---

## Quick Start

### POOH

```bash
# SumThreshold (default, recommended starting point)
catboss pooh data.ms --apply-flags

# IQR
catboss pooh data.ms --method iqr --apply-flags

# MAD
catboss pooh data.ms --method mad --apply-flags

# Dry run with diagnostic plots (no flags written)
catboss pooh data.ms --dry-run --plots --plot-dir my_plots
```

### NIMKI

```bash
# Basic Gabor fitting
catboss nimki data.ms --apply-flags

# Adaptive component selection
catboss nimki data.ms --roam-around --apply-flags

# With plots
catboss nimki data.ms --plots --apply-flags
```

### Help

```bash
catboss --help
catboss pooh --help
catboss nimki --help
```

---

## POOH: Dynamic Spectra Flagging

### Method Selection and Parameters

#### SumThreshold

The workhorse. Implements the Offringa et al. (2010) algorithm with multi-scale combinatorial thresholding. Works by sliding windows of increasing size across both time and frequency, with the threshold scaling down as the window grows — so it catches both sharp narrow spikes and broader, lower-amplitude RFI that a single-sample threshold would miss.

```bash
catboss pooh data.ms \
    --method sumthreshold \
    --sigma 6.0 \
    --rho 1.5 \
    --combinations 1,2,4,8,16,32,64 \
    --apply-flags
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--sigma` | 6.0 | Base threshold multiplier χ₁ |
| `--rho` | 1.5 | Window reduction factor ρ |
| `--combinations` | 1,2,4,8,16,32,64 | Window sizes M |

Threshold scaling: `χₘ = χ₁ / ρ^(log₂(M))` — with defaults at M=64, the threshold drops to ~1.8σ, which is aggressive but the cumulative flagging from smaller windows protects against false positives.

#### IQR

```bash
catboss pooh data.ms --method iqr --iqr-factor 1.5 --apply-flags
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--iqr-factor` | 1.5 | Flags above Q3 + factor×IQR |

#### MAD

```bash
catboss pooh data.ms --method mad --mad-sigma 5.0 --apply-flags
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--mad-sigma` | 5.0 | Flags above median + sigma×MAD×1.4826 |

### Bandpass Normalization

Before flagging, POOH normalizes the bandpass to remove the spectral response of the instrument. Without this, narrow-band RFI near band edges and strong spectral features can dominate the statistics.

```bash
catboss pooh data.ms \
    --poly-order 5 \
    --deviation-threshold 5.0 \
    --apply-flags

# Skip if your data is already bandpass-calibrated
catboss pooh data.ms --no-bandpass --apply-flags
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--poly-order` | 5 | Polynomial degree for bandpass fit (3–9 typical) |
| `--deviation-threshold` | 5.0 | Sigma for bad channel detection in the bandpass |
| `--no-bandpass` | False | Skip normalization entirely |

### Multi-Pass Processing

Each pass sees the flags from all previous passes — critical for iterative flagging where early passes catch the strong RFI and later passes find the fainter stuff that was hidden underneath.

```bash
# Same parameters, multiple passes
catboss pooh data.ms --passes 3 --apply-flags

# Different parameters per pass (config file)
catboss pooh data.ms --config passes.json --apply-flags
```

### Time-Frequency Chunking

For data where the noise statistics vary across the observation (most real data), local statistics give much better results than global ones. `--timebin` and `--freqbin` define the boundaries within which statistics are computed — they don't affect the GPU batching, just the stat windows.

```bash
# By samples / channels
catboss pooh data.ms --timebin 100 --freqbin 128 --apply-flags

# By physical units
catboss pooh data.ms --timebin-min 10 --freqbin-mhz 5 --apply-flags
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--timebin` | None | Time chunk in samples |
| `--timebin-min` | None | Time chunk in minutes |
| `--freqbin` | None | Frequency chunk in channels |
| `--freqbin-mhz` | None | Frequency chunk in MHz |

Without chunking, statistics are computed globally — fine for short observations with stationary noise.

### Data Selection

```bash
catboss pooh data.ms \
    --datacolumn DATA \
    --field 0,1 \
    --spw 0 \
    --corr 0,3 \
    --baseline "0-1,0-2,1-2" \
    --exclude-autocorr \
    --apply-flags
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--datacolumn` | DATA | DATA, CORRECTED_DATA, or RESIDUAL_DATA |
| `--field` | all | Field IDs (comma-separated) |
| `--spw` | all | SPW IDs (comma-separated) |
| `--corr` | all | Correlation indices (comma-separated) |
| `--baseline` | all | Baselines as "ant1-ant2,..." |
| `--exclude-autocorr` | False | Skip auto-correlations |

### Flag Behavior

```bash
catboss pooh data.ms --apply-flags          # Write flags to MS FLAG column
catboss pooh data.ms --dry-run --plots      # Diagnostic mode — no writing
catboss pooh data.ms --propagate-flags      # Copy flags to all correlations
```

---

## NIMKI: UV-Domain Flagging

### Overview

POOH is great for RFI that's localized in time-frequency space. But some interference — especially from compact, bright sources that scatter off terrestrial structures — appears as outliers in the UV plane rather than in time-frequency. NIMKI handles this by fitting the expected smooth visibility envelope and flagging deviations.

### Gabor Fitting Parameters

```bash
catboss nimki data.ms \
    --n-components 5 \
    --sigma 5.0 \
    --max-iter 500 \
    --tolerance 1e-8 \
    --apply-flags
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n-components` | 5 | Number of Gabor basis components |
| `--sigma` | 5.0 | Flagging threshold in units of MAD_σ |
| `--max-iter` | 500 | Max Levenberg-Marquardt iterations |
| `--tolerance` | 1e-8 | Convergence tolerance |

### Adaptive Component Selection ("Roam Around")

In "roam around" mode, NIMKI starts with the minimum number of components and keeps adding more as long as the RMS improves meaningfully. This avoids both underfitting (too few components) and overfitting (fitting the RFI itself).

```bash
catboss nimki data.ms \
    --roam-around \
    --n-components 3 \
    --max-components 12 \
    --min-improvement 0.05 \
    --apply-flags
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--roam-around` | False | Enable adaptive component selection |
| `--max-components` | 12 | Maximum components |
| `--min-improvement` | 0.05 | Stop adding if RMS improvement < 5% |

### Time Chunking

```bash
catboss nimki data.ms --timebin 30 --apply-flags   # 30-minute chunks
```

For long observations, process in time bins — the UV coverage changes and the noise statistics can drift.

### Data Selection and Flag Behavior

Same selection options as POOH (`--datacolumn`, `--field`, `--spw`, `--corr`).

```bash
catboss nimki data.ms --flag-all-corr --apply-flags      # Flag all corrs if any bad
catboss nimki data.ms --flag-selected-only --apply-flags  # Only flag the bad corr
```

### Performance

```bash
catboss nimki data.ms --ncpu 8 --apply-flags   # Use 8 cores
catboss nimki data.ms --ncpu 0 --apply-flags   # Use all available cores
```

---

## CLI Reference

### Complete POOH Options

```
catboss pooh <ms_path> [OPTIONS]

Method:
  --method {sumthreshold,iqr,mad}       Flagging method (default: sumthreshold)

SumThreshold:
  --sigma FLOAT                          Base threshold χ₁ (default: 6.0)
  --rho FLOAT                            Reduction factor ρ (default: 1.5)
  --combinations STR                     Window sizes (default: 1,2,4,8,16,32,64)

IQR:
  --iqr-factor FLOAT                     IQR multiplier (default: 1.5)

MAD:
  --mad-sigma FLOAT                      MAD sigma (default: 5.0)

Bandpass:
  --poly-order INT                       Polynomial degree (default: 5)
  --deviation-threshold FLOAT            Bad channel sigma (default: 5.0)
  --no-bandpass                          Skip bandpass normalization

Multi-Pass:
  --passes INT                           Number of passes (default: 1)
  --config FILE                          JSON config for per-pass parameters

Data Selection:
  --datacolumn {DATA,CORRECTED_DATA,RESIDUAL_DATA}
  --field STR                            Field IDs or "all"
  --spw STR                              SPW IDs or "all"
  --corr STR                             Correlations or "all"
  --baseline STR                         Baselines as "0-1,0-2" or "all"
  --exclude-autocorr                     Skip auto-correlations

Chunking:
  --timebin INT                          Time chunk (samples)
  --timebin-min FLOAT                    Time chunk (minutes)
  --freqbin INT                          Frequency chunk (channels)
  --freqbin-mhz FLOAT                    Frequency chunk (MHz)

Flags:
  --apply-flags                          Write flags to MS
  --dry-run                              Plots only, no writing
  --propagate-flags                      Propagate across correlations

Output:
  --plots                                Generate diagnostic plots
  --plot-dir DIR                         Plot directory (default: pooh_plots)

Performance:
  --mode {auto,cpu,gpu}                  Processing mode (default: auto)
  --chunk-size INT                       I/O chunk size in rows (default: 200000)
  --max-memory FLOAT                     Max memory fraction (default: 0.8)

General:
  -v, --verbose                          Verbose output
  --help                                 Show this message and exit
```

### Complete NIMKI Options

```
catboss nimki <ms_path> [OPTIONS]

Gabor Fitting:
  --n-components INT                     Number of components (default: 5)
  --roam-around                          Adaptive component selection
  --max-components INT                   Max components in adaptive mode (default: 12)
  --min-improvement FLOAT                Stop threshold (default: 0.05)
  --max-iter INT                         Max LM iterations (default: 500)
  --tolerance FLOAT                      Convergence tolerance (default: 1e-8)

Flagging:
  --sigma FLOAT                          Sigma threshold in MAD units (default: 5.0)

Data Selection:
  --datacolumn {DATA,CORRECTED_DATA,MODEL_DATA}
  --field STR                            Field IDs
  --spw STR                              SPW IDs
  --corr STR                             Correlations

Time Chunking:
  --timebin FLOAT                        Time bin in minutes (default: 30.0)

Flags:
  --apply-flags                          Write flags to MS
  --dry-run                              Plots only
  --flag-all-corr                        Flag all correlations if any bad (default)
  --flag-selected-only                   Only flag the detected bad correlation

Output:
  --plots                                Generate diagnostic plots
  --plot-dir DIR                         Plot directory (default: nimki_plots)

Performance:
  --ncpu INT                             CPUs to use (0 = all, default: 0)

General:
  -v, --verbose                          Verbose output
  --help                                 Show this message and exit
```

---

## Python API

### POOH

```python
from catboss.pooh import hunt_ms
from catboss.logger import setup_logger, print_banner

print_banner()
logger = setup_logger("catboss", verbose=True)

options = {
    'method': 'sumthreshold',
    'sigma': 6.0,
    'rho': 1.5,
    'combinations': '1,2,4,8,16,32,64',
    'poly_order': 5,
    'deviation_threshold': 5.0,
    'passes': 3,
    'field': '0,1',
    'corr': '0,3',
    'timebin': 100,
    'freqbin': 128,
    'apply_flags': True,
    'plots': True,
    'plot_dir': 'outputs',
    'mode': 'auto',
    'logger': logger,
}

results = hunt_ms('data.ms', options)

print(f"Time:          {results['total_processing_time']:.1f}s")
print(f"Baselines:     {results['baselines_processed']}")
print(f"New flags:     {results['new_flags']:,}")
print(f"New flagged:   {results['new_percent_flagged']:.2f}%")
```

### NIMKI

```python
from catboss.nimki import hunt_ms
from catboss.logger import setup_logger, print_banner

print_banner()
logger = setup_logger("catboss", verbose=True)

options = {
    'n_components': 5,
    'roam_around': True,
    'max_components': 12,
    'min_improvement': 0.05,
    'sigma': 5.0,
    'timebin': 30.0,
    'apply_flags': True,
    'plots': True,
    'ncpu': 0,
    'logger': logger,
}

results = hunt_ms('data.ms', options)

print(f"Outliers:      {results['total_outliers']:,}")
print(f"Flags written: {results['total_flags']:,}")
print(f"Flagged:       {results['percent_flagged']:.2f}%")
```

### Low-Level API

If you want to use the flagging methods directly without going through the full MS pipeline:

```python
import numpy as np
from catboss.pooh.methods import SumThresholdMethod, IQRMethod, MADMethod
from catboss.pooh.bandpass import normalize_bandpass

# Simulate some data — (n_time, n_freq) amplitude array
amp   = np.random.randn(500, 1024).astype(np.float32) + 10.0
flags = np.zeros_like(amp, dtype=bool)

# Bandpass normalize
amp_norm, flags, bandpass, bad_chans = normalize_bandpass(
    amp, flags, poly_order=5, deviation_threshold=5.0
)
print(f"Bad channels detected: {np.sum(bad_chans)}")

# SumThreshold
method = SumThresholdMethod({
    'sigma': 6.0,
    'rho': 1.5,
    'combinations': [1, 2, 4, 8, 16],
    'use_gpu': True,
})
new_flags = method.flag(amp_norm, flags)
print(f"Flagged: {np.sum(new_flags)} / {amp.size} samples ({100*np.mean(new_flags):.2f}%)")
```

---

## Algorithm Details

### POOH Processing Pipeline

```
1. BANDPASS NORMALIZATION
   ├── Compute median amplitude per frequency channel
   ├── Fit polynomial to the bandpass (iterative sigma-clipping)
   ├── Detect bad channels via residual analysis
   └── Normalize: amp[t, f] /= bandpass[f]

2. THRESHOLD CALCULATION
   ├── Per-channel robust statistics (median, MAD)
   ├── Base thresholds from MAD × sigma
   └── Window-scaled thresholds: χₘ = χ₁ / ρ^(log₂(M))

3. MULTI-PASS FLAGGING  (passes outer, chunks inner)
   For each pass:
     For each time chunk × frequency chunk:
       For each window size M in combinations:
         ├── Sum M consecutive samples in time → flag if sum > χₘ × threshold
         └── Sum M consecutive samples in freq → flag if sum > χₘ × threshold
       flags_chunk updated in-place

4. FLAG COMBINATION
   final_flags = existing_flags | new_flags
   (existing flags always preserved)
```

### NIMKI Processing Pipeline

```
1. DATA COLLECTION (per time chunk, per SPW)
   ├── Load unflagged visibilities
   ├── Calculate UV distances in wavelengths per channel
   └── Collect (uv, amplitude) pairs per correlation

2. GABOR FITTING
   ├── Initialize N Gabor components with reasonable guesses
   ├── Levenberg-Marquardt nonlinear least squares
   ├── [roam-around] Add components while ΔRMS > min_improvement
   └── Compute predicted amplitudes V̂(r)

3. OUTLIER DETECTION
   ├── residuals = |observed| - V̂(r)
   ├── MAD_σ = 1.4826 × median(|residuals - median(residuals)|)
   └── outliers = |residuals| > sigma × MAD_σ

4. FLAG APPLICATION
   ├── Map outliers back to (MS row, channel, correlation)
   └── Write to FLAG column (batch mode, OR with existing)
```

### SumThreshold Theory

The SumThreshold algorithm (Offringa et al. 2010) exploits the fact that RFI tends to be correlated across time or frequency while thermal noise is not. By summing M consecutive samples and applying a scaled threshold:

- **M=1**: Individual spikes above 6σ
- **M=2**: Two-sample bursts above 4σ (would be missed as single samples)
- **M=64**: Broad RFI above 1.8σ cumulative — catches wide structures

The threshold scaling `χₘ = χ₁ / ρ^(log₂(M))` ensures statistically consistent flagging rates across scales. The sequential flagging from M=1 upward means each window size operates on data where the strongest RFI is already flagged.

### Gabor Basis Fitting (NIMKI)

The visibility amplitude vs. UV distance is modeled as a sum of Gabor functions:

```
V(r) = Σᵢ Aᵢ · exp(-(r/σᵢ)²/2) · cos(ωᵢ·r + φᵢ)
```

**Why Gabor?** The Fourier relationship between the sky brightness distribution and the visibilities means:
- **Point sources** → narrow Gaussians in UV space (wide σᵢ in the model)
- **Extended sources** → broad Gaussians with oscillations (ωᵢ captures the spatial scale)
- **Double sources / rings** → strong oscillatory component

RFI that doesn't fit this model shows up as outliers in the residuals. The MAD-based detection ensures robustness — a few strong outliers don't blow up the threshold.

---

## Configuration Files

### Multi-Pass Configuration

The most powerful way to use POOH is with a per-pass config. Start aggressive to catch strong RFI, then refine:

```json
{
  "passes": [
    {
      "method": "sumthreshold",
      "sigma": 8.0,
      "rho": 1.5,
      "combinations": [1, 2, 4, 8]
    },
    {
      "method": "sumthreshold",
      "sigma": 5.5,
      "rho": 1.4,
      "combinations": [1, 2, 4, 8, 16, 32]
    },
    {
      "method": "mad",
      "mad_sigma": 4.5
    }
  ]
}
```

```bash
catboss pooh data.ms --config passes.json --apply-flags
```

### Full Configuration Example

```json
{
  "method": "sumthreshold",
  "sigma": 6.0,
  "rho": 1.5,
  "combinations": [1, 2, 4, 8, 16, 32, 64],
  "poly_order": 5,
  "deviation_threshold": 5.0,
  "passes": 2,
  "field": "0,1,2",
  "spw": "0",
  "corr": "0,3",
  "timebin": 100,
  "freqbin": 128,
  "apply_flags": true,
  "propagate_flags": true,
  "plots": true,
  "plot_dir": "rfi_outputs",
  "mode": "auto",
  "verbose": true
}
```

---

## Diagnostic Plots

### POOH Plots

When `--plots` is enabled, POOH generates a self-contained HTML viewer per field:

**Structure:**
```
pooh_plots/
├── index.html                          ← Master index of all fields
├── field0_3C147.html                   ← Field viewer (self-contained)
├── field1_3C147_OFF.html
└── ...
```

**Field viewer:**
- Clickable sidebar listing all baselines with flag statistics
- Left panel: raw dynamic spectra (inferno colormap)
- Right panel: same spectra with flag overlay
  - **Red** (`#FF4444`): pre-existing flags
  - **Cyan** (`#00FFFF`): new POOH flags
- Keyboard navigation (arrow keys)
- Prev / Next buttons

### NIMKI Plots

NIMKI generates one Bokeh HTML per `(field, spw)` with full interactivity (zoom, pan, hover):

**Structure:**
```
nimki_plots/
├── nimki_field0_spw0.html
├── nimki_field1_spw0.html
└── ...
```

**Per-file content:**
- Correlation tabs
- Left panel: UV distance (λ) vs amplitude — clean input data NIMKI worked with
- Right panel: same data + new NIMKI outliers in **magenta** (`#FF2D95`) + Gabor fit line
- Hover tool showing UV distance and amplitude
- Linked axes between left and right panels

---

## Performance Tuning

### GPU Acceleration

CATBOSS detects your GPU at startup and uses it automatically. If you want to check:

```python
from catboss.utils import is_gpu_available, print_gpu_info
print_gpu_info()
```

For maximum GPU utilization: let CATBOSS calculate the batch size itself. It probes the actual data dimensions, checks free VRAM, and pre-allocates device arrays for the entire field — reusing them across batches rather than allocating fresh on every iteration.

**GPU requirements:**
- NVIDIA GPU with CUDA support
- CUDA toolkit ≥ 11.0
- `conda install numba cudatoolkit`


## Citation

If CATBOSS contributed to your science, please cite:

```bibtex
@software{catboss2025,
  author      = {Pal, Arpan},
  title       = {CATBOSS},
  year        = {2025},
  institution = {National Centre for Radio Astrophysics, TIFR},
  url         = {https://github.com/arpan-52/CATBOSS}
}
```

For the SumThreshold algorithm implemented in POOH:

```bibtex
@article{offringa2010,
  author  = {Offringa, A. R. and de Bruyn, A. G. and Biehl, M. and others},
  title   = {Post-correlation radio frequency interference classification methods},
  journal = {Monthly Notices of the Royal Astronomical Society},
  volume  = {405},
  pages   = {155--167},
  year    = {2010},
  doi     = {10.1111/j.1365-2966.2010.16471.x}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Author

**Arpan Pal**
National Centre for Radio Astrophysics
Tata Institute of Fundamental Research (NCRA-TIFR)
Pune, India

---

*Built out of frustration with slow flaggers and named after two cats who have no idea what radio astronomy is but are very good at catching things.*
