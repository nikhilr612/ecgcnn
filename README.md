# ECG-CNN: Heartbeat Classification using 1D CNN

This project implements a 1D Convolutional Neural Network (CNN) for classifying heartbeats from ECG signals using the MIT-BIH Arrhythmia Database.

## Overview

ECG-CNN uses a ResNet-style architecture with 1D convolutional blocks to classify heartbeats into 6 different categories according to the AAMI5 standard:

- 0: Normal beats (N, R, L, e, j)
- 1: Ventricular ectopic beats (V, E, I)
- 2: Supraventricular ectopic beats (a, n, A, S)
- 3: Fusion beats (F, f, /)
- 4: Unknown beats (Q)
- 5: Ventricular flutter waves ([, !, ])

## Prerequisites

- Python 3.13 or higher
- [uv](https://github.com/astral-sh/uv) - An extremely fast Python package installer and resolver

## Installation

1. Install uv if you haven't already:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Clone the repository:
   ```bash
   git clone https://github.com/nikhilr612/ecgcnn.git
   cd ecgcnn
   ```

3. (Optional) Sync dependencies upfront:
   ```bash
   uv sync
   ```
   This pre-installs all dependencies and creates the virtual environment. If you skip this step, `uv run` will handle it automatically on first use.

That's it! `uv` will automatically handle virtual environment creation and dependency installation when you run commands.

## Dataset Preparation

### MIT-BIH Arrhythmia Database

The MIT-BIH Arrhythmia Database contains 48 half-hour excerpts of two-channel ambulatory ECG recordings, obtained from 47 subjects studied by the BIH Arrhythmia Laboratory between 1975 and 1979. The recordings were digitized at 360 samples per second per channel with 11-bit resolution over a 10 mV range.

The database includes:
- 23 recordings (numbered 100-124) chosen at random from a set of 4,000 24-hour ambulatory ECG recordings
- 25 recordings (numbered 200-234) selected to include less common but clinically significant arrhythmias

Each record contains two ECG leads - usually a modified limb lead II (MLII) and a modified lead V1 (occasionally V2 or V5).

### Downloading the Raw Data

1. The raw MIT-BIH data files must be present in the `data/raw` directory. You can download them from PhysioNet:
   ```bash
   # Create the data directory if it doesn't exist
   mkdir -p data/raw

   # Download the MIT-BIH Arrhythmia Database
   wget -r -np -nd -P data/raw https://physionet.org/files/mitdb/1.0.0/
   ```

2. Process the raw data to create the Parquet dataset:
   ```bash
   uv run makedataset.py
   ```

This will create a `mitbih.parquet` file in the `data` directory, which contains windowed signals for all annotated heartbeats. The script extracts 128-sample windows (with normalization) centered on each heartbeat annotation and categorizes them according to the AAMI5 standard.

## Training the Model

### Training Options

**Default behavior (hyperparameter search):**
```bash
uv run main.py --save-hparams best_config.json
```

**Use saved configuration:**
```bash
uv run main.py --model-config best_config.json
```

**Use defaults (skip search):**
```bash
uv run main.py --use-defaults
```

**With reproducible seed:**
```bash
uv run main.py --seed 42
```

### Key Command-Line Options

- `--use-defaults`: Skip hyperparameter search, use defaults
- `--save-hparams FILE`: Save best hyperparameters to JSON file
- `--model-config FILE`: Load model configuration from JSON file
- `--n-trials N`: Number of search trials (default: 20)
- `--seed N`: Random seed for reproducibility
- `--n-epochs N`: Training epochs (default: 50)
- `--batch-size N`: Batch size (default: 32)

See `uv run main.py --help` for all options.

## Model Architecture

The model consists of:

1. An initial ConvBlock that processes the input signal
2. Multiple stacked ConvBlocks (configurable via the `depth` parameter)
3. An MLP head for final classification

Each ConvBlock contains:
- Two 1D convolutional layers with batch normalization and ReLU activation
- A residual connection
- Max pooling for downsampling

## Training Process

### Hyperparameter Search

The automated hyperparameter search uses Optuna to optimize:
- Network depth (number of ConvBlocks)
- Kernel sizes for convolutional layers
- Number of channels
- Max pooling size

The search evaluates each configuration by training for the specified number of epochs and selecting the configuration with the best validation loss.

### Model Training

The training process uses:
- AdamW optimizer with a linear warmup
- Stochastic Weight Averaging (SWA) for better generalization
- Model checkpointing based on validation loss (every 2-4 epochs)
- Automatic learning rate tuning (if not specified)

The specified random seed is used during the training process for reproducibility. Note however, that the seed applies only to training and not to the hyperparameter search.

### Checkpointing

Model checkpoints are automatically saved during both hyperparameter search and final training:
- During search: Best model per trial saved in `models/` with prefix `opt-`
- During training: Top 5 models saved based on validation loss
- Checkpoints include model state, optimizer state, and training metadata

## Hyperparameter Search

The automated search optimizes network depth, kernel sizes, channels, and pooling sizes using Optuna. For faster experimentation, reduce epochs during search:

```bash
uv run main.py --n-epochs 20 --n-trials 30 --save-hparams config.json
```

Then use the saved configuration for full training:

```bash
uv run main.py --model-config config.json --n-epochs 50
```

## Results

After training, model checkpoints are saved in the `models/` directory. The best models (based on validation loss) are saved every 2 epochs. The models directory is automatically created if it doesn't exist.

## Requirements

- Python 3.13+
- Lightning 2.3.3+
- PyTorch 2.5.1+
- Polars 1.32.3+
- NumPy 2.3.2+
- WFDB 4.1.2+
- Altair 5.5.0+ (for visualization)
- uv (for dependency management)
- Optuna 4.5.0+ (for hyperparameter search)

All Python dependencies are defined in the `pyproject.toml` file and locked in `uv.lock` to ensure reproducible installations.

## Citation

If you use this code or the processed dataset in your research, please cite:

1. The original MIT-BIH Arrhythmia Database:
   ```
   Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database.
   IEEE Eng in Med and Biol 20(3):45-50 (May-June 2001). (PMID: 11446209)
   ```

2. PhysioNet:
   ```
   Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000).
   PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals.
   Circulation [Online]. 101 (23), pp. e215–e220.
   ```

3. WFDB Python package:
   ```
   Xie, C., McCullum, L., Johnson, A., Pollard, T., Gow, B., & Moody, B. (2023).
   Waveform Database Software Package (WFDB) for Python (version 4.1.0).
   PhysioNet. https://doi.org/10.13026/9njx-6322.
   ```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
