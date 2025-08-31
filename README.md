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
   git clone <repository-url>
   cd ecgcnn
   ```

3. (Optional) Sync dependencies upfront:
   ```bash
   uv sync
   ```
   This pre-installs all dependencies and creates the virtual environment. If you skip this step, `uv run` will handle it automatically on first use.

That's it! `uv` will automatically handle virtual environment creation and dependency installation when you run commands.

## Dataset Preparation

### About the MIT-BIH Arrhythmia Database

The MIT-BIH Arrhythmia Database contains 48 half-hour excerpts of two-channel ambulatory ECG recordings, obtained from 47 subjects studied by the BIH Arrhythmia Laboratory between 1975 and 1979. The recordings were digitized at 360 samples per second per channel with 11-bit resolution over a 10 mV range.

The database includes:
- 23 recordings (numbered 100-124) chosen at random from a set of 4,000 24-hour ambulatory ECG recordings
- 25 recordings (numbered 200-234) selected to include less common but clinically significant arrhythmias

Each record contains two ECG leads - usually a modified limb lead II (MLII) and a modified lead V1 (occasionally V2 or V5).

### Downloading the Raw Data

1. The raw MIT-BIH data files are included in the `data/raw` directory. If they're not already present, you can download them from PhysioNet:
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

You can train the model with default hyperparameters using:

```bash
uv run main.py
```

### Customizing Training

The training process can be customized with various command-line arguments:

#### Model Hyperparameters:
- `--depth`: Number of ConvBlocks to stack (default: 2)
- `--kernel-size1`: Kernel size for first conv in block (default: 3)
- `--kernel-size2`: Kernel size for second conv in block (default: 3)
- `--max-pool-size`: Max pool size in block (default: 5)
- `--model-channels`: Number of output channels per block (default: 32)

#### Training Hyperparameters:
- `--warmup-steps`: Number of warmup steps for LR scheduler (default: 500)
- `--learning-rate`: Learning rate (if not specified, will be tuned automatically)
- `--batch-size`: Batch size (default: 32)
- `--swa-lrs`: Stochastic Weight Averaging learning rate (default: 1e-3)
- `--n-epochs`: Number of epochs (default: 50)

#### Data:
- `--parquet-path`: Path to MIT-BIH parquet file (default: "data/mitbih.parquet")

Example:
```bash
uv run main.py --depth 3 --kernel-size1 5 --kernel-size2 3 --max-pool-size 3 --model-channels 64 --batch-size 64 --n-epochs 100
```

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

The training process uses:
- Lightning for training management
- AdamW optimizer with a linear warmup
- Stochastic Weight Averaging (SWA) for better generalization
- Model checkpointing based on validation loss
- Automatic learning rate tuning (if not specified)

## Results

After training, model checkpoints are saved in the `models/` directory. The best models (based on validation loss) are saved every 2 epochs. The models directory is automatically created if it doesn't exist.

## Project Structure

- `main.py`: Entry point for training the model
- `model.py`: Definition of the neural network architecture
- `train.py`: Training configuration and process
- `dataloader.py`: Data loading utilities for the MIT-BIH dataset
- `makedataset.py`: Script to process raw MIT-BIH data into a usable format
- `data/`: Directory containing the dataset
  - `raw/`: Raw MIT-BIH data files
  - `mitbih.parquet`: Processed dataset
- `models/`: Directory where trained model checkpoints are saved

## Requirements

- Python 3.13+
- Lightning 2.3.3+
- PyTorch 2.5.1+
- Polars 1.32.3+
- NumPy 2.3.2+
- WFDB 4.1.2+
- Altair 5.5.0+ (for visualization)
- uv (for dependency management)

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
