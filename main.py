import argparse
import json
import os
from dataclasses import asdict
import lightning as L
from model import EcgModelConfig
from train import train_ecgcnn, TrainingConfig, hparam_search, HparamTuningConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Train ECG CNN model")

    # Flag to control hyperparameter search behavior
    parser.add_argument(
        "--use-defaults",
        action="store_true",
        help="Use default hyperparameters instead of hyperparameter search",
    )

    # Hyperparameter file management
    parser.add_argument(
        "--save-hparams",
        type=str,
        help="Path to save best hyperparameters to (JSON format)",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        help="Path to load model configuration from (JSON format)",
    )

    # Hyperparameter search configuration
    parser.add_argument(
        "--n-trials",
        type=int,
        default=5,
        help="Number of trials for hyperparameter search (default: 5)",
    )
    parser.add_argument(
        "--min-depth",
        type=int,
        default=2,
        help="Minimum depth for hyperparameter search (default: 2)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Maximum depth for hyperparameter search (default: 6)",
    )

    # Model hyperparameters (used when --use-defaults is specified)
    parser.add_argument(
        "--depth", type=int, default=2, help="Number of ConvBlocks to stack"
    )
    parser.add_argument(
        "--kernel-size1",
        type=int,
        default=3,
        help="Kernel size for first conv in block",
    )
    parser.add_argument(
        "--kernel-size2",
        type=int,
        default=3,
        help="Kernel size for second conv in block",
    )
    parser.add_argument(
        "--max-pool-size", type=int, default=5, help="Max pool size in block"
    )
    parser.add_argument(
        "--model-channels",
        type=int,
        default=32,
        help="Number of output channels per block",
    )

    # Training hyperparameters
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=500,
        help="Number of warmup steps for LR scheduler",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=None, help="Learning rate"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--swa-lrs", type=float, default=1e-3, help="SWA learning rate")
    parser.add_argument("--n-epochs", type=int, default=50, help="Number of epochs")

    # Data
    parser.add_argument(
        "--parquet-path",
        type=str,
        default="data/mitbih.parquet",
        help="Path to MIT-BIH parquet file",
    )

    # Reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def save_hyperparameters(model_config: EcgModelConfig, filepath: str) -> None:
    """Save model configuration to a JSON file."""
    config_dict = asdict(model_config)
    os.makedirs(os.path.dirname(filepath), exist_ok=True) if os.path.dirname(
        filepath
    ) else None
    with open(filepath, "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"Hyperparameters saved to: {filepath}")


def load_model_config(filepath: str) -> EcgModelConfig:
    """Load model configuration from a JSON file."""
    with open(filepath, "r") as f:
        config_dict = json.load(f)
    print(f"Model configuration loaded from: {filepath}")
    return EcgModelConfig(**config_dict)


def get_default_model_config(args) -> EcgModelConfig:
    """Create model config from CLI arguments."""
    return EcgModelConfig(
        depth=args.depth,
        kernel_size1=args.kernel_size1,
        kernel_size2=args.kernel_size2,
        max_pool_size=args.max_pool_size,
        model_channels=args.model_channels,
    )


def get_model_config_from_search(args) -> EcgModelConfig:
    """Perform hyperparameter search and return best config."""
    print("Performing hyperparameter search...")
    search_config = HparamTuningConfig(
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        kernel_size1_choices=[3, 5, 7],
        kernel_size2_choices=[3, 5, 7],
        model_channels_choices=[16, 32, 64],
        max_pool_size_choices=[2, 3, 5],
        n_trials=args.n_trials,
    )

    training_config = TrainingConfig(
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        swa_lrs=args.swa_lrs,
        n_epochs=args.n_epochs,
    )

    print("Search config:", asdict(search_config))
    print("Training config:", asdict(training_config))

    best_model_config = hparam_search(search_config, training_config, args.parquet_path)
    print("\nBest hyperparameters found:")
    print("Best model config:", asdict(best_model_config))

    return best_model_config


def main():
    args = parse_args()

    # Set seed for reproducibility if specified
    if args.seed is not None:
        L.seed_everything(args.seed)
        print(f"Seed set to: {args.seed}")

    # Determine model configuration
    model_config = None

    # Try to load from file first
    if args.model_config and os.path.exists(args.model_config):
        try:
            model_config = load_model_config(args.model_config)
        except Exception as e:
            print(f"Error loading model configuration: {e}")
            model_config = None

    # If no config loaded, determine how to get it
    if model_config is None:
        if args.use_defaults:
            print("Using default hyperparameters")
            model_config = get_default_model_config(args)
        else:
            model_config = get_model_config_from_search(args)

            # Save if requested
            if args.save_hparams:
                save_hyperparameters(model_config, args.save_hparams)

    # Create training configuration
    training_config = TrainingConfig(
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        swa_lrs=args.swa_lrs,
        n_epochs=args.n_epochs,
    )

    print("\nFinal configuration:")
    print("Model config:", asdict(model_config))
    print("Training config:", asdict(training_config))

    # Train the model
    print("\nStarting training...")
    train_ecgcnn(
        model_config,
        training_config,
        parquet_path=args.parquet_path,
    )


if __name__ == "__main__":
    main()
