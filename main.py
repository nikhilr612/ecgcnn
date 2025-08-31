import argparse
from dataclasses import asdict
from model import EcgModelConfig
from train import train_ecgcnn, TrainingConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Train ECG CNN model")
    # Model hyperparameters
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
    return parser.parse_args()


def main():
    args = parse_args()
    model_config = EcgModelConfig(
        depth=args.depth,
        kernel_size1=args.kernel_size1,
        kernel_size2=args.kernel_size2,
        max_pool_size=args.max_pool_size,
        model_channels=args.model_channels,
    )
    training_config = TrainingConfig(
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        swa_lrs=args.swa_lrs,
        n_epochs=args.n_epochs,
    )
    print("Training ECG CNN with config:")
    print("Model config:", asdict(model_config))
    print("Training config:", asdict(training_config))
    train_ecgcnn(
        model_config,
        training_config,
        parquet_path=args.parquet_path,
    )


if __name__ == "__main__":
    main()
