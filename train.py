"""
Script to train the ECG CNN model.
"""

from dataclasses import dataclass, field
from lightning.pytorch.callbacks import ModelCheckpoint, StochasticWeightAveraging
import optuna
from dataloader import MitBih
from model import EcgModelConfig, LitEcg
import lightning as L
from lightning.pytorch.tuner.tuning import Tuner


@dataclass
class TrainingConfig:
    warmup_steps: int = 500
    learning_rate: float | None = None
    batch_size: int = 32
    swa_lrs: float = 1e-3
    n_epochs: int = 50


def train_ecgcnn(
    model_config: EcgModelConfig,
    training_config: TrainingConfig,
    parquet_path: str = "./data/mitbih.parquet",
):
    model = LitEcg(
        model_config,
        128,  # 128 samples for now.
        training_config.learning_rate or 1e-3,
        training_config.warmup_steps,
    )
    datamodule = MitBih(
        parquet_path=parquet_path,
        batch_size=training_config.batch_size,
    )
    trainer = L.Trainer(
        max_epochs=training_config.n_epochs,
        callbacks=[
            StochasticWeightAveraging(swa_lrs=training_config.swa_lrs),
            ModelCheckpoint(
                dirpath="./models/",
                filename="{epoch}-{val_loss:.2f}",
                monitor="val_loss",
                save_top_k=5,
                save_on_exception=True,
                every_n_epochs=2,
            ),
        ],
    )

    if training_config.learning_rate is None:
        print("Learning rate not specified in training configuration. Tuning LR.")
        tuner = Tuner(trainer)
        tuner.lr_find(model, datamodule=datamodule)

    trainer.fit(model, datamodule=datamodule)


@dataclass
class HparamTuningConfig:
    min_depth: int = 2
    max_depth: int = 6
    kernel_size1_choices: list[int] = field(default_factory=lambda: [3, 5, 7])
    kernel_size2_choices: list[int] = field(default_factory=lambda: [3, 5, 7])
    model_channels_choices: list[int] = field(default_factory=lambda: [16, 32, 64])
    max_pool_size_choices: list[int] = field(default_factory=lambda: [2, 3, 5])
    n_trials: int = 20


def hparam_search(
    search_config: HparamTuningConfig,
    training_config: TrainingConfig,
    parquet_path: str = "./data/mitbih.parquet",
) -> EcgModelConfig:
    """
    Perform hyperparameter search to find the best model configuration.
    No learning rate tuning here; uses the one from training_config. If training_config.learning_rate is None, then 1e-3 is used as a default.
    """

    if training_config.n_epochs > 10:
        print(
            f"Warning: Hyperparameter search with {training_config.n_epochs} epochs may take a long time. Consider reducing the number of epochs for faster results."
        )

    datamodule = MitBih(
        parquet_path=parquet_path,
        batch_size=training_config.batch_size,
    )

    def objective(trial: optuna.Trial) -> float:
        depth = trial.suggest_int(
            "depth", search_config.min_depth, search_config.max_depth
        )
        kernel_size1 = trial.suggest_categorical(
            "kernel_size1", search_config.kernel_size1_choices
        )
        kernel_size2 = trial.suggest_categorical(
            "kernel_size2", search_config.kernel_size2_choices
        )
        max_pool_size = trial.suggest_categorical(
            "max_pool_size", search_config.max_pool_size_choices
        )
        model_channels = trial.suggest_categorical(
            "model_channels", search_config.model_channels_choices
        )

        model_config = EcgModelConfig(
            depth=depth,
            kernel_size1=kernel_size1,
            kernel_size2=kernel_size2,
            max_pool_size=max_pool_size,
            model_channels=model_channels,
        )

        # Train the model and return the validation loss
        model = LitEcg(
            model_config,
            128,  # Input sequence length
            training_config.learning_rate or 1e-3,
            training_config.warmup_steps,
        )

        trainer = L.Trainer(
            max_epochs=training_config.n_epochs,
            callbacks=[
                ModelCheckpoint(
                    dirpath="./models/",
                    filename="opt-{epoch}-{val_loss:.2f}",
                    monitor="val_loss",
                    save_top_k=1,
                    save_on_exception=True,
                    every_n_epochs=4,
                ),
                StochasticWeightAveraging(swa_lrs=training_config.swa_lrs),
            ],
        )

        trainer.fit(model, datamodule=datamodule)
        return trainer.callback_metrics["val_loss"].item()

    study = optuna.create_study(
        study_name="ecgnn-hps",
        direction="minimize",
    )
    study.optimize(objective, n_trials=search_config.n_trials, catch=(AssertionError,))
    return EcgModelConfig(**study.best_params)
