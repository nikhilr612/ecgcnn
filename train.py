"""
Script to train the ECG CNN model.
"""

from dataclasses import dataclass
from lightning.pytorch.callbacks import ModelCheckpoint, StochasticWeightAveraging
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
