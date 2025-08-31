"""
Implement the extracted MIT-BIH dataset as a lightning `DataModule`.
"""

import lightning as L
import polars as pl
import torch as tch
from torch.utils.data.dataloader import DataLoader


class MitBih(L.LightningDataModule):
    def __init__(
        self,
        parquet_path: str = "data/mitbih.parquet",
        batch_size: int = 8,
    ) -> None:
        """ """
        super().__init__()
        self.batch_size = batch_size
        self.file_path = parquet_path

    def setup(self, stage: str) -> None:
        self.df = pl.read_parquet(self.file_path).with_columns(
            pl.int_range(0, pl.len()).alias("id")
        )
        train_df = self.df.sample(fraction=0.7, shuffle=True)
        validation_df = self.df.join(train_df, on=["id"], how="anti")
        self.train_dataset = PolarsDataset(train_df, "aami5", ["signal_a"])
        self.validation_dataset = PolarsDataset(validation_df, "aami5", ["signal_a"])
        return super().setup(stage)

    def _collate(self, x):
        xf, t = x
        xf: tch.Tensor = xf.to(tch.float32)
        return (xf.unsqueeze(1), t)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self._collate,  # no need for collate.
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            collate_fn=self._collate,  # no need for collate
        )


class PolarsDataset(tch.utils.data.Dataset):
    def __init__(self, df: pl.DataFrame, target_col: str, features: list[str]) -> None:
        super().__init__()
        # Drop any `id`s
        self.df = df.drop("id") if "id" in df.columns else df
        self.features = features
        self.target_col = target_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[tch.Tensor, int]:
        row = self.df[idx]
        features = row.select(self.features).to_torch()
        target = int(row[self.target_col][0])

        return features, target

    def __getitems__(self, indices: list[int]) -> tuple[tch.Tensor, tch.Tensor]:
        rows = self.df[indices]
        features = rows.select(self.features).to_torch()
        target = rows.select(self.target_col).to_torch()

        return features, target.squeeze(1)
