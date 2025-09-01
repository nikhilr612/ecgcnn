"""
Implement a 1D CNN-based classifier.
"""

from dataclasses import asdict, dataclass
from math import floor
from typing import OrderedDict
from lightning.pytorch.utilities.types import OptimizerLRScheduler
import torch as tch
import torch.nn as nn
import lightning as L


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        k1: int = 3,  # kernel-1 size
        k2: int = 3,  # kernel-2 size
        st1: int = 1,  # stride-1; paper gives 2 - doesn't make sense.
        st2: int = 2,  # stride-2; down-sample.
        pd1: int = 1,  # padding-1
        pd2: int = 1,  # padding-2
        kmp: int = 5,  # Max-pool size
        out_channels: int = 32,
    ):
        nn.Module.__init__(self)
        self.k1 = k1
        self.k2 = k2
        self.st1 = st1
        self.st2 = st2
        self.pd1 = pd1
        self.pd2 = pd2
        self.kmp = kmp
        self.upper_half = nn.Sequential(
            OrderedDict(
                [
                    (
                        "up_conv_1",
                        nn.Conv1d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=k1,
                            stride=st1,
                            padding=pd1,
                        ),
                    ),
                    ("up_relu_1", nn.ReLU()),
                    ("up_batch_norm_1", nn.BatchNorm1d(out_channels)),
                    (
                        "up_conv_2",
                        nn.Conv1d(
                            in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=k2,
                            stride=st1,
                            padding=pd2,
                        ),
                    ),
                    ("up_relu_2", nn.ReLU()),
                    ("up_batch_norm_2", nn.BatchNorm1d(out_channels)),
                ]
            )
        )  # .cuda()
        self.lower_half = nn.Sequential(
            OrderedDict(
                [
                    ("lo_relu_1", nn.ReLU()),
                    ("lo_batch_norm_1", nn.BatchNorm1d(out_channels)),
                    (
                        "lo_maxpool",
                        nn.MaxPool1d(kernel_size=kmp, stride=st2),
                    ),
                ]
            )
        )  # .cuda()

    def forward(self, x):
        tx = x + self.upper_half(x)
        return self.lower_half(tx)

    def _conv_output_calc(self, l_in: int, k: int, stride: int, padding: int) -> int:
        return int(floor((l_in + 2 * padding - k) / stride + 1))

    def output_seq_len(self, input_seq_len: int) -> int:
        i1 = self._conv_output_calc(input_seq_len, self.k1, self.st1, self.pd1)
        i2 = self._conv_output_calc(i1, self.k2, self.st1, self.pd2)
        r = (i2 - self.kmp) // self.st2 + 1
        print(f"{input_seq_len} -> {i1} -> {i2} -> {r}")
        return r


class MLPHead(nn.Module):
    def __init__(self, in_channels: int, in_seq_len: int, out_featues: int = 6) -> None:
        nn.Module.__init__(self)
        self.input_dim = in_channels * in_seq_len
        self.layers = nn.Sequential(
            OrderedDict(
                [
                    (
                        "head_mlp_1",
                        nn.Linear(
                            in_features=self.input_dim, out_features=self.input_dim
                        ),
                    ),
                    ("head_relu_1", nn.ReLU()),
                    (
                        "final_mlp",
                        nn.Linear(in_features=self.input_dim, out_features=out_featues),
                    ),
                ]
            )
        )

    def forward(self, x) -> tch.Tensor:
        xf = tch.flatten(x, start_dim=1)
        logits = self.layers(xf)
        return logits


@dataclass
class EcgModelConfig:
    depth: int = 2  # Number of additional ConvBlocks to stack.
    kernel_size1: int = (
        3  # The size of the kernels in the first convolutional layer in a block.
    )
    kernel_size2: int = (
        3  # The size of the kernels in the second convolutional layer in a block.
    )
    max_pool_size: int = 5  # ...
    model_channels: int = 32  # The number of output channels of each ConvBlock.


class LitEcg(L.LightningModule):
    def __init__(
        self,
        config: EcgModelConfig,
        input_seq_len: int,
        learning_rate: float,
        warmup_steps: float,
    ) -> None:
        L.LightningModule.__init__(self)
        self.save_hyperparameters(asdict(config))
        self.learning_rate = learning_rate
        self.warmup_steps = max(1, warmup_steps)
        self.mouth = ConvBlock(
            in_channels=1,
            out_channels=config.model_channels,
            k1=config.kernel_size1,
            k2=config.kernel_size2,
            kmp=config.max_pool_size,
        )
        self.x_conv_blocks = nn.ModuleList(
            [
                ConvBlock(
                    in_channels=config.model_channels,
                    out_channels=config.model_channels,
                    k1=config.kernel_size1,
                    k2=config.kernel_size2,
                    kmp=config.max_pool_size,
                )
                for i in range(config.depth)
            ]
        )
        # this is where linen's automatic shaping would've helped.
        seq_len = self.mouth.output_seq_len(input_seq_len)
        for block in self.x_conv_blocks:
            print(f"seq_len={seq_len}")
            seq_len = block.output_seq_len(seq_len)  # type: ignore
        assert seq_len >= 1, (
            f"Sequence Length cannot be 0 or negative. seq_len={seq_len}."
        )
        self.final_layer = MLPHead(config.model_channels, seq_len, out_featues=6)

    def forward(self, x: tch.Tensor) -> tch.Tensor:
        x_in = self.mouth(x)
        for block in self.x_conv_blocks:
            x_in = block(x_in)  # note: no res connection here.
        logits = self.final_layer(x_in)
        return logits

    def training_step(self, batch: tuple[tch.Tensor, tch.Tensor]):
        x, target = batch
        logits: tch.Tensor = self(x)
        train_loss = tch.nn.functional.cross_entropy(logits, target)
        self.log_dict({"train_loss": train_loss})
        return train_loss

    def validation_step(self, batch: tuple[tch.Tensor, tch.Tensor]):
        x, target = batch
        validation_loss = tch.nn.functional.cross_entropy(
            self(x),
            target,
        )
        self.log_dict({"val_loss": validation_loss})
        return validation_loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optim = tch.optim.AdamW(self.parameters(), lr=self.learning_rate)

        def linear_warmup(step: int) -> float:
            if step < self.warmup_steps:
                return float(step) / float(self.warmup_steps)
            return 1.0

        scheduler = tch.optim.lr_scheduler.LambdaLR(optim, linear_warmup)
        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
