from lightning.pytorch import LightningDataModule, LightningModule
from lightning.pytorch.profilers import PassThroughProfiler

from ...util.typing_utils import mixin_base_type


class ProfilerMixin(mixin_base_type(LightningModule)):
    @property
    def profiler(self):
        if not isinstance(self, (LightningModule, LightningDataModule)):
            raise TypeError(
                "`profiler` can only be used on LightningModule or LightningDataModule"
            )

        if (trainer := self.trainer) is None:
            raise RuntimeError("trainer is not defined")

        if not hasattr(trainer, "profiler"):
            raise RuntimeError("trainer does not have profiler")

        if (profiler := getattr(trainer, "profiler")) is None:
            profiler = PassThroughProfiler()

        return profiler
