"""
Config structure. Config should be loaded as dictionary and parsed into GlobalConfig Python object. Benefits:
- Structure and type validation (using dacite library)
- Custom validations
- Python IDE autocomplete
"""
import logging
from dataclasses import field
from typing import Optional, List, Iterator

from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pydantic.dataclasses import dataclass
from torch import nn
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

from mot_jepa.common import project, conventions
from mot_jepa.datasets.dataset import DatasetIndex, MOTClipDataset

logger = logging.getLogger('ConfigParser')


@dataclass
class DatasetIndexConfig:
    type: str
    params: dict
    sequence_list: Optional[List[str]] = None


@dataclass
class DatasetReIDConfig:
    path: str
    n_pairs: int
    n_other: int


@dataclass
class DatasetConfig:
    index: DatasetIndexConfig
    n_tracks: int
    clip_length: int
    min_clip_tracks: int = 1
    clip_sampling_step: int = 1
    val_clip_sampling_step: Optional[int] = None

    # ReID pre-train
    reid: Optional[DatasetReIDConfig] = None

    def __post_init__(self) -> None:
        """
        Postprocess and validation.
        """
        if self.val_clip_sampling_step is None:
            self.val_clip_sampling_step = self.clip_sampling_step

    def build_dataset(self, index: DatasetIndex) -> MOTClipDataset:
        test = index.split in ['val', 'test']
        clip_sampling_step = self.val_clip_sampling_step if test else self.clip_sampling_step

        return MOTClipDataset(
            index=index,
            n_tracks=self.n_tracks,
            clip_length=self.clip_length,
            clip_sampling_step=clip_sampling_step,
            use_augmentation=not test,
            test=test
        )


@dataclass
class ModelConfig:
    params: dict


@dataclass
class ResourcesConfig:
    batch_size: int
    accelerator: str
    num_workers: int


@dataclass
class TrainCheckpointConfig:
    metric_monitor: str = 'val/loss'
    resume_from: Optional[str] = None


@dataclass
class TrainConfig:
    max_epochs: int
    optimizer_config: dict
    scheduler_config: dict

    gradient_clip: Optional[float] = None

    resume: bool = False
    truncate: bool = False
    checkpoint_cfg: TrainCheckpointConfig = field(default_factory=TrainCheckpointConfig)

    def build_optimizer(self, params: Iterator[nn.Parameter]) -> Optimizer:
        return instantiate(OmegaConf.create(self.optimizer_config), params=params)

    def build_scheduler(self, optimizer: Optimizer, epoch_steps: int) -> LRScheduler:
        return instantiate(
            OmegaConf.create(self.scheduler_config),
            optimizer=optimizer,
            epoch_steps=epoch_steps,
            epochs=self.max_epochs
        )


@dataclass
class EvalConfig:
    split: str
    batch_size: Optional[int] = None
    checkpoint: Optional[str] = None


@dataclass
class PathConfig:
    master: str

    @classmethod
    def default(cls) -> 'PathConfig':
        """
        Default path configuration is used if it is not defined in configs.

        Returns: Path configuration.
        """
        # noinspection PyArgumentList
        return cls(
            master=project.OUTPUTS_PATH
        )


@dataclass
class GlobalConfig:
    """
    Scripts GlobalConfig
    """
    experiment_name: str
    dataset_name: str
    resources: ResourcesConfig
    dataset: DatasetConfig
    train: TrainConfig
    eval: EvalConfig
    model: ModelConfig

    # noinspection PyUnresolvedReferences
    path: PathConfig = field(default_factory=PathConfig.default)

    def __post_init__(self):
        if self.eval.checkpoint is None:
            experiment_path = conventions.get_experiment_path(self.path.master, self.dataset_name, self.experiment_name)
            self.eval.checkpoint = conventions.get_latest_checkpoint_path(experiment_path)

        if self.eval.batch_size is None:
            # No override
            self.eval.batch_size = self.resources.batch_size

# Configuring hydra config store
# If config has `- global_config` in defaults then
# full config is recursively instantiated
cs = ConfigStore.instance()
cs.store(name='global_config', node=GlobalConfig)
