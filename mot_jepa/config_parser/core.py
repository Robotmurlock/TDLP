"""
Config structure. Config should be loaded as dictionary and parsed into GlobalConfig Python object. Benefits:
- Structure and type validation (using the dacite library)
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
from torch.utils.data import Sampler

from mot_jepa.common import project, conventions
from mot_jepa.datasets.dataset import DatasetIndex, MOTClipDataset
from mot_jepa.datasets.dataset.augmentations import Augmentation, IdentityAugmentation
from mot_jepa.datasets.dataset.transform import Transform, IdentityTransform

logger = logging.getLogger('ConfigParser')


@dataclass
class DatasetIndexConfig:
    type: str
    params: dict
    sequence_list: Optional[List[str]] = None


@dataclass
class FeatureExtractorConfig:
    extractor_type: str
    extractor_params: dict


@dataclass
class DatasetConfig:
    index: DatasetIndexConfig
    n_tracks: int
    clip_length: int
    min_clip_tracks: int = 1
    clip_sampling_step: int = 1
    val_clip_sampling_step: Optional[int] = None

    feature_extractor: Optional[FeatureExtractorConfig] = None
    transform: Optional[dict] = None
    augmentations: Optional[dict] = None
    sampler: Optional[dict] = None
    use_batch_sampler: bool = False

    def __post_init__(self) -> None:
        """
        Postprocess and validation.
        """
        if self.val_clip_sampling_step is None:
            self.val_clip_sampling_step = self.clip_sampling_step

    def build_transform(self, disable_transform: bool = False) -> Transform:
        return IdentityTransform() if disable_transform or self.transform is None else instantiate(self.transform)

    def build_augmentations(self, disable_augmentations: bool = False) -> Augmentation:
        return IdentityAugmentation() if disable_augmentations or self.augmentations is None else instantiate(self.augmentations)

    def build_dataset(
        self,
        index: DatasetIndex,
        disable_transform: bool = False,
        disable_augmentations: bool = False
    ) -> MOTClipDataset:
        test = index.split in ['val', 'test']
        clip_sampling_step = self.val_clip_sampling_step if test else self.clip_sampling_step
        disable_augmentations = disable_augmentations or test

        return MOTClipDataset(
            index=index,
            n_tracks=self.n_tracks,
            clip_length=self.clip_length,
            clip_sampling_step=clip_sampling_step,
            transform=self.build_transform(disable_transform=disable_transform),
            augmentations=self.build_augmentations(disable_augmentations=disable_augmentations),
            feature_extractor_type=self.feature_extractor.extractor_type if self.feature_extractor is not None else None,
            feature_extractor_params=self.feature_extractor.extractor_params if self.feature_extractor is not None else None
        )

    def build_sampler(self, dataset: MOTClipDataset) -> Optional[Sampler]:
        if self.sampler is None:
            return None

        # noinspection PyUnresolvedReferences
        if self.sampler['_target_'].endswith('.from_dataset'):
            self.sampler['dataset'] = dataset

        return instantiate(self.sampler)


@dataclass
class ResourcesConfig:
    batch_size: int
    accelerator: str
    num_workers: int

    val_batch_size: Optional[int] = None

    def __post_init__(self) -> None:
        self.val_batch_size = self.batch_size if self.val_batch_size is None else self.val_batch_size


@dataclass
class TrainCheckpointConfig:
    metric_monitor: str = 'val-epoch/loss'
    resume_from: Optional[str] = None


@dataclass
class TrainConfig:
    max_epochs: int

    loss_config: dict
    optimizer_config: dict
    scheduler_config: dict

    gradient_clip: Optional[float] = None
    mixed_precision: bool = False

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

    def build_loss_func(self):
        return instantiate(
            OmegaConf.create(self.loss_config),
        )


@dataclass
class EvalObjectDetectionConfig:
    type: str
    params: dict
    lookup_path: Optional[str] = None
    cache_path: Optional[str] = None
    oracle: bool = False


@dataclass
class EvalTrackerConfig:
    remember_threshold: int
    detection_threshold: float
    new_tracklet_detection_threshold: float
    initialization_threshold: int
    sim_threshold: float


@dataclass
class EvalPostprocessConfig:
    init_threshold: int = 2  # Activate `init_threshold` starting bboxes
    linear_interpolation_threshold: int = 3  # Maximum distance to perform linear interpolation
    linear_interpolation_min_tracklet_length: int = 30  # Minimum tracklet length to perform linear interpolation
    min_tracklet_length: int = 20  # Remove all tracklets that are shorter than this


@dataclass
class EvalConfig:
    object_detection: EvalObjectDetectionConfig
    tracker: Optional[EvalTrackerConfig] = None
    split: str = 'val'
    checkpoint: Optional[str] = None
    visualize: bool = False
    postprocess_enable: bool = True
    postprocess: EvalPostprocessConfig = field(default_factory=EvalPostprocessConfig)


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
    model_config: dict

    # noinspection PyUnresolvedReferences
    path: PathConfig = field(default_factory=PathConfig.default)
    eval: Optional[EvalConfig] = None
    analysis: Optional[dict] = None

    def __post_init__(self):
        if self.eval.checkpoint is None:
            experiment_path = conventions.get_experiment_path(self.path.master, self.dataset_name, self.experiment_name)
            self.eval.checkpoint = conventions.get_latest_checkpoint_path(experiment_path)

    def build_model(self):
        return instantiate(
            OmegaConf.create(self.model_config),
        )


# Configuring hydra config store
# If config has `- the_global_config` in defaults, then
# full config is recursively instantiated
cs = ConfigStore.instance()
cs.store(name='the_global_config', node=GlobalConfig)

