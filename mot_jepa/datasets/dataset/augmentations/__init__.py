from mot_jepa.datasets.dataset.augmentations.base import (
    Augmentation,
    IdentityAugmentation,
    CompositionAugmentation,
    NonDeterministicAugmentation
)
from mot_jepa.datasets.dataset.augmentations.bbox import BBoxGaussianNoiseAugmentation
from mot_jepa.datasets.dataset.augmentations.video import OcclusionAugmentations, IdentitySwitchAugmentation
