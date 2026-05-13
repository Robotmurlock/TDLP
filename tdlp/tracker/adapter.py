"""
Adapter that registers the TDLP tracker in Motrack's TRACKER_CATALOG.
Loads model + transform from config in __init__, so the tracker is
fully self-contained and instantiable via motrack.tracker.tracker_factory.
"""
from typing import Any, Dict, Optional

import torch
from hydra.utils import instantiate
from motrack.tracker.trackers.catalog import TRACKER_CATALOG
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field
from torch import nn

from tdlp.datasets.dataset.transform import IdentityTransform, Transform
from tdlp.tracker.online import TDLPOnlineTracker

# Module-level model cache — keyed by checkpoint path.
# Required for optimization: without this, a 100-trial run reloads the
# checkpoint from disk 100 times.
_MODEL_CACHE: Dict[str, nn.Module] = {}


@TRACKER_CATALOG.register_config('tdlp')
class TDLPTrackerConfig(BaseModel):
    """
    Config schema for the TDLP tracker.

    NOTE: The architecture-config field is named ``architecture`` (not
    ``model_config``) because Pydantic v2 reserves ``model_config`` as
    the class-level ConfigDict slot (see ``model_config = ConfigDict(...)``
    below).
    """

    # Pydantic v2 class-level config — not a regular field.
    model_config = ConfigDict(extra='forbid')

    # --- Fixed: model / transform / device (NOT optimizable) ---
    architecture: Dict[str, Any]                # Hydra-instantiatable; has _target_
    checkpoint: str
    device: str = 'cpu'
    transform: Optional[Dict[str, Any]] = None  # Hydra-instantiatable; None → Identity

    # --- Optimizable: tracker hyperparams ---
    detection_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    sim_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    initialization_threshold: int = Field(default=1, ge=0)
    remember_threshold: int = Field(default=30, ge=1)
    new_tracklet_detection_threshold: float = Field(default=0.9, ge=0.0, le=1.0)
    use_conf: bool = True

    # ---- Builders ---------------------------------------------------------
    def build_model(self) -> nn.Module:
        """Build and load the TDLP model. Cached by checkpoint path."""
        if self.checkpoint in _MODEL_CACHE:
            model = _MODEL_CACHE[self.checkpoint]
        else:
            model = instantiate(OmegaConf.create(self.architecture))
            state_dict = torch.load(self.checkpoint, map_location='cpu')
            model.load_state_dict(state_dict['model'])
            _MODEL_CACHE[self.checkpoint] = model
        # Move to configured device for this trial. Cheap when already there;
        # correct after a previous trial released it to CPU in __del__.
        model.to(self.device)
        return model

    def build_transform(self) -> Transform:
        """Build the TDLP feature transform."""
        if self.transform is None:
            return IdentityTransform()
        return instantiate(OmegaConf.create(self.transform))


@TRACKER_CATALOG.register('tdlp')
class TDLPAdapterTracker(TDLPOnlineTracker):
    """Self-contained TDLP tracker; loads model/transform from config."""

    def __init__(self, config: TDLPTrackerConfig):
        self._cached_checkpoint = config.checkpoint  # for __del__
        model = config.build_model()
        transform = config.build_transform()

        super().__init__(
            transform=transform,
            model=model,
            device=config.device,
            detection_threshold=config.detection_threshold,
            sim_threshold=config.sim_threshold,
            initialization_threshold=config.initialization_threshold,
            remember_threshold=config.remember_threshold,
            new_tracklet_detection_threshold=config.new_tracklet_detection_threshold,
            use_conf=config.use_conf,
        )

    def __del__(self):
        # When this trial's tracker is GC'd, push the cached model back to CPU
        # so GPU memory is freed for the next trial. The next trial's
        # build_model will move it back to the configured device.
        try:
            cached = _MODEL_CACHE.get(self._cached_checkpoint)
            if cached is not None:
                cached.to('cpu')
        except Exception:
            # __del__ must not raise during interpreter shutdown
            pass
