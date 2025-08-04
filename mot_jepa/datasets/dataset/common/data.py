from dataclasses import dataclass, asdict
from typing import Dict, Optional, Callable
from mot_jepa.common.data import JSON

import torch


@dataclass
class VideoClipPart:
    ids: Optional[torch.Tensor]
    ts: torch.Tensor
    mask: torch.Tensor
    features: Dict[str, torch.Tensor]

    def remove_temporal_dimension(self) -> None:
        assert all(seq.shape[1] == 1 for seq in [self.ids, self.ts, self.mask]), \
            f'Can\'t remove temporal dim unless it has length of 1! {self.ids.shape=}, {self.ts.shape=}, {self.mask.shape=}'
        self.ids = self.ids[:, 0]
        self.ts = self.ts[:, 0]
        self.mask = self.mask[:, 0]
        for key in self.features:
            self.features[key] = self.features[key][:, 0]

    def serialize(self) -> JSON:
        return asdict(self)

    def apply(self, func: Callable[[torch.Tensor], torch.Tensor]) -> None:
        if self.ids is not None:
            self.ids = func(self.ids)
        self.ts = func(self.ts)
        self.mask = func(self.mask)
        self.features = {k: func(v) for k, v in self.features.items()}


@dataclass
class VideoClipData:
    observed: VideoClipPart
    unobserved: VideoClipPart

    def serialize(self) -> JSON:
        return asdict(self)

    def apply(self, func: Callable[[torch.Tensor], torch.Tensor]) -> None:
        self.observed.apply(func)
        self.unobserved.apply(func)
