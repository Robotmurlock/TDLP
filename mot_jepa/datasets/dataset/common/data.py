from dataclasses import dataclass, asdict
import torch
from typing import Dict


@dataclass
class VideoClipData:
    observed_bboxes: torch.Tensor
    observed_ts: torch.Tensor
    observed_temporal_mask: torch.Tensor
    unobserved_bboxes: torch.Tensor
    unobserved_ts: torch.Tensor
    unobserved_temporal_mask: torch.Tensor

    def serialize(self) -> Dict[str, torch.Tensor]:
        return asdict(self)
