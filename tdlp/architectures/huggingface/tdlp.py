from typing import Any, Dict, Optional, Tuple

from huggingface_hub import PyTorchModelHubMixin
import torch
from torch import nn

from tdlp.architectures.tdlp.core import build_mm_tdsp_model
from tdlp.huggingface.constants import DOCS_URL, LICENSE, PAPER_URL, REPO_URL


class MultiModalTDLPHuggingFaceModel(
    nn.Module,
    PyTorchModelHubMixin,
    repo_url=REPO_URL,
    license=LICENSE,
    paper_url=PAPER_URL,
    docs_url=DOCS_URL,
    tags=["TDLP"]
):
    def __init__(
        self,
        per_feature_params: Dict[str, Any],
        common_params: Dict[str, Any],
        sph_per_feature_params: Dict[str, Any],
        sph_common_params: Dict[str, Any],
        mm_dim: int,
        aggregator_type: str,
        aggregator_params: Dict[str, Any],
        similarity_prediction_head_hidden_dim: int = 256,
        object_interaction_encoder_enable: bool = False,
        object_interaction_encoder_params: Optional[Dict[str, Any]] = None
    ):
        super().__init__()

        self._model = build_mm_tdsp_model(
            per_feature_params=per_feature_params,
            common_params=common_params,
            sph_per_feature_params=sph_per_feature_params,
            sph_common_params=sph_common_params,
            mm_dim=mm_dim,
            aggregator_type=aggregator_type,
            aggregator_params=aggregator_params,
            similarity_prediction_head_hidden_dim=similarity_prediction_head_hidden_dim,
            object_interaction_encoder_enable=object_interaction_encoder_enable,
            object_interaction_encoder_params=object_interaction_encoder_params,
        )

    @property
    def model(self):
        """Get the model."""
        return self._model

    def forward(
        self,
        track_features: Dict[str, torch.Tensor],
        track_mask: torch.Tensor,
        det_features: Dict[str, torch.Tensor],
        det_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return self._model(track_features, track_mask, det_features, det_mask)



HF_MODEL_IMPORT_ALIASES = {
    "tdlp.architectures.tdlp.core.build_mm_tdsp_model": "tdlp.architectures.huggingface.tdlp.MultiModalTDLPHuggingFaceModel"
}
