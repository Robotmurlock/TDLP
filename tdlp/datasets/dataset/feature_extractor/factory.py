from typing import Dict, Any

from tdlp.datasets.dataset.feature_extractor.feature_extractor import FeatureExtractor
from tdlp.datasets.dataset.feature_extractor.gt_bbox_feature_extractor import GTBBoxFeatureExtractor
from tdlp.datasets.dataset.feature_extractor.pred_bbox_feature_extractor import PredictionBBoxFeatureExtractor
from tdlp.datasets.dataset.index.index import DatasetIndex

FEATURE_EXTRACTOR_CATALOG = {
    'gt_bbox': GTBBoxFeatureExtractor,
    'pred_bbox': PredictionBBoxFeatureExtractor
}


def feature_extractor_factory(
    extractor_type: str,
    extractor_params: Dict[str, Any],
    index: DatasetIndex,
    n_tracks: int,
    object_id_mapping: Dict[str, int]
) -> FeatureExtractor:
    assert extractor_type in FEATURE_EXTRACTOR_CATALOG, \
        f'Unknown feature extractor type "{extractor_type}". Available: {list(FEATURE_EXTRACTOR_CATALOG.keys())}'

    return FEATURE_EXTRACTOR_CATALOG[extractor_type](
        index=index,
        n_tracks=n_tracks,
        object_id_mapping=object_id_mapping,
        **extractor_params
    )
