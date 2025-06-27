from typing import Optional, List

from mot_jepa.datasets.dataset.index.index import DatasetIndex
from mot_jepa.datasets.dataset.index.mot import MOTDatasetIndex

DATASET_INDEX_CATALOG = {
    'mot': MOTDatasetIndex
}


def dataset_index_factory(
    name: str,
    params: dict,
    split: str,
    sequence_list: Optional[List[str]] = None
) -> DatasetIndex:
    name = name.lower()
    return DATASET_INDEX_CATALOG[name](
        **params,
        split=split,
        sequence_list=sequence_list
    )
