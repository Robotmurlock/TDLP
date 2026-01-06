"""Factory for dataset index implementations."""
from typing import List, Optional

from tdlp.datasets.dataset.index.index import DatasetIndex
from tdlp.datasets.dataset.index.mot import MOTDatasetIndex

DATASET_INDEX_CATALOG = {
    'mot': MOTDatasetIndex
}


def dataset_index_factory(
    name: str,
    params: dict,
    split: str,
    sequence_list: Optional[List[str]] = None
) -> DatasetIndex:
    """
    Factory for dataset index implementations.

    Args:
        name: Name of the dataset index implementation.
        params: Parameters for the dataset index implementation.
        split: Split of the dataset.
        sequence_list: Sequence list of the dataset.

    Returns:
        Dataset index implementation.
    """
    name = name.lower()
    return DATASET_INDEX_CATALOG[name](
        **params,
        split=split,
        sequence_list=sequence_list
    )
