"""
Registers TDLP's MotrackDatasetWrapper as a regular Motrack dataset type,
so motrack.datasets.dataset_factory('tdlp_dancetrack', ...) works.
"""
from typing import List, Optional

from motrack.datasets.catalog import DATASET_CATALOG

from tdlp.datasets.dataset import dataset_index_factory
from tdlp.datasets.dataset.motrack import MotrackDatasetWrapper


@DATASET_CATALOG.register('tdlp_dancetrack')
def _build_tdlp_dancetrack(
    path: str,                     # noqa: ARG001  (motrack-derived, ignored: TDLP uses index_params.paths)
    test: bool,
    index_params: dict,
    sequence_list: Optional[List[str]] = None,
    split: str = 'val',
):
    """Build a TDLP-backed dataset wrapped to look like a Motrack dataset.

    Motrack's ``dataset_factory`` injects ``path`` and ``test`` into ``params``
    and unpacks the whole dict as kwargs. ``path`` is the motrack-derived
    fullpath (``<assets>/<dataset.path>/<split>``), but TDLP's MOT index needs
    its own ``paths`` list (parent dir, no split suffix), so we ignore ``path``
    and use ``index_params`` from the yaml.
    """
    actual_split = 'test' if test else split
    dataset_index = dataset_index_factory(
        name='mot',
        params=index_params,
        split=actual_split,
        sequence_list=sequence_list,
    )
    return MotrackDatasetWrapper(dataset_index)
