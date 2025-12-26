from collections import defaultdict
from typing import Callable, List, Dict, TypeVar

T = TypeVar('T')
K = TypeVar('K')

def group_by(items: List[T], key_func: Callable[[T], K]) -> Dict[K, List[T]]:
    """
    Groups items in a list based on a mapping function.

    Args:
        items: The list of items to group.
        key_func: A function that maps an item to its group key.

    Returns:
        Dict[K, List[T]]: A dictionary where keys are group identifiers, and values are lists of items.
    """
    grouped = defaultdict(list)
    for item in items:
        grouped[key_func(item)].append(item)
    return dict(grouped)
