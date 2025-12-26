from collections.abc import Sequence
from typing import Any, List

def expand_pattern(spec: Any) -> List[Any]:
    """
    Expand a shorthand like:
      - [[N1, ITEMS1], [N2, ITEMS2], ..., ITEMS_K, elem, ...] -> flat list
    Rules:
      - A pair [N, ITEMS] repeats the sequence ITEMS N times.
      - ITEMS is a (list/tuple) of elements; elements can be scalars or tuples (tuples are atomic).
      - If an entry is just ITEMS (no N), it's appended once (no repetition).
      - If the entire `spec` is a single pair [N, ITEMS], that’s supported too.
      - A plain non-sequence value becomes a single-element list.

    Examples:
      expand_spec([[2, [0.1, 0.2]], [0.3]]) -> [0.1, 0.2, 0.1, 0.2, 0.3]
      expand_spec([0.1, 0.2])               -> [0.1, 0.2]
      expand_spec([2, [1, 2]])              -> [1, 2, 1, 2]
      expand_spec([[3, [(1,2)]], [3]])      -> [(1,2), (1,2), (1,2), 3]
    """
    def is_seq(x: Any) -> bool:
        return isinstance(x, Sequence) and not isinstance(x, (str, bytes))

    # If spec itself is a single [N, ITEMS] pair, normalize to a one-entry list
    if is_seq(spec) and len(spec) == 2 and isinstance(spec[0], int) and is_seq(spec[1]):
        entries = [spec]
    elif is_seq(spec):
        entries = list(spec)
    else:
        # Plain value -> single-element list
        return [spec]

    out: List[Any] = []
    for entry in entries:
        if is_seq(entry) and len(entry) == 2 and isinstance(entry[0], int) and is_seq(entry[1]):
            # Repeat pair [N, ITEMS]
            n = entry[0]
            if n < 0:
                raise ValueError("Repeat count must be >= 0")
            items = entry[1]
            for _ in range(n):
                for item in items:
                    out.append(item)
        elif is_seq(entry):
            # Bare ITEMS (no repeat): append once
            for item in entry:
                out.append(item)
        else:
            # Single element
            out.append(entry)
    return out
