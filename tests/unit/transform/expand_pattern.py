import pytest
from tdlp.datasets.dataset.transform.utils import expand_pattern


def test_examples_from_prompt():
    assert expand_pattern([[2, [0.1, 0.2]], [0.3]]) == [0.1, 0.2, 0.1, 0.2, 0.3]
    assert expand_pattern([0.1, 0.2]) == [0.1, 0.2]


def test_single_pair_top_level():
    assert expand_pattern([2, [1, 2]]) == [1, 2, 1, 2]
    assert expand_pattern([3, [(1, 2)]]) == [(1, 2), (1, 2), (1, 2)]


def test_zero_and_empty():
    assert expand_pattern([[0, [1, 2]], [3]]) == [3]
    assert expand_pattern([[2, []], [3]]) == [3]


def test_negative_repeat_raises():
    with pytest.raises(ValueError):
        expand_pattern([ -1, [1] ])
    with pytest.raises(ValueError):
        expand_pattern([[ -2, [1, 2] ]])


def test_bare_items_once():
    # Bare list appended once
    assert expand_pattern([[1, 2], [3, 4]]) == [1, 2, 3, 4]
    # Bare tuple appended once
    assert expand_pattern([(1, 2), (3, 4)]) == [(1, 2), (3, 4)]


def test_mixed_forms():
    spec = [[2, [1, 2]], [3, 4], 5, [2, (9, 9)], (7, 8)]
    # [3,4] is appended once; (7,8) is atomic element appended once
    assert expand_pattern(spec) == [1, 2, 1, 2, 3, 4, 5, (9, 9), (9, 9), (7, 8)]


def test_strings_are_atomic_not_iterated():
    # Strings are treated as atomic elements (not sequences)
    assert expand_pattern(["ab", "cd"]) == ["ab", "cd"]
    # But they can be repeated if wrapped in a sequence as ITEMS
    assert expand_pattern([2, ["a", "b"]]) == ["a", "b", "a", "b"]


def test_plain_scalar():
    assert expand_pattern(5) == [5]
    assert expand_pattern((1, 2)) == [(1, 2)]  # tuple as a single element stays atomic


def test_idempotent_for_flat_scalars_and_tuples():
    flat = [1, 2, (3, 4), "x", 5.0]
    assert expand_pattern(flat) == flat