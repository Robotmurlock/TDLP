"""Shared JSON type alias used across the project."""
from typing import Any, Dict, List, Union


JSON = Union[str, int, float, bool, None, Dict[str, Any], List[Any]]
