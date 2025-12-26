"""
Scene basic information.
"""
from dataclasses import dataclass
from typing import Union


@dataclass
class BasicSceneInfo:
    """
    Basic scene information
    """
    name: str
    category: str
    seqlength: Union[int, str]  # Can be parsed from `str`
    imheight: Union[int, str]
    imwidth: Union[int, str]

    def __post_init__(self):
        """
        Convert to proper type.
        """
        self.seqlength = int(self.seqlength)
        self.imheight = int(self.imheight)
        self.imwidth = int(self.imwidth)
