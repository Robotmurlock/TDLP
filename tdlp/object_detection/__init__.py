"""
Custom object detection implementations registered to the motrack catalog.
Importing this package registers all detectors so they are available via DetectionManager.
"""
from tdlp.object_detection import mmdet_yolox  # noqa: F401
