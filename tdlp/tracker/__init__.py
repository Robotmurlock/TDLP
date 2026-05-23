"""
Tracker interface. Trackers are implemented using motrack library.
Source: https://github.com/Robotmurlock/Motrack/
"""
from tdlp.tracker.online import TDLPOnlineTracker
from tdlp.tracker.offline import TDLPOfflineTracker
from tdlp.tracker import adapter  # noqa: F401 — registers 'tdlp' in TRACKER_CATALOG

__all__ = [
    'TDLPOnlineTracker',
    'TDLPOfflineTracker',
]
