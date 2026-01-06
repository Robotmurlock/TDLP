"""
Tracker interface. Trackers are implemented using motrack library.
Source: https://github.com/Robotmurlock/Motrack/
"""
from tdlp.tracker.online import TDLPOnlineTracker
from tdlp.tracker.offline import TDLPOfflineTracker

__all__ = [
    'TDLPOnlineTracker',
    'TDLPOfflineTracker',
]
