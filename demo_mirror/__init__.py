"""
Demo Mirror
===========

Face tracking + gesture mirroring demo using ainex_api.

- Tracks faces with head movement
- Mirrors arm gestures (raise left/right/both)
- Responds to waving with greeting motion
"""

from .main import DemoMirror, main
from .behavior import BehaviorController, RobotState

__version__ = '2.2.0'
__all__ = ['DemoMirror', 'main', 'BehaviorController', 'RobotState']
