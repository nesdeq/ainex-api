"""
AINEX Bare Metal Python API
============================

Pure Python API for controlling the AINEX humanoid robot without ROS.
Communicates directly with the STM32 controller via serial.

Usage:
    from ainex_api import Robot
    robot = Robot()
    robot.stand()
    robot.head.move(pan=500, tilt=500)
    robot.motion.play('greet')
    robot.vision.start()
    face = robot.vision.get_face()
"""

from .board import Board
from .servos import ServoController
from .head import HeadController
from .motion import MotionPlayer
from .sensors import SensorReader
from .peripherals import Peripherals
from .camera import Camera, open_camera
from .vision import VisionSystem, FaceData, GestureData, FaceAnalyzer, GestureAnalyzer
from .robot import Robot
from .remote_vision import RemoteVisionClient, RemoteVisionServer

__version__ = '2.2.0'
__all__ = ['Board', 'ServoController', 'HeadController', 'MotionPlayer',
           'SensorReader', 'Peripherals', 'Camera', 'open_camera', 'VisionSystem',
           'FaceData', 'GestureData', 'FaceAnalyzer', 'GestureAnalyzer',
           'RemoteVisionClient', 'RemoteVisionServer', 'Robot']
