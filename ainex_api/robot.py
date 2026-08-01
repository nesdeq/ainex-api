"""
Robot - High Level API
======================

Main entry point combining all subsystems.
"""

import time
from typing import Optional
from .board import Board
from .servos import ServoController, SERVO_CENTER, BODY_SERVO_IDS
from .head import HeadController
from .motion import MotionPlayer, MOTION_STOP_TIMEOUT_S
from .sensors import SensorReader
from .peripherals import Peripherals
from .camera import Camera
from .vision import VisionSystem


# Arm poses shared by the raise/lower methods (servo_id: position)
_LEFT_ARM_NEUTRAL = {13: 835, 15: 830, 17: 500, 19: 150}
_LEFT_ARM_RAISED = {13: 360, 15: 830, 17: 500, 19: 40}
_RIGHT_ARM_NEUTRAL = {14: 165, 16: 170, 18: 500, 20: 850}
_RIGHT_ARM_RAISED = {14: 640, 16: 165, 18: 500, 20: 960}

_MIN_PORT, _MAX_PORT = 1, 65535


def _parse_vision_address(address: str) -> tuple:
    """Split a remote vision "host:port" string, rejecting anything malformed."""
    host, _, port = address.partition(':')
    if not host or not port:
        raise ValueError(f"remote_vision={address!r} is not in host:port form")
    if not port.isdigit():
        raise ValueError(f"remote_vision port {port!r} is not a number")
    port = int(port)
    if not _MIN_PORT <= port <= _MAX_PORT:
        raise ValueError(f"remote_vision port {port} outside {_MIN_PORT}-{_MAX_PORT}")
    return host, port


class Robot:
    """
    High-level AINEX robot API.

    Combines all subsystems into a unified interface.

    Usage:
        robot = Robot()
        robot.stand()
        robot.head.center()
        robot.motion.play('greet')
        robot.vision.start()
        face = robot.vision.get_face()
    """

    def __init__(self, device: str = "/dev/ttyAMA0",
                 undistort_camera: bool = False, distortion_k1: float = -0.15,
                 remote_vision: Optional[str] = None):
        """
        Initialize robot.

        Args:
            device: Serial device path (default: /dev/ttyAMA0 for Pi GPIO UART)
            undistort_camera: Enable lens distortion correction
            distortion_k1: Radial distortion coefficient (negative=barrel, positive=pincushion)
            remote_vision: Remote vision server address "host:port" (e.g., "192.168.1.100:9999").
                          If set, camera frames are sent to remote PC for processing.
                          Run server on PC with: python -m ainex_api.remote_vision --port 9999
        """
        # Parsed before any hardware is opened, so a malformed address cannot
        # leave an open serial port and a live receiver thread behind.
        host, port = _parse_vision_address(remote_vision) if remote_vision else (None, None)

        self.board = Board(device=device)

        # Anything that fails past this point must still release the port.
        try:
            self.servos = ServoController(self.board)
            self.head = HeadController(self.board)
            self.motion = MotionPlayer(self.board)
            self.sensors = SensorReader(self.board)
            self.peripherals = Peripherals(self.board)

            self._camera = Camera(undistort=undistort_camera, distortion_k1=distortion_k1)

            if remote_vision:
                from .remote_vision import RemoteVisionClient
                self.vision = RemoteVisionClient(host, port, self._camera)
                self._vision_mode = 'remote'
            else:
                self.vision = VisionSystem(self._camera)
                self._vision_mode = 'local'
        except Exception:
            self.board.close()
            raise

        self._last_pose = None

    def close(self) -> bool:
        """
        Cleanup and close connections.

        Returns:
            True if the motion stopped before the port was closed, False if it
            was still running and the shutdown went ahead anyway.
        """
        self.motion.stop()
        stopped = self.motion.wait(timeout=MOTION_STOP_TIMEOUT_S)
        if not stopped:
            print(f"[Robot] Motion did not stop within {MOTION_STOP_TIMEOUT_S}s; closing anyway")
        self.sensors.stop()
        self.vision.stop()
        self.board.close()
        return stopped

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ========== Quick Actions ==========

    def stand(self, blocking: bool = True):
        """Stand in default pose (stand.d6a)"""
        return self.play('stand', blocking)

    def stand_low(self, blocking: bool = True):
        """Stand in low/crouched pose (stand_low.d6a)"""
        return self.play('stand_low', blocking)

    def zero(self, duration: float = 1.0):
        """Move all body servos to centre"""
        self.servos.set_body([SERVO_CENTER] * len(BODY_SERVO_IDS), duration)
        self._last_pose = 'zero'
        time.sleep(duration)

    def relax(self):
        """Disable all servo torque"""
        self.servos.enable_all_torque(False)

    def enable(self):
        """Enable all servo torque"""
        self.servos.enable_all_torque(True)

    # ========== Motion Methods (uses d6a files) ==========

    def play(self, motion_name: str, blocking: bool = True) -> bool:
        """Play a motion sequence from d6a file"""
        started = self.motion.play(motion_name, blocking)
        if started:
            self._last_pose = motion_name
        return started

    def greet(self, blocking: bool = True):
        """Play greeting motion (greet.d6a)"""
        return self.play('greet', blocking)

    def wave(self, blocking: bool = True):
        """Play wave motion (wave.d6a)"""
        return self.play('wave', blocking)

    # Walking motions
    def walk_forward(self, blocking: bool = True):
        """Walk forward one cycle (forward.d6a)"""
        return self.play('forward', blocking)

    def walk_backward(self, blocking: bool = True):
        """Walk backward one cycle (back.d6a)"""
        return self.play('back', blocking)

    def step_left(self, blocking: bool = True):
        """Side step left (move_left.d6a)"""
        return self.play('move_left', blocking)

    def step_right(self, blocking: bool = True):
        """Side step right (move_right.d6a)"""
        return self.play('move_right', blocking)

    def turn_left(self, blocking: bool = True):
        """Turn left in place (turn_left.d6a)"""
        return self.play('turn_left', blocking)

    def turn_right(self, blocking: bool = True):
        """Turn right in place (turn_right.d6a)"""
        return self.play('turn_right', blocking)

    # Stair motions
    def climb_stairs(self, blocking: bool = True):
        """Climb stairs (climb_stairs.d6a)"""
        return self.play('climb_stairs', blocking)

    def descend_stairs(self, blocking: bool = True):
        """Descend stairs (descend_stairs.d6a)"""
        return self.play('descend_stairs', blocking)

    # Recovery motions
    def get_up_from_front(self, blocking: bool = True):
        """Get up from lying face down (lie_to_stand.d6a)"""
        return self.play('lie_to_stand', blocking)

    def get_up_from_back(self, blocking: bool = True):
        """Get up from lying on back (recline_to_stand.d6a)"""
        return self.play('recline_to_stand', blocking)

    # List available motions
    def list_motions(self):
        """List all available d6a motion files"""
        return self.motion.list_motions()

    # ========== Arm Control ==========

    def raise_left_arm(self, duration: float = 1.0):
        """Raise LEFT arm, lower right to neutral (true mirroring: user's RIGHT hand)."""
        self.servos.set_positions({**_LEFT_ARM_RAISED, **_RIGHT_ARM_NEUTRAL}, duration)

    def raise_right_arm(self, duration: float = 1.0):
        """Raise RIGHT arm, lower left to neutral (true mirroring: user's LEFT hand)."""
        self.servos.set_positions({**_RIGHT_ARM_RAISED, **_LEFT_ARM_NEUTRAL}, duration)

    def raise_both_arms(self, duration: float = 1.0):
        """Raise BOTH arms."""
        self.servos.set_positions({**_LEFT_ARM_RAISED, **_RIGHT_ARM_RAISED}, duration)

    def lower_arms(self, duration: float = 1.0):
        """Lower both arms to neutral (stand) position."""
        self.servos.set_positions({**_LEFT_ARM_NEUTRAL, **_RIGHT_ARM_NEUTRAL}, duration)

    def lower_left_arm(self, duration: float = 1.0):
        """Lower left arm only to neutral."""
        self.servos.set_positions(_LEFT_ARM_NEUTRAL, duration)

    def lower_right_arm(self, duration: float = 1.0):
        """Lower right arm only to neutral."""
        self.servos.set_positions(_RIGHT_ARM_NEUTRAL, duration)

    # ========== Peripheral Shortcuts ==========

    def beep(self, freq: int = 2000, duration: float = 0.1):
        """Play a beep sound"""
        self.peripherals.beep(freq, duration)

    def chirp(self):
        """Quick chirp sound"""
        self.peripherals.chirp()

    # ========== Sensor Shortcuts ==========

    def get_battery(self) -> Optional[float]:
        """Get battery percentage"""
        return self.sensors.get_battery_percent()

    def get_battery_status(self) -> Optional[str]:
        """Get battery level as 'ok', 'low' or 'critical'"""
        return self.sensors.get_battery_status()

    def start_sensors(self):
        """Start sensor polling"""
        self.sensors.start()

    def stop_sensors(self):
        """Stop sensor polling"""
        self.sensors.stop()

    # ========== Status ==========

    def status(self) -> dict:
        """Get robot status"""
        battery = self.sensors.get_battery_state()
        return {
            'connected': self.board.port.is_open,
            'battery': battery.percent if battery else None,
            'battery_voltage': battery.voltage_mv if battery else None,
            'battery_status': battery.status if battery else None,
            'last_pose': self._last_pose,
            'motion_playing': self.motion.is_playing,
            'current_motion': self.motion.current_motion,
            'available_motions': self.motion.list_motions(),
            'vision_mode': self._vision_mode,
        }

    def print_status(self):
        """Print robot status"""
        s = self.status()
        print("=" * 40)
        print("AINEX Robot Status")
        print("=" * 40)
        print(f"Connected: {s['connected']}")
        if s['battery'] is None:
            print("Battery: N/A")
        else:
            print(f"Battery: {s['battery']:.1f}% "
                  f"({s['battery_voltage'] / 1000:.2f}V, {s['battery_status']})")
        print(f"Last Pose: {s['last_pose']}")
        print(f"Motion Playing: {s['motion_playing']}")
        print(f"Available Motions: {len(s['available_motions'])}")
        print("=" * 40)
