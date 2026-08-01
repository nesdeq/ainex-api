"""
Head Controller
===============

Controls head pan/tilt servos with PID tracking support.
"""

import time
from typing import Optional, Tuple
from .board import Board
from .servos import SERVO_CENTER, servo_limits


# Head servo configuration
HEAD_PAN_ID = 23
HEAD_TILT_ID = 24

# Travel limits come from servos.SERVO_LIMITS so both APIs enforce the same range.
PAN_MIN, PAN_MAX = servo_limits(HEAD_PAN_ID)
PAN_CENTER = SERVO_CENTER

TILT_MIN, TILT_MAX = servo_limits(HEAD_TILT_ID)
TILT_CENTER = SERVO_CENTER

# Default image dimensions (for tracking)
DEFAULT_IMAGE_WIDTH = 640
DEFAULT_IMAGE_HEIGHT = 480


class PIDController:
    """Simple PID controller with anti-windup"""

    def __init__(self, kp: float = 0.15, ki: float = 0.002, kd: float = 0.005,
                 windup_limit: float = 100.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.windup_limit = windup_limit
        self._integral = 0.0
        self._prev_error = None
        self._last_time = time.monotonic()

    def update(self, error: float, dt: Optional[float] = None) -> float:
        """Compute PID output"""
        if dt is None:
            now = time.monotonic()
            dt = max(0.001, min(0.1, now - self._last_time))
            self._last_time = now

        p = self.kp * error

        self._integral += error * dt
        self._integral = max(-self.windup_limit, min(self.windup_limit, self._integral))
        i = self.ki * self._integral

        if self._prev_error is None:
            d = 0.0
        else:
            d = self.kd * (error - self._prev_error) / dt
        self._prev_error = error

        return p + i + d

    def reset(self):
        """Reset controller state"""
        self._integral = 0.0
        self._prev_error = None
        self._last_time = time.monotonic()


class HeadController:
    """
    Head servo controller with PID tracking.

    Controls pan (left/right) and tilt (up/down) head motion.
    Provides high-level track_point() for smooth face following.
    """

    def __init__(self, board: Board,
                 pan_pid: Tuple[float, float, float] = (0.12, 0.001, 0.003),
                 tilt_pid: Tuple[float, float, float] = (0.10, 0.001, 0.002),
                 dead_zone: Tuple[int, int] = (25, 25)):
        self.board = board

        # Current positions
        self._pan = PAN_CENTER
        self._tilt = TILT_CENTER

        # PID controllers for tracking
        self._pan_pid = PIDController(kp=pan_pid[0], ki=pan_pid[1], kd=pan_pid[2])
        self._tilt_pid = PIDController(kp=tilt_pid[0], ki=tilt_pid[1], kd=tilt_pid[2])

        # Dead zone (pixels from center to ignore)
        self._dead_zone_x = dead_zone[0]
        self._dead_zone_y = dead_zone[1]

        # Smoothing for face position
        self._smooth_x = DEFAULT_IMAGE_WIDTH / 2
        self._smooth_y = DEFAULT_IMAGE_HEIGHT / 2
        self._smoothing_factor = 0.4

    @property
    def pan(self) -> int:
        """Current pan position"""
        return self._pan

    @property
    def tilt(self) -> int:
        """Current tilt position"""
        return self._tilt

    def move(self, pan: Optional[int] = None, tilt: Optional[int] = None,
             duration: float = 0.033):
        """
        Move head to absolute position.

        Args:
            pan: Pan position (125-875, 500=center)
            tilt: Tilt position (315-625, 500=center)
            duration: Movement duration in seconds
        """
        if pan is not None:
            self._pan = max(PAN_MIN, min(PAN_MAX, pan))
        if tilt is not None:
            self._tilt = max(TILT_MIN, min(TILT_MAX, tilt))

        self.board.bus_servo_set_position(duration, [
            [HEAD_PAN_ID, int(self._pan)],
            [HEAD_TILT_ID, int(self._tilt)]
        ])

    def center(self, duration: float = 1.0, blocking: bool = False):
        """
        Center head position.

        Args:
            duration: Movement duration in seconds
            blocking: If True, wait for movement to complete
        """
        self.move(PAN_CENTER, TILT_CENTER, duration)
        self.reset_tracking()
        if blocking:
            time.sleep(duration + 0.05)

    def track_point(self, x: int, y: int,
                    image_width: int = DEFAULT_IMAGE_WIDTH,
                    image_height: int = DEFAULT_IMAGE_HEIGHT,
                    duration: float = 0.05) -> bool:
        """
        Track a point in image coordinates using PID control.

        Call this repeatedly with face/object coordinates to smoothly
        track it with the head. Handles all PID math internally.

        Args:
            x: Target X coordinate in pixels
            y: Target Y coordinate in pixels
            image_width: Image width in pixels
            image_height: Image height in pixels
            duration: Servo movement duration

        Returns:
            True if head moved, False if target already centered (in dead zone)
        """
        center_x = image_width / 2
        center_y = image_height / 2

        # Smooth the input position
        self._smooth_x = (self._smoothing_factor * x +
                         (1 - self._smoothing_factor) * self._smooth_x)
        self._smooth_y = (self._smoothing_factor * y +
                         (1 - self._smoothing_factor) * self._smooth_y)

        # Calculate error from center
        error_x = self._smooth_x - center_x
        error_y = self._smooth_y - center_y

        # Apply dead zone
        if abs(error_x) < self._dead_zone_x:
            error_x = 0
        if abs(error_y) < self._dead_zone_y:
            error_y = 0

        # Skip if within dead zone
        if error_x == 0 and error_y == 0:
            return False

        # PID control
        pan_adj = -self._pan_pid.update(error_x)
        tilt_adj = -self._tilt_pid.update(error_y)

        # Apply adjustments (clamping handled by move())
        new_pan = self._pan + pan_adj
        new_tilt = self._tilt + tilt_adj

        self.move(int(new_pan), int(new_tilt), duration)
        return True

    def reset_tracking(self):
        """Reset PID controllers and smoothing state"""
        self._pan_pid.reset()
        self._tilt_pid.reset()
        self._smooth_x = DEFAULT_IMAGE_WIDTH / 2
        self._smooth_y = DEFAULT_IMAGE_HEIGHT / 2

    def nod(self, amplitude: int = 50, duration: float = 0.3, count: int = 2):
        """Make the robot nod (yes)"""
        for _ in range(count):
            self.move(tilt=TILT_CENTER - amplitude, duration=duration)
            time.sleep(duration)
            self.move(tilt=TILT_CENTER + amplitude, duration=duration)
            time.sleep(duration)
        self.move(tilt=TILT_CENTER, duration=duration)
        time.sleep(duration)

    def shake(self, amplitude: int = 75, duration: float = 0.3, count: int = 2):
        """Make the robot shake head (no)"""
        for _ in range(count):
            self.move(pan=PAN_CENTER - amplitude, duration=duration)
            time.sleep(duration)
            self.move(pan=PAN_CENTER + amplitude, duration=duration)
            time.sleep(duration)
        self.move(pan=PAN_CENTER, duration=duration)
        time.sleep(duration)
