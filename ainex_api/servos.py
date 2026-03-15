"""
Servo Controller
================

High-level servo control with named joints and body mapping.
"""

from typing import Dict, List, Optional
from .board import Board


# AINEX servo mapping (servo_id: name)
# FROM: /home/qp/playground/artemis/docs/servo.md (AUTHORITATIVE SOURCE)
SERVO_MAP = {
    # Left Leg (odd: 1,3,5,7,9,11)
    1: 'l_ankle_roll',
    3: 'l_ankle_pitch',
    5: 'l_knee',
    7: 'l_hip_pitch',
    9: 'l_hip_roll',
    11: 'l_hip_yaw',
    # Right Leg (even: 2,4,6,8,10,12)
    2: 'r_ankle_roll',
    4: 'r_ankle_pitch',
    6: 'r_knee',
    8: 'r_hip_pitch',
    10: 'r_hip_roll',
    12: 'r_hip_yaw',
    # Arms (13-20) - odd=left, even=right
    13: 'l_shoulder_pitch',  # Raised: 360, Neutral: 835
    14: 'r_shoulder_pitch',  # Raised: 640, Neutral: 165
    15: 'l_shoulder_roll',   # 830
    16: 'r_shoulder_roll',   # Raised: 165, Neutral: 170
    17: 'l_elbow_pitch',     # 500
    18: 'r_elbow_pitch',     # 500
    19: 'l_elbow_yaw',       # Raised: 40, Neutral: 150
    20: 'r_elbow_yaw',       # Raised: 960, Neutral: 850
    # Grippers (21-22)
    21: 'l_gripper',
    22: 'r_gripper',
    # Head (23-24)
    23: 'head_pan',          # Range: 125-875, Center: 500
    24: 'head_tilt',         # Range: 315-625, Center: 500
}

# Reverse mapping
NAME_TO_ID = {v: k for k, v in SERVO_MAP.items()}

# Body servo IDs (excluding head)
BODY_SERVO_IDS = list(range(1, 23))
ALL_SERVO_IDS = list(range(1, 25))


class ServoController:
    """
    High-level servo control.

    Provides named joint access and body-wide operations.
    """

    def __init__(self, board: Board):
        self.board = board

    def set_position(self, servo_id: int, position: int, duration: float = 0.5):
        """
        Set single servo position.

        Args:
            servo_id: Servo ID (1-24)
            position: Target position (0-1000)
            duration: Movement time in seconds
        """
        self.board.bus_servo_set_position(duration, [[servo_id, position]])

    def set_positions(self, positions: Dict[int, int], duration: float = 0.5):
        """
        Set multiple servo positions.

        Args:
            positions: Dict of {servo_id: position}
            duration: Movement time in seconds
        """
        pos_list = [[k, v] for k, v in positions.items()]
        self.board.bus_servo_set_position(duration, pos_list)

    def set_by_name(self, name: str, position: int, duration: float = 0.5):
        """
        Set servo position by joint name.

        Args:
            name: Joint name (e.g., 'head_pan', 'r_shoulder_pitch')
            position: Target position (0-1000)
            duration: Movement time in seconds
        """
        servo_id = NAME_TO_ID.get(name)
        if servo_id is None:
            raise ValueError(f"Unknown servo name: {name}")
        self.set_position(servo_id, position, duration)

    def set_body(self, positions: List[int], duration: float = 1.0):
        """
        Set all body servos (1-22) at once.

        Args:
            positions: List of 22 positions for servos 1-22
            duration: Movement time in seconds
        """
        if len(positions) != 22:
            raise ValueError(f"Expected 22 positions, got {len(positions)}")
        pos_list = [[i + 1, positions[i]] for i in range(22)]
        self.board.bus_servo_set_position(duration, pos_list)

    def get_position(self, servo_id: int, use_cache: bool = False) -> Optional[int]:
        """Read servo position"""
        return self.board.bus_servo_read_position(servo_id, use_cache)

    def get_positions(self, servo_ids: List[int], use_cache: bool = True) -> Dict[int, int]:
        """Read multiple servo positions"""
        result = {}
        for sid in servo_ids:
            pos = self.get_position(sid, use_cache)
            if pos is not None:
                result[sid] = pos
        return result

    def enable_torque(self, servo_id: int, enable: bool = True):
        """Enable/disable servo torque"""
        self.board.bus_servo_enable_torque(servo_id, enable)

    def enable_all_torque(self, enable: bool = True):
        """Enable/disable all servo torque"""
        for sid in ALL_SERVO_IDS:
            self.enable_torque(sid, enable)

    def stop(self, servo_ids: Optional[List[int]] = None):
        """Stop specified servos (default: all)"""
        if servo_ids is None:
            servo_ids = ALL_SERVO_IDS
        self.board.bus_servo_stop(servo_ids)

    def read_temperature(self, servo_id: int) -> Optional[int]:
        """Read servo temperature in Celsius"""
        return self.board.bus_servo_read_temperature(servo_id)

    def read_voltage(self, servo_id: int) -> Optional[int]:
        """Read servo voltage in mV"""
        return self.board.bus_servo_read_voltage(servo_id)

    def set_offset(self, servo_id: int, offset: int):
        """Set servo offset (-128 to 127)"""
        self.board.bus_servo_set_offset(servo_id, offset)

    def save_offset(self, servo_id: int):
        """Save servo offset to EEPROM"""
        self.board.bus_servo_save_offset(servo_id)

    @staticmethod
    def get_servo_name(servo_id: int) -> str:
        """Get servo name by ID"""
        return SERVO_MAP.get(servo_id, f"servo_{servo_id}")

    @staticmethod
    def get_servo_id(name: str) -> Optional[int]:
        """Get servo ID by name"""
        return NAME_TO_ID.get(name)
