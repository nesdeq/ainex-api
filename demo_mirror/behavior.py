"""
Behavior Controller
===================

Simple gesture response and mirroring.
Uses high-level ainex_api.
"""

import time
from enum import Enum, auto
from ainex_api import Robot


class RobotState(Enum):
    IDLE = auto()
    TRACKING = auto()


class BehaviorController:
    """
    Simple behavior controller:
    - Face tracking (head follows face)
    - Gesture mirroring (arms mirror user's pose)
    - Waving → greet animation
    """

    def __init__(self, robot: Robot, action_cooldown: float = 4.0):
        self.robot = robot
        self.current_state = RobotState.IDLE

        # Current pose being held
        self._current_pose = 'none'

        # Confirmation: need same gesture N times before acting
        self._last_gesture = 'none'
        self._gesture_count = 0
        self._confirm_frames = 4

        # Cooldown for actions (greet/wave)
        self._action_cooldown = action_cooldown
        self._last_action_time = 0

        # Face tracking
        self._face_lost_time = None
        self._face_timeout = 3.0

        # Stats
        self.stats = {'frames': 0, 'gestures_mirrored': 0, 'state_transitions': 0}

    def update(self):
        """Main update - call once per loop."""
        self.stats['frames'] += 1

        face = self.robot.vision.get_face()
        gesture = self.robot.vision.get_gesture()
        gesture_type = gesture.gesture if gesture else 'none'

        # Face tracking
        self._handle_face(face)

        # Don't process gestures while motion is playing
        if self.robot.motion.is_playing:
            return

        # Confirm gesture (same gesture for N frames)
        confirmed = self._confirm_gesture(gesture_type)
        if not confirmed:
            return

        # Act on confirmed gesture
        self._act_on_gesture(gesture_type)

    def _handle_face(self, face):
        """Track face with head."""
        if face:
            self._face_lost_time = None
            if self.current_state == RobotState.IDLE:
                self._transition(RobotState.TRACKING)
            self.robot.head.track_point(face.x, face.y)
        else:
            if self._face_lost_time is None:
                self._face_lost_time = time.monotonic()
            elif time.monotonic() - self._face_lost_time > self._face_timeout:
                if self.current_state != RobotState.IDLE:
                    self._transition(RobotState.IDLE)

    def _confirm_gesture(self, gesture_type: str) -> bool:
        """Return True if gesture is confirmed (seen N times in a row)."""
        if gesture_type == self._last_gesture:
            self._gesture_count += 1
        else:
            self._last_gesture = gesture_type
            self._gesture_count = 1

        return self._gesture_count >= self._confirm_frames

    def _act_on_gesture(self, gesture_type: str):
        """Execute action for confirmed gesture."""
        now = time.monotonic()
        cooldown_ok = now - self._last_action_time > self._action_cooldown

        # Actions that need cooldown
        if gesture_type == 'waving' and cooldown_ok:
            print("[ACTION] waving -> greet")
            self._last_action_time = now
            self._current_pose = 'none'
            self.robot.greet(blocking=False)
            self.stats['gestures_mirrored'] += 1
            return

        # Mirroring (no cooldown, just check if pose changed)
        if gesture_type == self._current_pose:
            return

        # Mirrored, not copied: the user's left hand raises the robot's right arm,
        # so the raised arm appears on the same side as the user's.
        if gesture_type == 'left_hand_raised':
            print("[MIRROR] -> left_hand_raised")
            self._current_pose = gesture_type
            self.robot.raise_right_arm(duration=0.8)
            self.stats['gestures_mirrored'] += 1

        elif gesture_type == 'right_hand_raised':
            print("[MIRROR] -> right_hand_raised")
            self._current_pose = gesture_type
            self.robot.raise_left_arm(duration=0.8)
            self.stats['gestures_mirrored'] += 1

        elif gesture_type == 'both_hands_raised':
            print("[MIRROR] -> both_hands_raised")
            self._current_pose = gesture_type
            self.robot.raise_both_arms(duration=0.8)
            self.stats['gestures_mirrored'] += 1

        elif gesture_type == 'none' and self._current_pose != 'none':
            print("[MIRROR] -> none (lower arms)")
            self._current_pose = 'none'
            self.robot.lower_arms(duration=0.8)

    def _transition(self, new_state: RobotState):
        """State transition."""
        if new_state != self.current_state:
            print(f"[STATE] {self.current_state.name} -> {new_state.name}")
            self.current_state = new_state
            self.stats['state_transitions'] += 1
            if new_state == RobotState.TRACKING:
                self.robot.head.reset_tracking()
