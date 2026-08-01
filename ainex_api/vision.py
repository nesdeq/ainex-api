"""
Vision Module
=============

Face tracking and gesture recognition using MediaPipe Pose.

Architecture:
- Single Pose model provides landmarks for everything
- FaceAnalyzer extracts face position from landmarks
- GestureAnalyzer extracts gestures from landmarks
- Each analyzer is independent and can be improved separately

Public API (VisionSystem):
- update() - capture frame, run detection
- get_face() - cached FaceData or None
- get_gesture() - cached GestureData
- get_frame() - raw BGR frame
"""

import cv2
import time
import numpy as np
from collections import deque, Counter
from dataclasses import dataclass
from typing import Optional

try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False
    print("[Vision] WARNING: MediaPipe not installed")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class FaceData:
    """Face detection result."""
    x: int
    y: int
    width: int
    height: int
    timestamp: float

    @property
    def size(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> tuple:
        return (self.x, self.y)


@dataclass
class GestureData:
    """Gesture detection result."""
    gesture: str  # 'none', 'left_hand_raised', 'right_hand_raised', 'both_hands_raised', 'waving'
    confidence: float
    timestamp: float


# =============================================================================
# Analyzers - Independent, improvable components
# =============================================================================

class FaceAnalyzer:
    """
    Extract face position from pose landmarks.

    Uses nose landmark for position, eye distance for size estimation.
    Can be improved independently of gesture detection.
    """

    def __init__(self, min_visibility: float = 0.5):
        self._min_visibility = min_visibility

    def analyze(self, landmarks, width: int, height: int) -> Optional[FaceData]:
        """
        Extract face position from pose landmarks.

        Args:
            landmarks: MediaPipe pose landmarks
            width: Frame width in pixels
            height: Frame height in pixels

        Returns:
            FaceData or None if face not visible
        """
        if landmarks is None:
            return None

        lm = landmarks.landmark
        nose = lm[mp_pose.PoseLandmark.NOSE]

        if nose.visibility < self._min_visibility:
            return None

        # Estimate face size from eye distance
        left_eye = lm[mp_pose.PoseLandmark.LEFT_EYE]
        right_eye = lm[mp_pose.PoseLandmark.RIGHT_EYE]
        eye_dist = abs(left_eye.x - right_eye.x) * width
        face_size = int(eye_dist * 2.5) if eye_dist > 10 else 80

        # Landmarks are extrapolated past the frame edge; a tracking target
        # outside the image is meaningless, so pin it to the border.
        return FaceData(
            x=min(max(int(nose.x * width), 0), width - 1),
            y=min(max(int(nose.y * height), 0), height - 1),
            width=face_size,
            height=face_size,
            timestamp=time.time()
        )


# A landmark below this is a guess at an occluded joint, not an observation.
MIN_LANDMARK_VISIBILITY = 0.4
WAVE_MIN_VISIBILITY = 0.5

# Wrist clearance above the shoulder, in normalised frame height, for a raise.
RAISE_MARGIN = 0.12


class GestureAnalyzer:
    """
    Extract gestures from pose landmarks.

    Detects: hand raises (left/right/both), waving.
    Can be improved independently of face detection.
    """

    def __init__(self, confirm_frames: int = 3):
        self._confirm_frames = confirm_frames

        # Wave detection history (~2s at 20fps)
        self._elbow_angle_history = {
            'left': deque(maxlen=40),
            'right': deque(maxlen=40)
        }
        self._wrist_x_history = {
            'left': deque(maxlen=40),
            'right': deque(maxlen=40)
        }

        # Gesture confirmation (need N consecutive frames)
        self._gesture_history = deque(maxlen=confirm_frames)
        self._confirmed_gesture = 'none'

    def _calc_elbow_angle(self, shoulder, elbow, wrist) -> float:
        """
        Calculate angle at elbow joint in 3D (in degrees).
        Straight arm = 180°, fully bent = small angle.
        """
        es = np.array([
            shoulder.x - elbow.x,
            shoulder.y - elbow.y,
            (shoulder.z - elbow.z) * 0.5
        ])
        ew = np.array([
            wrist.x - elbow.x,
            wrist.y - elbow.y,
            (wrist.z - elbow.z) * 0.5
        ])

        mag_es = np.linalg.norm(es)
        mag_ew = np.linalg.norm(ew)

        if mag_es < 0.001 or mag_ew < 0.001:
            return 180.0

        cos_angle = np.clip(np.dot(es, ew) / (mag_es * mag_ew), -1, 1)
        return np.degrees(np.arccos(cos_angle))

    def _in_wave_position(self, wrist, elbow, shoulder) -> bool:
        """Check if hand is in wave position (wrist above elbow and shoulder)."""
        return wrist.y < elbow.y and wrist.y < shoulder.y

    def _count_reversals(self, values, min_change: float) -> int:
        """Count direction reversals in a sequence."""
        reversals = 0
        prev_dir = 0
        for i in range(1, len(values)):
            diff = values[i] - values[i - 1]
            if abs(diff) < min_change:
                continue
            curr_dir = 1 if diff > 0 else -1
            if prev_dir != 0 and curr_dir != prev_dir:
                reversals += 1
            prev_dir = curr_dir
        return reversals

    def analyze(self, landmarks) -> GestureData:
        """
        Extract gesture from pose landmarks.

        Args:
            landmarks: MediaPipe pose landmarks

        Returns:
            GestureData (always returns, 'none' if no gesture)
        """
        if landmarks is None:
            self._elbow_angle_history['left'].clear()
            self._elbow_angle_history['right'].clear()
            self._wrist_x_history['left'].clear()
            self._wrist_x_history['right'].clear()
            return self._confirm('none')

        lm = landmarks.landmark

        # Get key landmarks
        # Left arm: 11=shoulder, 13=elbow, 15=wrist
        # Right arm: 12=shoulder, 14=elbow, 16=wrist
        ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]   # 11
        rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]  # 12
        le = lm[mp_pose.PoseLandmark.LEFT_ELBOW]      # 13
        re = lm[mp_pose.PoseLandmark.RIGHT_ELBOW]     # 14
        lw = lm[mp_pose.PoseLandmark.LEFT_WRIST]      # 15
        rw = lm[mp_pose.PoseLandmark.RIGHT_WRIST]     # 16

        # Need visible shoulders for gesture detection
        if ls.visibility < MIN_LANDMARK_VISIBILITY or rs.visibility < MIN_LANDMARK_VISIBILITY:
            return self._confirm('none')

        # Track angle + wrist x ONLY when in wave position
        if le.visibility > MIN_LANDMARK_VISIBILITY and lw.visibility > MIN_LANDMARK_VISIBILITY:
            if self._in_wave_position(lw, le, ls):
                self._elbow_angle_history['left'].append(self._calc_elbow_angle(ls, le, lw))
                self._wrist_x_history['left'].append(lw.x)
            else:
                self._elbow_angle_history['left'].clear()
                self._wrist_x_history['left'].clear()

        if re.visibility > MIN_LANDMARK_VISIBILITY and rw.visibility > MIN_LANDMARK_VISIBILITY:
            if self._in_wave_position(rw, re, rs):
                self._elbow_angle_history['right'].append(self._calc_elbow_angle(rs, re, rw))
                self._wrist_x_history['right'].append(rw.x)
            else:
                self._elbow_angle_history['right'].clear()
                self._wrist_x_history['right'].clear()

        # --- Gesture Detection (priority order) ---

        # 1. Waving - elbow angle oscillation while hand raised
        if self._is_waving(lm):
            return self._confirm('waving')

        # 2. Hands raised - visible wrist above shoulder
        left_up = lw.visibility >= MIN_LANDMARK_VISIBILITY and lw.y < ls.y - RAISE_MARGIN
        right_up = rw.visibility >= MIN_LANDMARK_VISIBILITY and rw.y < rs.y - RAISE_MARGIN

        if left_up and right_up:
            return self._confirm('both_hands_raised')
        elif left_up:
            return self._confirm('left_hand_raised')
        elif right_up:
            return self._confirm('right_hand_raised')

        return self._confirm('none')

    def _is_waving(self, lm) -> bool:
        """
        Detect waving: combination of elbow angle oscillation AND horizontal wrist movement.

        Wave position = wrist above BOTH elbow and shoulder.
        Detection triggers on:
        - Strong angle oscillation alone (>=25°, >=3 reversals)
        - Strong horizontal movement alone (>=0.10, >=3 reversals)
        - Combined weaker signals (lower thresholds for both)
        """
        checks = [
            ('left',
             lm[mp_pose.PoseLandmark.LEFT_WRIST],    # 15
             lm[mp_pose.PoseLandmark.LEFT_ELBOW],    # 13
             lm[mp_pose.PoseLandmark.LEFT_SHOULDER]),  # 11
            ('right',
             lm[mp_pose.PoseLandmark.RIGHT_WRIST],   # 16
             lm[mp_pose.PoseLandmark.RIGHT_ELBOW],   # 14
             lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]),  # 12
        ]

        for hand, wrist, elbow, shoulder in checks:
            if wrist.visibility < WAVE_MIN_VISIBILITY or elbow.visibility < WAVE_MIN_VISIBILITY:
                continue

            if not self._in_wave_position(wrist, elbow, shoulder):
                continue

            angles = list(self._elbow_angle_history[hand])
            wrist_xs = list(self._wrist_x_history[hand])

            if len(angles) < 15 or len(wrist_xs) < 15:
                continue

            # Analyze elbow angle oscillation
            angle_range = max(angles) - min(angles)
            angle_reversals = self._count_reversals(angles, 2.0)

            # Analyze horizontal wrist movement
            x_range = max(wrist_xs) - min(wrist_xs)
            x_reversals = self._count_reversals(wrist_xs, 0.01)

            # Strong angle oscillation alone
            if angle_range >= 25 and angle_reversals >= 3:
                return True

            # Strong horizontal movement alone
            if x_range >= 0.10 and x_reversals >= 3:
                return True

            # Combined signals (lower thresholds)
            if (angle_range >= 12 and x_range >= 0.05 and
                angle_reversals >= 2 and x_reversals >= 2):
                return True

        return False

    def _confirm(self, raw_gesture: str) -> GestureData:
        """Confirm gesture (need majority in recent frames)."""
        self._gesture_history.append(raw_gesture)

        if len(self._gesture_history) >= 2:
            counter = Counter(self._gesture_history)
            most_common, count = counter.most_common(1)[0]
            if count >= 2:
                self._confirmed_gesture = most_common

        return GestureData(
            gesture=self._confirmed_gesture,
            confidence=1.0,
            timestamp=time.time()
        )

    def reset(self):
        """Reset analyzer state."""
        self._elbow_angle_history['left'].clear()
        self._elbow_angle_history['right'].clear()
        self._wrist_x_history['left'].clear()
        self._wrist_x_history['right'].clear()
        self._gesture_history.clear()
        self._confirmed_gesture = 'none'


# =============================================================================
# Pose skeleton for debug visualization
# =============================================================================

POSE_CONNECTIONS = [
    # Torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Left arm
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    # Right arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    # Left leg
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    # Right leg
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]


# =============================================================================
# VisionSystem - Public API
# =============================================================================

class VisionSystem:
    """
    Combined face + gesture detection from camera.

    Uses single MediaPipe Pose model for everything.
    Face and gesture are extracted independently from landmarks.

    Usage:
        vision = VisionSystem(camera)
        vision.start()
        while True:
            vision.update()
            face = vision.get_face()
            gesture = vision.get_gesture()
    """

    def __init__(self, camera, pose_model: int = 1):
        """
        Args:
            camera: Camera instance
            pose_model: 0=Lite (fast), 1=Full (balanced), 2=Heavy (accurate)
        """
        self.camera = camera

        # Pose detection (single model for everything)
        self._pose = None
        if HAS_MEDIAPIPE:
            self._pose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=pose_model,
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )

        # Independent analyzers
        self._face_analyzer = FaceAnalyzer()
        self._gesture_analyzer = GestureAnalyzer()

        # Cached state
        self._last_frame = None
        self._last_frame_hw = (480, 640)
        self._landmarks = None
        self._cached_face = None
        self._cached_gesture = None

    def start(self) -> bool:
        """Start camera capture."""
        return self.camera.start()

    def stop(self):
        """Stop camera and release resources."""
        self.camera.stop()
        if self._pose:
            self._pose.close()

    def update(self) -> bool:
        """
        Capture frame and run detection.

        Call once per loop, then use get_face()/get_gesture().

        Returns:
            True if frame captured, False otherwise
        """
        self._last_frame = self.camera.read()
        if self._last_frame is None:
            self._landmarks = None
            self._cached_face = None
            self._cached_gesture = GestureData(gesture='none', confidence=0, timestamp=time.time())
            return False

        self._last_frame_hw = self._last_frame.shape[:2]
        h, w = self._last_frame_hw

        # Run pose detection
        if self._pose:
            rgb = cv2.cvtColor(self._last_frame, cv2.COLOR_BGR2RGB)
            results = self._pose.process(rgb)
            self._landmarks = results.pose_landmarks
        else:
            self._landmarks = None

        # Extract face and gesture (independent analyzers)
        self._cached_face = self._face_analyzer.analyze(self._landmarks, w, h)
        self._cached_gesture = self._gesture_analyzer.analyze(self._landmarks)

        return True

    def get_face(self) -> Optional[FaceData]:
        """Get face position from last update()."""
        return self._cached_face

    def get_gesture(self) -> Optional[GestureData]:
        """Get gesture from last update()."""
        return self._cached_gesture

    def get_frame(self) -> Optional[np.ndarray]:
        """Get raw BGR frame from last update()."""
        return self._last_frame

    @property
    def landmarks(self):
        """Raw pose landmarks for custom analysis."""
        return self._landmarks

    @property
    def has_mediapipe(self) -> bool:
        """Check if MediaPipe is available."""
        return HAS_MEDIAPIPE

    def draw_debug(self, frame: np.ndarray = None,
                   face: bool = False, gesture: bool = False, pose: bool = False) -> np.ndarray:
        """
        Draw debug overlays on frame.

        Args:
            frame: BGR frame (default: last captured)
            face: Draw face box
            gesture: Draw gesture label
            pose: Draw skeleton

        Returns:
            Frame with overlays
        """
        if frame is None:
            frame = self._last_frame
        if frame is None:
            return None

        out = frame.copy()
        h, w = out.shape[:2]

        # Face box (green)
        if face and self._cached_face:
            f = self._cached_face
            x1, y1 = f.x - f.width // 2, f.y - f.height // 2
            x2, y2 = f.x + f.width // 2, f.y + f.height // 2
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.drawMarker(out, (f.x, f.y), (0, 255, 0), cv2.MARKER_CROSS, 10, 2)

        # Gesture label (yellow)
        if gesture and self._cached_gesture:
            text = self._cached_gesture.gesture
            color = (0, 255, 255) if text != 'none' else (128, 128, 128)
            cv2.putText(out, f'Gesture: {text}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Pose skeleton
        if pose and self._landmarks:
            lm = self._landmarks.landmark

            # Connections (white)
            for start_idx, end_idx in POSE_CONNECTIONS:
                start, end = lm[start_idx], lm[end_idx]
                if start.visibility < 0.5 or end.visibility < 0.5:
                    continue
                p1 = (int(start.x * w), int(start.y * h))
                p2 = (int(end.x * w), int(end.y * h))
                cv2.line(out, p1, p2, (255, 255, 255), 1)

            # Joints (cyan) with index labels
            for i, pt in enumerate(lm):
                if pt.visibility < 0.5:
                    continue
                x, y = int(pt.x * w), int(pt.y * h)
                radius = 4 if i in (11, 12, 13, 14, 15, 16, 23, 24) else 2
                cv2.circle(out, (x, y), radius, (255, 255, 0), -1)
                cv2.putText(out, str(i), (x + 3, y - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)

        return out
