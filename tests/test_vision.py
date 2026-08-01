"""Face extraction and gesture classification from pose landmarks."""

import pytest

from ainex_api.vision import (
    FaceAnalyzer, GestureAnalyzer, FaceData, GestureData,
    MIN_LANDMARK_VISIBILITY, RAISE_MARGIN, POSE_CONNECTIONS, HAS_MEDIAPIPE,
)

pytestmark = pytest.mark.skipif(not HAS_MEDIAPIPE, reason="mediapipe not installed")

if HAS_MEDIAPIPE:
    from ainex_api.vision import mp_pose
    PL = mp_pose.PoseLandmark
    LANDMARK_COUNT = len(PL)


class Landmark:
    def __init__(self, x=0.5, y=0.5, z=0.0, visibility=1.0):
        self.x, self.y, self.z, self.visibility = x, y, z, visibility


class Landmarks:
    """Stands in for MediaPipe's pose_landmarks result."""

    def __init__(self):
        self.landmark = [Landmark() for _ in range(LANDMARK_COUNT)]

    def set(self, index, **kwargs):
        for key, value in kwargs.items():
            setattr(self.landmark[index], key, value)
        return self


def standing():
    """Shoulders visible, both wrists hanging below them."""
    lm = Landmarks()
    lm.set(PL.LEFT_SHOULDER, x=0.4, y=0.4, visibility=1.0)
    lm.set(PL.RIGHT_SHOULDER, x=0.6, y=0.4, visibility=1.0)
    lm.set(PL.LEFT_ELBOW, x=0.4, y=0.6, visibility=1.0)
    lm.set(PL.RIGHT_ELBOW, x=0.6, y=0.6, visibility=1.0)
    lm.set(PL.LEFT_WRIST, x=0.4, y=0.8, visibility=1.0)
    lm.set(PL.RIGHT_WRIST, x=0.6, y=0.8, visibility=1.0)
    return lm


def raise_wrist(lm, side):
    wrist = PL.LEFT_WRIST if side == "left" else PL.RIGHT_WRIST
    shoulder = PL.LEFT_SHOULDER if side == "left" else PL.RIGHT_SHOULDER
    lm.set(wrist, y=lm.landmark[shoulder].y - RAISE_MARGIN - 0.05)
    return lm


def settle(analyzer, lm, frames=4):
    """Run enough frames for the confirmation window to agree."""
    result = None
    for _ in range(frames):
        result = analyzer.analyze(lm)
    return result.gesture


# ------------------------------------------------------------------- face

def test_face_is_none_without_landmarks():
    assert FaceAnalyzer().analyze(None, 640, 480) is None


def test_face_is_none_when_the_nose_is_not_visible():
    lm = standing().set(PL.NOSE, visibility=0.1)
    assert FaceAnalyzer().analyze(lm, 640, 480) is None


def test_face_centre_is_scaled_into_pixels():
    lm = standing().set(PL.NOSE, x=0.5, y=0.25, visibility=1.0)
    face = FaceAnalyzer().analyze(lm, 640, 480)
    assert isinstance(face, FaceData)
    assert face.x == 320
    assert face.y == 120


@pytest.mark.parametrize("nx,ny", [(-0.5, -0.5), (1.5, 1.5), (2.0, -1.0)])
def test_face_outside_the_frame_is_clamped_to_the_border(nx, ny):
    """Landmarks extrapolate past the edge; a tracking target must not."""
    lm = standing().set(PL.NOSE, x=nx, y=ny, visibility=1.0)
    face = FaceAnalyzer().analyze(lm, 640, 480)
    assert 0 <= face.x <= 639
    assert 0 <= face.y <= 479


def test_face_size_is_positive():
    lm = standing().set(PL.NOSE, visibility=1.0)
    face = FaceAnalyzer().analyze(lm, 640, 480)
    assert face.width > 0 and face.height > 0
    assert face.size == face.width * face.height
    assert face.center == (face.x, face.y)


# ---------------------------------------------------------------- gesture

def test_no_landmarks_gives_none_gesture():
    analyzer = GestureAnalyzer()
    result = analyzer.analyze(None)
    assert isinstance(result, GestureData)
    assert settle(analyzer, None) == 'none'


def test_hands_down_is_none():
    assert settle(GestureAnalyzer(), standing()) == 'none'


def test_left_wrist_above_the_shoulder_is_a_left_raise():
    assert settle(GestureAnalyzer(), raise_wrist(standing(), "left")) == 'left_hand_raised'


def test_right_wrist_above_the_shoulder_is_a_right_raise():
    assert settle(GestureAnalyzer(), raise_wrist(standing(), "right")) == 'right_hand_raised'


def test_both_wrists_above_the_shoulders_is_a_double_raise():
    lm = raise_wrist(raise_wrist(standing(), "left"), "right")
    assert settle(GestureAnalyzer(), lm) == 'both_hands_raised'


def test_an_occluded_wrist_does_not_raise_an_arm():
    """Regression: a guessed wrist position used to trigger a raise."""
    lm = raise_wrist(standing(), "left")
    lm.set(PL.LEFT_WRIST, visibility=MIN_LANDMARK_VISIBILITY - 0.1)
    assert settle(GestureAnalyzer(), lm) == 'none'


def test_invisible_shoulders_suppress_gesture_detection():
    lm = raise_wrist(standing(), "left")
    lm.set(PL.LEFT_SHOULDER, visibility=0.1)
    assert settle(GestureAnalyzer(), lm) == 'none'


def test_a_wrist_just_below_the_margin_is_not_a_raise():
    lm = standing()
    shoulder_y = lm.landmark[PL.LEFT_SHOULDER].y
    lm.set(PL.LEFT_WRIST, y=shoulder_y - RAISE_MARGIN + 0.01)
    assert settle(GestureAnalyzer(), lm) == 'none'


def test_a_single_frame_does_not_flip_a_confirmed_gesture():
    analyzer = GestureAnalyzer()
    raised = raise_wrist(standing(), "left")
    assert settle(analyzer, raised) == 'left_hand_raised'
    assert analyzer.analyze(standing()).gesture == 'left_hand_raised'


def test_reset_clears_the_confirmed_gesture():
    analyzer = GestureAnalyzer()
    settle(analyzer, raise_wrist(standing(), "left"))
    analyzer.reset()
    assert analyzer._confirmed_gesture == 'none'
    assert analyzer._elbow_angle_history['left'] == analyzer._elbow_angle_history['right']


def test_losing_the_person_clears_the_wave_history():
    analyzer = GestureAnalyzer()
    lm = raise_wrist(standing(), "left")
    for _ in range(20):
        analyzer.analyze(lm)
    analyzer.analyze(None)
    assert len(analyzer._elbow_angle_history['left']) == 0
    assert len(analyzer._wrist_x_history['left']) == 0


def test_a_still_raised_hand_is_not_a_wave():
    lm = raise_wrist(standing(), "left")
    lm.set(PL.LEFT_ELBOW, y=lm.landmark[PL.LEFT_SHOULDER].y - 0.05)
    analyzer = GestureAnalyzer()
    for _ in range(30):
        analyzer.analyze(lm)
    assert analyzer.analyze(lm).gesture != 'waving'


def test_an_oscillating_hand_reads_as_a_wave():
    analyzer = GestureAnalyzer()
    lm = standing()
    shoulder_y = lm.landmark[PL.LEFT_SHOULDER].y
    lm.set(PL.LEFT_ELBOW, y=shoulder_y - 0.05)
    lm.set(PL.LEFT_WRIST, y=shoulder_y - 0.20)
    for i in range(30):
        lm.set(PL.LEFT_WRIST, x=0.30 if i % 2 else 0.55)
        result = analyzer.analyze(lm)
    assert result.gesture == 'waving'


def test_elbow_angle_of_a_straight_arm_is_wide():
    analyzer = GestureAnalyzer()
    shoulder = Landmark(x=0.5, y=0.2)
    elbow = Landmark(x=0.5, y=0.5)
    wrist = Landmark(x=0.5, y=0.8)
    assert analyzer._calc_elbow_angle(shoulder, elbow, wrist) == pytest.approx(180.0, abs=1.0)


def test_elbow_angle_of_a_folded_arm_is_narrow():
    analyzer = GestureAnalyzer()
    shoulder = Landmark(x=0.5, y=0.2)
    elbow = Landmark(x=0.5, y=0.5)
    wrist = Landmark(x=0.5, y=0.2)
    assert analyzer._calc_elbow_angle(shoulder, elbow, wrist) < 10.0


def test_degenerate_limb_lengths_do_not_divide_by_zero():
    analyzer = GestureAnalyzer()
    point = Landmark(x=0.5, y=0.5)
    assert analyzer._calc_elbow_angle(point, point, point) == 180.0


def test_reversal_counting_ignores_jitter():
    analyzer = GestureAnalyzer()
    assert analyzer._count_reversals([0.0, 0.001, 0.0, 0.001], 0.01) == 0
    assert analyzer._count_reversals([0.0, 1.0, 0.0, 1.0], 0.5) == 2


# ------------------------------------------------------------------ skeleton

def test_pose_connections_reference_real_landmarks():
    for start, end in POSE_CONNECTIONS:
        assert 0 <= start < LANDMARK_COUNT
        assert 0 <= end < LANDMARK_COUNT
        assert start != end
