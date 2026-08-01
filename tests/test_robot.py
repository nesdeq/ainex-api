"""The facade: construction, address parsing, lifecycle and reported state."""

import pytest

from ainex_api.robot import (
    Robot, _parse_vision_address, _LEFT_ARM_NEUTRAL, _LEFT_ARM_RAISED,
    _RIGHT_ARM_NEUTRAL, _RIGHT_ARM_RAISED,
)
from ainex_api.servos import check_position, SERVO_MAP


class StubVision:
    def __init__(self, camera, *args, **kwargs):
        self.camera = camera
        self.stopped = False

    def stop(self):
        self.stopped = True

    @property
    def has_mediapipe(self):
        return False


@pytest.fixture
def robot(fake_serial, monkeypatch):
    """A Robot on a fake serial port, without loading a pose model."""
    monkeypatch.setattr("ainex_api.robot.VisionSystem", StubVision)
    r = Robot(device="/dev/fake")
    r.board.port_fake = fake_serial["port"]
    yield r
    if r.board.port.is_open:
        r.close()


# ------------------------------------------------------- address parsing

def test_valid_address_parses():
    assert _parse_vision_address("192.168.0.3:9999") == ("192.168.0.3", 9999)
    assert _parse_vision_address("localhost:1") == ("localhost", 1)
    assert _parse_vision_address("host:65535") == ("host", 65535)


@pytest.mark.parametrize("bad", [
    "nohost", ":9999", "host:", "host:abc", "host:0", "host:65536",
    "host:-1", "", ":", "host:99 99",
])
def test_malformed_address_raises(bad):
    with pytest.raises(ValueError):
        _parse_vision_address(bad)


def test_a_bad_address_never_opens_the_serial_port(fake_serial, monkeypatch):
    """Regression: the address used to be parsed after the board was live."""
    monkeypatch.setattr("ainex_api.robot.VisionSystem", StubVision)
    with pytest.raises(ValueError):
        Robot(device="/dev/fake", remote_vision="garbage")
    assert "port" not in fake_serial


# ------------------------------------------------------------ lifecycle

def test_subsystems_are_wired_to_one_board(robot):
    assert robot.servos.board is robot.board
    assert robot.head.board is robot.board
    assert robot.motion.board is robot.board
    assert robot.sensors.board is robot.board
    assert robot.peripherals.board is robot.board


def test_local_mode_is_the_default(robot):
    assert robot._vision_mode == 'local'
    assert isinstance(robot.vision, StubVision)


def test_close_releases_the_port_and_the_vision_system(robot):
    assert robot.close() is True
    assert robot.board.port.is_open is False
    assert robot.vision.stopped is True


def test_close_reports_a_motion_that_would_not_stop(robot, monkeypatch, capsys):
    monkeypatch.setattr(robot.motion, "wait", lambda timeout=None: False)
    assert robot.close() is False
    assert "did not stop" in capsys.readouterr().out


def test_context_manager_closes_on_exit(fake_serial, monkeypatch):
    monkeypatch.setattr("ainex_api.robot.VisionSystem", StubVision)
    with Robot(device="/dev/fake") as r:
        board = r.board
    assert board.port.is_open is False


def test_failed_subsystem_construction_hands_the_port_back(fake_serial, monkeypatch):
    """Regression: a partial init used to strand an open port and a live thread."""
    def explode(*args, **kwargs):
        raise RuntimeError("no pose model")

    monkeypatch.setattr("ainex_api.robot.VisionSystem", explode)
    with pytest.raises(RuntimeError):
        Robot(device="/dev/fake")
    assert fake_serial["port"].is_open is False


# ----------------------------------------------------------- last pose

def test_last_pose_starts_empty(robot):
    assert robot.status()['last_pose'] is None


def test_every_motion_updates_the_last_pose(robot):
    robot.motion.timing_multiplier = 0
    assert robot.greet() is True
    assert robot.status()['last_pose'] == 'greet'
    assert robot.stand() is True
    assert robot.status()['last_pose'] == 'stand'


def test_a_refused_motion_does_not_claim_the_pose(robot, monkeypatch):
    """Regression: stand() used to record the pose even when play() refused."""
    robot.motion.timing_multiplier = 0
    robot.greet()
    monkeypatch.setattr(robot.motion, "play", lambda *a, **kw: False)
    assert robot.stand() is False
    assert robot.status()['last_pose'] == 'greet'


def test_zero_records_its_own_pose(robot, monkeypatch):
    monkeypatch.setattr("ainex_api.robot.time.sleep", lambda _s: None)
    robot.zero(duration=0.1)
    assert robot.status()['last_pose'] == 'zero'


def test_zero_leaves_the_head_alone(robot, monkeypatch):
    monkeypatch.setattr("ainex_api.robot.time.sleep", lambda _s: None)
    robot.zero(duration=0.1)
    data = robot.board.port_fake.writes[-1][4:-1]
    ids = [data[4 + i * 3] for i in range(data[3])]
    assert 23 not in ids and 24 not in ids


# ------------------------------------------------------------ arm poses

@pytest.mark.parametrize("pose", [_LEFT_ARM_NEUTRAL, _LEFT_ARM_RAISED,
                                  _RIGHT_ARM_NEUTRAL, _RIGHT_ARM_RAISED])
def test_arm_pose_constants_are_reachable_positions(pose):
    for servo_id, position in pose.items():
        assert servo_id in SERVO_MAP
        check_position(servo_id, position)


def test_left_and_right_arm_poses_touch_disjoint_servos():
    assert set(_LEFT_ARM_NEUTRAL) == set(_LEFT_ARM_RAISED)
    assert set(_RIGHT_ARM_NEUTRAL) == set(_RIGHT_ARM_RAISED)
    assert not set(_LEFT_ARM_NEUTRAL) & set(_RIGHT_ARM_NEUTRAL)


def test_raising_one_arm_lowers_the_other(robot):
    robot.raise_left_arm(duration=0.1)
    data = robot.board.port_fake.writes[-1][4:-1]
    ids = {data[4 + i * 3] for i in range(data[3])}
    assert ids == set(_LEFT_ARM_RAISED) | set(_RIGHT_ARM_NEUTRAL)


def test_raising_both_arms_addresses_both_sides(robot):
    robot.raise_both_arms(duration=0.1)
    data = robot.board.port_fake.writes[-1][4:-1]
    ids = {data[4 + i * 3] for i in range(data[3])}
    assert ids == set(_LEFT_ARM_RAISED) | set(_RIGHT_ARM_RAISED)


# --------------------------------------------------------------- status

def test_status_reports_the_expected_keys(robot):
    status = robot.status()
    for key in ('connected', 'battery', 'battery_voltage', 'battery_status',
                'last_pose', 'motion_playing', 'current_motion',
                'available_motions', 'vision_mode'):
        assert key in status
    assert status['connected'] is True
    assert status['vision_mode'] == 'local'
    assert status['available_motions']


def test_status_survives_an_unknown_battery(robot, capsys):
    assert robot.status()['battery'] is None
    robot.print_status()
    assert "Battery: N/A" in capsys.readouterr().out
