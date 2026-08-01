"""The d6a codec over every shipped motion, and the playback state machine."""

import os
import sqlite3
import threading
import time

import pytest

from ainex_api.board import BUS_SERVO_POSITION_MIN, BUS_SERVO_POSITION_MAX
from ainex_api.motion import MotionPlayer, Motion, MotionFrame, MOTION_STOP_TIMEOUT_S
from ainex_api.servos import BODY_SERVO_IDS

MOTION_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "ainex_api", "motions")
MOTION_NAMES = sorted(f[:-4] for f in os.listdir(MOTION_DIR) if f.endswith(".d6a"))


def _frame_count(name: str) -> int:
    conn = sqlite3.connect(os.path.join(MOTION_DIR, f"{name}.d6a"))
    try:
        return conn.execute("SELECT COUNT(*) FROM ActionGroup").fetchone()[0]
    finally:
        conn.close()


# Several tests need a motion long enough to still be running when observed;
# stand.d6a is a single frame and would finish first.
LONG_MOTION = max(MOTION_NAMES, key=_frame_count)


def test_the_repo_ships_motions():
    assert MOTION_NAMES


@pytest.mark.parametrize("name", MOTION_NAMES)
def test_every_shipped_motion_has_the_expected_schema(name):
    path = os.path.join(MOTION_DIR, f"{name}.d6a")
    conn = sqlite3.connect(path)
    try:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(ActionGroup)")]
    finally:
        conn.close()
    assert columns[0] == "Index"
    assert columns[1] == "Time"
    assert columns[2:] == [f"Servo{i}" for i in BODY_SERVO_IDS]


@pytest.mark.parametrize("name", MOTION_NAMES)
def test_every_shipped_motion_is_playable(board, name):
    """Every frame must survive the range checks bus_servo_set_position applies."""
    player = MotionPlayer(board, motion_dir=MOTION_DIR)
    motion = player.load(name)
    assert motion is not None, name
    assert motion.frames, name
    for f in motion.frames:
        assert len(f.positions) == len(BODY_SERVO_IDS), (name, f.frame_id)
        assert f.duration_ms > 0, (name, f.frame_id)
        for servo_id, pos in zip(BODY_SERVO_IDS, f.positions):
            assert BUS_SERVO_POSITION_MIN <= pos <= BUS_SERVO_POSITION_MAX, \
                (name, f.frame_id, servo_id, pos)


@pytest.mark.parametrize("name", MOTION_NAMES)
def test_frames_come_back_in_ascending_order(board, name):
    player = MotionPlayer(board, motion_dir=MOTION_DIR)
    ids = [f.frame_id for f in player.load(name).frames]
    assert ids == sorted(ids), name


def test_no_shipped_frame_outlasts_the_shutdown_timeout(board):
    """MOTION_STOP_TIMEOUT_S assumes the longest single frame is shorter than it."""
    player = MotionPlayer(board, motion_dir=MOTION_DIR)
    longest = max(f.duration_ms
                  for name in MOTION_NAMES
                  for f in player.load(name).frames)
    assert longest / 1000.0 * player.timing_multiplier < MOTION_STOP_TIMEOUT_S


def test_list_motions_matches_the_directory(board):
    player = MotionPlayer(board, motion_dir=MOTION_DIR)
    assert sorted(player.list_motions()) == MOTION_NAMES


def test_missing_motion_loads_as_none(board):
    player = MotionPlayer(board, motion_dir=MOTION_DIR)
    assert player.load("definitely_not_a_motion") is None
    assert player.play("definitely_not_a_motion") is False


def test_loaded_motions_are_cached(board):
    player = MotionPlayer(board, motion_dir=MOTION_DIR)
    first = player.load("stand")
    assert player.load("stand") is first
    player.clear_cache()
    assert player.load("stand") is not first


def test_total_duration_sums_the_frames():
    motion = Motion(name="m", frames=[MotionFrame(1, 100, [500] * 22),
                                      MotionFrame(2, 250, [500] * 22)])
    assert motion.total_duration_ms == 350
    assert motion.frame_count == 2


def test_playback_writes_one_packet_per_frame(board):
    player = MotionPlayer(board, motion_dir=MOTION_DIR)
    player.timing_multiplier = 0
    motion = player.load("stand")
    assert player.play("stand", blocking=True) is True
    assert len(board.port_fake.writes) == motion.frame_count


def test_playback_addresses_servos_one_to_twentytwo(board):
    player = MotionPlayer(board, motion_dir=MOTION_DIR)
    player.timing_multiplier = 0
    player.play("stand", blocking=True)
    data = board.port_fake.writes[-1][4:-1]
    ids = [data[4 + i * 3] for i in range(data[3])]
    assert ids == BODY_SERVO_IDS


def test_only_one_motion_plays_at_a_time(board):
    player = MotionPlayer(board, motion_dir=MOTION_DIR)
    player.timing_multiplier = 0.5
    assert player.play(LONG_MOTION, blocking=False) is True
    assert player.play("greet", blocking=False) is False
    player.stop()
    player.wait(timeout=MOTION_STOP_TIMEOUT_S)


def test_state_is_reported_and_cleared_around_playback(board):
    player = MotionPlayer(board, motion_dir=MOTION_DIR)
    player.timing_multiplier = 0.5
    assert player.current_motion is None
    player.play(LONG_MOTION, blocking=False)
    assert player.is_playing is True
    assert player.current_motion == LONG_MOTION
    player.stop()
    assert player.wait(timeout=MOTION_STOP_TIMEOUT_S) is True
    assert player.is_playing is False
    assert player.current_motion is None


def test_concurrent_play_calls_start_exactly_one_motion(board):
    """Regression for the check-then-set race in play()."""
    player = MotionPlayer(board, motion_dir=MOTION_DIR)
    player.timing_multiplier = 0.5
    started = []
    lock = threading.Lock()
    barrier = threading.Barrier(8)

    def race():
        barrier.wait()
        result = player.play(LONG_MOTION, blocking=False)
        with lock:
            started.append(result)

    threads = [threading.Thread(target=race) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    player.stop()
    player.wait(timeout=MOTION_STOP_TIMEOUT_S)
    assert started.count(True) == 1


def test_stop_ends_playback_early(board):
    player = MotionPlayer(board, motion_dir=MOTION_DIR)
    player.timing_multiplier = 0.5
    motion = player.load(LONG_MOTION)
    assert motion.frame_count > 1
    player.play(LONG_MOTION, blocking=False)
    time.sleep(0.05)
    player.stop()
    assert player.wait(timeout=MOTION_STOP_TIMEOUT_S) is True
    assert len(board.port_fake.writes) < motion.frame_count


def test_wait_reports_a_timeout(board):
    player = MotionPlayer(board, motion_dir=MOTION_DIR)
    player.timing_multiplier = 1.0
    player.play("stand", blocking=False)
    assert player.wait(timeout=0.01) is False
    player.stop()
    player.wait(timeout=MOTION_STOP_TIMEOUT_S)


def test_wait_returns_immediately_when_idle(board):
    player = MotionPlayer(board, motion_dir=MOTION_DIR)
    assert player.wait(timeout=0.01) is True


def test_negative_timing_multiplier_is_rejected_at_assignment(board):
    player = MotionPlayer(board, motion_dir=MOTION_DIR)
    with pytest.raises(ValueError):
        player.timing_multiplier = -1.0
    assert player.timing_multiplier == 0.8


def test_completion_callback_runs(board):
    player = MotionPlayer(board, motion_dir=MOTION_DIR)
    player.timing_multiplier = 0
    done = threading.Event()
    player.play("stand", blocking=False, on_complete=done.set)
    assert done.wait(timeout=MOTION_STOP_TIMEOUT_S)


def test_failing_callback_does_not_leave_playback_stuck(board):
    player = MotionPlayer(board, motion_dir=MOTION_DIR)
    player.timing_multiplier = 0

    def boom():
        raise RuntimeError("callback blew up")

    player.play("stand", blocking=False, on_complete=boom)
    assert player.wait(timeout=MOTION_STOP_TIMEOUT_S) is True
    assert player.play("greet", blocking=False) is True
    player.stop()
    player.wait(timeout=MOTION_STOP_TIMEOUT_S)


def test_set_servos_direct_enforces_travel(board):
    player = MotionPlayer(board, motion_dir=MOTION_DIR)
    with pytest.raises(ValueError):
        player.set_servos_direct(500, [[23, 9000]])
    assert board.port_fake.writes == []


def test_set_servos_direct_converts_milliseconds(board):
    import struct
    player = MotionPlayer(board, motion_dir=MOTION_DIR)
    player.set_servos_direct(1500, [[13, 500]])
    data = board.port_fake.writes[-1][4:-1]
    assert struct.unpack("<H", data[1:3])[0] == 1500


def test_corrupt_motion_file_loads_as_none(board, tmp_path):
    bad = tmp_path / "broken.d6a"
    bad.write_bytes(b"this is not a sqlite database")
    player = MotionPlayer(board, motion_dir=str(tmp_path))
    assert player.load("broken") is None


def test_frames_are_read_in_index_order_regardless_of_insert_order(board, tmp_path):
    """The query orders explicitly instead of trusting the rowid scan."""
    path = tmp_path / "scrambled.d6a"
    conn = sqlite3.connect(str(path))
    columns = ", ".join(f"Servo{i} INT" for i in BODY_SERVO_IDS)
    conn.execute(f"CREATE TABLE ActionGroup ([Index] INTEGER PRIMARY KEY, Time INT, {columns})")
    placeholders = ", ".join("?" * (2 + len(BODY_SERVO_IDS)))
    for index in (3, 1, 2):
        conn.execute(f"INSERT INTO ActionGroup VALUES ({placeholders})",
                     [index, 100] + [500] * len(BODY_SERVO_IDS))
    conn.commit()
    conn.close()

    player = MotionPlayer(board, motion_dir=str(tmp_path))
    assert [f.frame_id for f in player.load("scrambled").frames] == [1, 2, 3]
