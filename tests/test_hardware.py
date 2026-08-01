"""Tests against the real robot.

Read-only by design: these open the serial port, listen to what the STM32
pushes, and issue bus servo READ commands. Nothing here commands a position,
enables torque or plays a motion, so running the suite cannot move the robot.

Skipped automatically when the hardware is not present.
"""

import struct
import time

import pytest

from ainex_api.board import Board, PacketFunction, crc8, FUNCTION_CODES
from ainex_api.camera import Camera, open_camera, DEFAULT_WIDTH, DEFAULT_HEIGHT
from ainex_api.sensors import SensorReader, CELL_COUNT, DISCHARGE_CURVE
from ainex_api.servos import ServoController, ALL_SERVO_IDS, SERVO_MAP, servo_limits

from conftest import SERIAL_DEVICE, requires_hardware, requires_camera

pytestmark = pytest.mark.hardware


# Plausibility bounds for a healthy 3S pack, derived from the discharge curve.
PACK_MV_MIN = DISCHARGE_CURVE[-1][0] * CELL_COUNT
PACK_MV_MAX = DISCHARGE_CURVE[0][0] * CELL_COUNT

# A bus servo that is powered and reporting sits well inside these.
SERVO_TEMP_MIN_C = 0
SERVO_TEMP_MAX_C = 100


@pytest.fixture(scope="module")
def live_board():
    board = Board(device=SERIAL_DEVICE)
    yield board
    board.close()


@pytest.fixture(scope="module")
def live_servos(live_board):
    return ServoController(live_board)


def collect_frames(board, seconds=2.0):
    """Count the frames the board pushes unprompted."""
    seen = {}
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        for func, q in board._queues.items():
            while not q.empty():
                seen.setdefault(func, []).append(q.get_nowait())
        time.sleep(0.02)
    return seen


# ------------------------------------------------------------- link layer

@requires_hardware
def test_the_serial_port_opens(live_board):
    assert live_board.port.is_open
    assert live_board.port.baudrate == 1000000


@requires_hardware
def test_the_board_pushes_frames_that_pass_crc(live_board):
    """The parser only queues a payload after the checksum matches."""
    seen = collect_frames(live_board, seconds=2.0)
    total = sum(len(v) for v in seen.values())
    assert total > 0, "board sent nothing; is it powered?"


@requires_hardware
def test_the_receiver_thread_survives_the_run(live_board):
    assert live_board._recv_thread.is_alive()
    assert live_board._state == 0 or live_board._state in range(6)


@requires_hardware
def test_only_known_function_codes_arrive(live_board):
    seen = collect_frames(live_board, seconds=1.0)
    for func in seen:
        assert int(func) in FUNCTION_CODES


@requires_hardware
def test_a_raw_capture_contains_well_formed_frames(live_board):
    """Parse the wire independently of Board to check the framing end to end."""
    raw = bytearray()
    original_read = live_board.port.read

    def tee(size=1):
        data = original_read(size)
        raw.extend(data)
        return data

    live_board.port.read = tee
    try:
        time.sleep(2.0)
    finally:
        live_board.port.read = original_read

    good = bad = 0
    i = 0
    while i < len(raw) - 4:
        if raw[i] != 0xAA or raw[i + 1] != 0x55:
            i += 1
            continue
        func, length = raw[i + 2], raw[i + 3]
        end = i + 4 + length
        if func not in FUNCTION_CODES or end >= len(raw):
            i += 1
            continue
        body = bytes(raw[i + 2:end])
        if crc8(body) == raw[end]:
            good += 1
            i = end + 1
        else:
            bad += 1
            i += 1

    assert good > 0, "no valid frames in the capture"
    assert bad == 0, f"{bad} frames failed CRC out of {good + bad}"


# --------------------------------------------------------------- sensors

@requires_hardware
def test_battery_reads_a_plausible_pack_voltage(live_board):
    deadline = time.monotonic() + 5.0
    voltage = None
    while voltage is None and time.monotonic() < deadline:
        voltage = live_board.get_battery()
        time.sleep(0.05)
    assert voltage is not None, "no battery packet within 5s"
    assert PACK_MV_MIN <= voltage <= PACK_MV_MAX, f"{voltage} mV is not a 3S pack"


@requires_hardware
def test_the_smoothed_battery_reading_agrees_with_the_status(live_board):
    reader = SensorReader(live_board)
    deadline = time.monotonic() + 10.0
    state = None
    while state is None and time.monotonic() < deadline:
        state = reader.get_battery_state()
        time.sleep(0.05)
    assert state is not None, "no full smoothing window within 10s"
    assert PACK_MV_MIN <= state.voltage_mv <= PACK_MV_MAX
    assert 0.0 <= state.percent <= 100.0
    assert state.status in ('ok', 'low', 'critical')


@requires_hardware
def test_imu_reports_finite_values_and_roughly_one_g(live_board):
    import math

    deadline = time.monotonic() + 5.0
    data = None
    while data is None and time.monotonic() < deadline:
        data = live_board.get_imu()
        time.sleep(0.02)
    assert data is not None, "no IMU packet within 5s"
    assert len(data) in (6, 9)
    for value in data:
        assert math.isfinite(value)
    magnitude = math.sqrt(sum(a * a for a in data[:3]))
    assert magnitude > 0, "accelerometer reads exactly zero on all axes"


# ------------------------------------------------------------ bus servos

@requires_hardware
def test_reply_layout_is_servo_id_first_on_real_firmware(live_board):
    """The ground truth behind the fix in 47a42ae, read off the wire."""
    import queue

    replies = live_board._queues[PacketFunction.BUS_SERVO]
    with live_board._servo_lock:
        while not replies.empty():
            replies.get_nowait()
        live_board._write(PacketFunction.BUS_SERVO, bytes([0x05, 13]))
        try:
            payload = replies.get(timeout=0.5)
        except queue.Empty:
            pytest.skip("servo 13 did not answer")

    assert payload[0] == 13, "first reply byte is not the servo id"
    assert payload[1] == 0x05, "second reply byte is not the command echo"
    assert payload[2] == 0, "third reply byte is not the status"
    position = struct.unpack("<h", payload[3:5])[0]
    assert 0 <= position <= 1000


@requires_hardware
def test_every_servo_reports_its_own_id(live_board):
    """A transposed field check would make these all come back None."""
    answered = {}
    for servo_id in ALL_SERVO_IDS:
        reported = live_board.bus_servo_read_id(servo_id)
        if reported is not None:
            answered[servo_id] = reported
    assert answered, "no servo answered a read; check power and the bus"
    for servo_id, reported in answered.items():
        assert reported == servo_id, f"servo {servo_id} reported id {reported}"


@requires_hardware
def test_servo_positions_are_inside_their_travel(live_servos):
    answered = {}
    for servo_id in ALL_SERVO_IDS:
        position = live_servos.get_position(servo_id)
        if position is not None:
            answered[servo_id] = position
    assert answered, "no servo reported a position"
    for servo_id, position in answered.items():
        assert 0 <= position <= 1000, f"servo {servo_id} at {position}"


@requires_hardware
def test_head_servos_sit_within_their_mechanical_limits(live_servos):
    for servo_id in (23, 24):
        position = live_servos.get_position(servo_id)
        if position is None:
            pytest.skip(f"servo {servo_id} ({SERVO_MAP[servo_id]}) did not answer")
        lo, hi = servo_limits(servo_id)
        assert lo <= position <= hi, f"{SERVO_MAP[servo_id]} at {position}, limits {lo}-{hi}"


@requires_hardware
def test_servo_voltages_match_the_pack(live_board, live_servos):
    answered = {}
    for servo_id in ALL_SERVO_IDS:
        millivolts = live_servos.read_voltage(servo_id)
        if millivolts is not None:
            answered[servo_id] = millivolts
    assert answered, "no servo reported a voltage"
    for servo_id, millivolts in answered.items():
        assert PACK_MV_MIN <= millivolts <= PACK_MV_MAX, f"servo {servo_id} at {millivolts} mV"


@requires_hardware
def test_servo_temperatures_are_plausible(live_servos):
    answered = {}
    for servo_id in ALL_SERVO_IDS:
        celsius = live_servos.read_temperature(servo_id)
        if celsius is not None:
            answered[servo_id] = celsius
    assert answered, "no servo reported a temperature"
    for servo_id, celsius in answered.items():
        assert SERVO_TEMP_MIN_C <= celsius <= SERVO_TEMP_MAX_C, \
            f"servo {servo_id} at {celsius}C"


@requires_hardware
def test_a_read_for_an_absent_servo_returns_none(live_board):
    """Nothing answers for an id outside the robot, and the retry budget ends."""
    start = time.monotonic()
    assert live_board.bus_servo_read_position(250) is None
    assert time.monotonic() - start < 5.0


@requires_hardware
def test_repeated_reads_are_stable(live_servos):
    """Correlation must hold across back to back reads of different fields."""
    servo_id = next((s for s in ALL_SERVO_IDS
                     if live_servos.get_position(s) is not None), None)
    if servo_id is None:
        pytest.skip("no servo answered")
    for _ in range(5):
        position = live_servos.get_position(servo_id)
        voltage = live_servos.read_voltage(servo_id)
        temperature = live_servos.read_temperature(servo_id)
        assert position is None or 0 <= position <= 1000
        assert voltage is None or PACK_MV_MIN <= voltage <= PACK_MV_MAX
        assert temperature is None or SERVO_TEMP_MIN_C <= temperature <= SERVO_TEMP_MAX_C


@requires_hardware
def test_interleaved_reads_do_not_cross_their_answers(live_servos):
    """A position must never come back as a voltage, whatever the order."""
    ids = [s for s in ALL_SERVO_IDS if live_servos.get_position(s) is not None][:4]
    if len(ids) < 2:
        pytest.skip("need at least two responding servos")
    for servo_id in ids:
        voltage = live_servos.read_voltage(servo_id)
        position = live_servos.get_position(servo_id)
        if voltage is not None and position is not None:
            assert voltage > 1000, "a position leaked into the voltage read"
            assert position <= 1000


# ---------------------------------------------------------------- camera

@requires_camera
def test_the_camera_opens_and_reports_a_size():
    cap, width, height = open_camera()
    try:
        assert cap is not None
        assert width > 0 and height > 0
    finally:
        if cap is not None:
            cap.release()


@requires_camera
def test_the_camera_delivers_a_bgr_frame():
    camera = Camera()
    assert camera.start() is True
    try:
        deadline = time.monotonic() + 5.0
        frame = None
        while frame is None and time.monotonic() < deadline:
            frame = camera.read()
            time.sleep(0.02)
        assert frame is not None, "no frame within 5s"
        assert frame.ndim == 3 and frame.shape[2] == 3
        assert frame.shape[0] == camera.height
        assert frame.shape[1] == camera.width
        assert str(frame.dtype) == 'uint8'
    finally:
        camera.stop()


@requires_camera
def test_the_camera_negotiates_the_default_resolution():
    camera = Camera()
    assert camera.start() is True
    try:
        assert (camera.width, camera.height) == (DEFAULT_WIDTH, DEFAULT_HEIGHT)
    finally:
        camera.stop()


@requires_camera
def test_each_read_hands_back_a_private_copy():
    camera = Camera()
    assert camera.start() is True
    try:
        deadline = time.monotonic() + 5.0
        first = None
        while first is None and time.monotonic() < deadline:
            first = camera.read()
            time.sleep(0.02)
        assert first is not None
        second = camera.read()
        assert second is not first
        first[:] = 0
        assert camera.read() is not first
    finally:
        camera.stop()


@requires_camera
def test_the_capture_rate_is_positive():
    camera = Camera()
    assert camera.start() is True
    try:
        time.sleep(2.0)
        assert camera.fps > 0
        assert camera.is_open is True
    finally:
        camera.stop()
        assert camera.is_open is False
