"""Wire protocol: checksum, framing, parser, packet layouts, reply correlation."""

import struct
import threading
import time

import pytest

from ainex_api.board import (
    Board, PacketFunction, KeyEvent, crc8, CRC8_TABLE, FUNCTION_CODES,
    BROADCAST_SERVO_ID, BUS_SERVO_POSITION_MIN, BUS_SERVO_POSITION_MAX,
    PWM_SERVO_POSITION_MIN, PWM_SERVO_POSITION_MAX,
    MOTOR_SPEED_MIN, MOTOR_SPEED_MAX, SYS_BATTERY,
    _u8, _u16, _i8,
)

from conftest import FakeSerial, ServoModel, frame


# ---------------------------------------------------------------- checksum

def test_crc8_matches_published_maxim_check_value():
    assert crc8(b"123456789") == 0xA1


def test_crc8_table_is_a_complete_byte_permutation():
    assert len(CRC8_TABLE) == 256
    assert sorted(CRC8_TABLE) == list(range(256))


def test_crc8_of_empty_input_is_zero():
    assert crc8(b"") == 0


@pytest.mark.parametrize("payload", [b"\x00", b"\xff" * 8, bytes(range(64))])
def test_crc8_always_fits_a_byte(payload):
    assert 0 <= crc8(payload) <= 0xFF


# ---------------------------------------------------------------- framing

def test_written_frame_has_preamble_function_length_and_checksum(board):
    board._write(PacketFunction.LED, b"\x01\x02")
    sent = board.port_fake.writes[-1]
    assert sent[:2] == b"\xAA\x55"
    assert sent[2] == int(PacketFunction.LED)
    assert sent[3] == 2
    assert sent[4:6] == b"\x01\x02"
    assert sent[6] == crc8(sent[2:6])


def test_checksum_excludes_the_preamble(board):
    board._write(PacketFunction.SYS, b"\x09")
    sent = board.port_fake.writes[-1]
    assert sent[-1] == crc8(sent[2:-1])
    assert sent[-1] != crc8(sent[:-1])


def test_write_and_parse_round_trip(board):
    board._write(PacketFunction.IMU, b"payload")
    sent = board.port_fake.writes[-1]
    for byte in sent:
        board._parse_byte(byte)
    assert board._queues[PacketFunction.IMU].get_nowait() == b"payload"


# ---------------------------------------------------------------- parser

def feed_and_settle(board, data, timeout=1.0):
    """Push bytes at the receiver thread and wait for it to drain them."""
    board.port_fake.feed(data)
    deadline = time.monotonic() + timeout
    while board.port_fake.in_waiting and time.monotonic() < deadline:
        time.sleep(0.005)
    time.sleep(0.05)


def test_valid_frame_reaches_its_queue(board):
    feed_and_settle(board, frame(int(PacketFunction.KEY), b"\x01\x20"))
    assert board._queues[PacketFunction.KEY].get_nowait() == b"\x01\x20"


def test_frame_with_bad_checksum_is_dropped(board):
    good = bytearray(frame(int(PacketFunction.KEY), b"\x01\x20"))
    good[-1] ^= 0xFF
    feed_and_settle(board, bytes(good))
    assert board._queues[PacketFunction.KEY].empty()


def test_leading_garbage_does_not_stop_the_next_frame(board):
    feed_and_settle(board, b"\x00\xde\xad\xbe\xef" +
                    frame(int(PacketFunction.KEY), b"\x02\x20"))
    assert board._queues[PacketFunction.KEY].get_nowait() == b"\x02\x20"


def test_repeated_preamble_byte_still_syncs(board):
    """Noise ending in 0xAA must not consume the frame that follows it."""
    feed_and_settle(board, b"\xAA\xAA\xAA" + frame(int(PacketFunction.KEY), b"\x09\x20"))
    assert board._queues[PacketFunction.KEY].get_nowait() == b"\x09\x20"


def test_zero_length_payload_frame_parses(board):
    feed_and_settle(board, frame(int(PacketFunction.SYS), b""))
    assert board._queues[PacketFunction.SYS].get_nowait() == b""


@pytest.mark.parametrize("bogus_func", [10, 11, 12, 200, 255])
def test_unknown_function_code_does_not_eat_the_following_frame(board, bogus_func):
    assert bogus_func not in FUNCTION_CODES
    poison = b"\xAA\x55" + bytes([bogus_func, 2]) + b"\x01\x02\x00"
    feed_and_settle(board, poison + frame(int(PacketFunction.KEY), b"\x03\x20"))
    assert board._queues[PacketFunction.KEY].get_nowait() == b"\x03\x20"


def test_sbus_frame_is_consumed_without_desyncing_the_next(board):
    sbus = frame(int(PacketFunction.SBUS), b"\x00" * 36)
    feed_and_settle(board, sbus + frame(int(PacketFunction.KEY), b"\x04\x20"))
    assert PacketFunction.SBUS not in board._queues
    assert board._queues[PacketFunction.KEY].get_nowait() == b"\x04\x20"


def test_frame_split_across_reads_still_parses(board):
    whole = frame(int(PacketFunction.KEY), b"\x05\x20")
    for i in range(len(whole)):
        board.port_fake.feed(whole[i:i + 1])
        time.sleep(0.002)
    time.sleep(0.1)
    assert board._queues[PacketFunction.KEY].get_nowait() == b"\x05\x20"


def test_queue_overflow_drops_instead_of_blocking(board):
    for i in range(10):
        board.port_fake.feed(frame(int(PacketFunction.KEY), bytes([i, 0x20])))
    time.sleep(0.2)
    q = board._queues[PacketFunction.KEY]
    assert q.qsize() <= 2
    assert board._recv_thread.is_alive()


def test_truncated_frame_does_not_swallow_the_next_one(board):
    """A frame that claims more payload than it delivers must not desync."""
    truncated = b"\xAA\x55" + bytes([int(PacketFunction.KEY), 8]) + b"\x01\x02"
    feed_and_settle(board, truncated + frame(int(PacketFunction.KEY), b"\x06\x20"))
    # The truncated header consumes the next frame's bytes as payload, so the
    # recovery point is the frame after that.
    feed_and_settle(board, frame(int(PacketFunction.KEY), b"\x07\x20"))
    assert board._queues[PacketFunction.KEY].get_nowait() == b"\x07\x20"


def test_read_error_in_the_receiver_leaves_the_parser_usable(board):
    original_read = board.port_fake.read
    calls = {"n": 0}

    def flaky(size=1):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("transient serial fault")
        return original_read(size)

    board.port_fake.read = flaky
    time.sleep(0.05)
    feed_and_settle(board, frame(int(PacketFunction.KEY), b"\x08\x20"))
    assert board._state == 0
    assert board._recv_thread.is_alive()
    assert board._queues[PacketFunction.KEY].get_nowait() == b"\x08\x20"


# ---------------------------------------------- bus servo reply correlation

def test_read_position_returns_the_value_from_the_matching_reply(board):
    board.port_fake.responder = ServoModel({(13, 0x05): 835})
    assert board.bus_servo_read_position(13) == 835


def test_reply_layout_is_servo_id_then_command(board):
    """Regression for the transposed field check fixed in 47a42ae."""
    model = ServoModel({(13, 0x05): 835})
    board.port_fake.responder = model
    board.bus_servo_read_position(13)
    reply = model(board.port_fake.writes[-1])
    payload = reply[4:-1]
    assert payload[0] == 13
    assert payload[1] == 0x05
    assert payload[2] == 0


def test_reply_from_a_different_servo_is_rejected(board):
    def wrong_servo(packet):
        if packet[2] != int(PacketFunction.BUS_SERVO):
            return None
        body = struct.pack("<BBb", 5, 0x05, 0) + struct.pack("<h", 999)
        return frame(int(PacketFunction.BUS_SERVO), body)

    board.port_fake.responder = wrong_servo
    assert board.bus_servo_read_position(13) is None


def test_voltage_reply_is_not_returned_as_a_position(board):
    """Both are five bytes, so only the command echo separates them."""
    def voltage_only(packet):
        if packet[2] != int(PacketFunction.BUS_SERVO):
            return None
        body = struct.pack("<BBb", 13, 0x07, 0) + struct.pack("<H", 11800)
        return frame(int(PacketFunction.BUS_SERVO), body)

    board.port_fake.responder = voltage_only
    assert board.bus_servo_read_position(13) is None


def test_nonzero_status_is_rejected(board):
    def failing(packet):
        if packet[2] != int(PacketFunction.BUS_SERVO):
            return None
        body = struct.pack("<BBb", 13, 0x05, 1) + struct.pack("<h", 500)
        return frame(int(PacketFunction.BUS_SERVO), body)

    board.port_fake.responder = failing
    assert board.bus_servo_read_position(13) is None


def test_silent_bus_returns_none_after_the_retry_budget(board):
    board.port_fake.responder = lambda packet: None
    start = time.monotonic()
    assert board.bus_servo_read_position(13) is None
    assert len(board.port_fake.writes) == board._retry_count
    assert time.monotonic() - start < 5.0


def test_broadcast_read_accepts_any_responding_servo(board):
    board.port_fake.responder = ServoModel({(7, 0x12): 7})

    def any_servo(packet):
        body = struct.pack("<BBb", 7, 0x12, 0) + struct.pack("<B", 7)
        return frame(int(PacketFunction.BUS_SERVO), body)

    board.port_fake.responder = any_servo
    assert board.bus_servo_read_id(BROADCAST_SERVO_ID) == 7


def test_broadcast_read_still_checks_the_command_echo(board):
    def wrong_cmd(packet):
        body = struct.pack("<BBb", 7, 0x05, 0) + struct.pack("<B", 7)
        return frame(int(PacketFunction.BUS_SERVO), body)

    board.port_fake.responder = wrong_cmd
    assert board.bus_servo_read_id(BROADCAST_SERVO_ID) is None


def test_stale_reply_is_drained_before_each_attempt(board):
    board._queues[PacketFunction.BUS_SERVO].put_nowait(b"\xff\xff\xff\xff\xff")
    board.port_fake.responder = ServoModel({(13, 0x05): 700})
    assert board.bus_servo_read_position(13) == 700


@pytest.mark.parametrize("reader,cmd,value,expected", [
    ("bus_servo_read_position", 0x05, 835, 835),
    ("bus_servo_read_voltage", 0x07, 11888, 11888),
    ("bus_servo_read_temperature", 0x09, 38, 38),
    ("bus_servo_read_id", 0x12, 13, 13),
    ("bus_servo_read_offset", 0x22, -21, -21),
])
def test_every_read_decodes_its_own_format(board, reader, cmd, value, expected):
    board.port_fake.responder = ServoModel({(13, cmd): value})
    assert getattr(board, reader)(13) == expected


def test_cached_position_read_does_not_touch_the_bus(board):
    board.bus_servo_set_position(0.5, [[13, 640]])
    writes_before = len(board.port_fake.writes)
    assert board.bus_servo_read_position(13, use_cache=True) == 640
    assert len(board.port_fake.writes) == writes_before


# ---------------------------------------------------------- packet layouts

def payload_of(packet: bytes) -> bytes:
    return packet[4:-1]


def test_set_position_packet_layout(board):
    board.bus_servo_set_position(1.5, [[13, 500], [14, 640]])
    data = payload_of(board.port_fake.writes[-1])
    assert data[0] == 0x01
    assert struct.unpack("<H", data[1:3])[0] == 1500
    assert data[3] == 2
    assert struct.unpack("<BH", data[4:7]) == (13, 500)
    assert struct.unpack("<BH", data[7:10]) == (14, 640)


def test_stop_packet_layout(board):
    board.bus_servo_stop([1, 2, 3])
    data = payload_of(board.port_fake.writes[-1])
    assert data == bytes([0x03, 3, 1, 2, 3])


@pytest.mark.parametrize("enable,cmd", [(True, 0x0B), (False, 0x0C)])
def test_torque_packet_layout(board, enable, cmd):
    board.bus_servo_enable_torque(13, enable)
    assert payload_of(board.port_fake.writes[-1]) == bytes([cmd, 13])


def test_set_offset_packet_layout(board):
    board.bus_servo_set_offset(13, -21)
    assert payload_of(board.port_fake.writes[-1]) == struct.pack("<BBb", 0x20, 13, -21)


def test_save_offset_packet_layout(board):
    board.bus_servo_save_offset(13)
    assert payload_of(board.port_fake.writes[-1]) == bytes([0x24, 13])


def test_angle_limit_packet_layout(board):
    board.bus_servo_set_angle_limit(13, 100, 900)
    assert payload_of(board.port_fake.writes[-1]) == struct.pack("<BBHH", 0x30, 13, 100, 900)


def test_read_request_is_command_then_servo_id(board):
    board.port_fake.responder = lambda packet: None
    board.bus_servo_read_position(13)
    assert payload_of(board.port_fake.writes[-1]) == bytes([0x05, 13])


def test_pwm_servo_packet_layout(board):
    board.pwm_servo_set_position(0.5, [[1, 1500]])
    data = payload_of(board.port_fake.writes[-1])
    assert data[0] == 0x01
    assert struct.unpack("<H", data[1:3])[0] == 500
    assert data[3] == 1
    assert struct.unpack("<BH", data[4:7]) == (1, 1500)


def test_buzzer_packet_layout(board):
    board.set_buzzer(2400, 0.1, 0.9, 3)
    assert payload_of(board.port_fake.writes[-1]) == struct.pack("<HHHH", 2400, 100, 900, 3)


def test_led_packet_layout(board):
    board.set_led(0.5, 0.25, 3, led_id=2)
    assert payload_of(board.port_fake.writes[-1]) == struct.pack("<BHHH", 2, 500, 250, 3)


def test_motor_packet_uses_zero_based_ids(board):
    board.set_motor_speed([(1, 0.5), (2, -0.5)])
    data = payload_of(board.port_fake.writes[-1])
    assert data[0:2] == bytes([0x01, 2])
    assert struct.unpack("<Bf", data[2:7]) == (0, 0.5)
    assert struct.unpack("<Bf", data[7:12]) == (1, -0.5)


# ------------------------------------------------------------ range checks

@pytest.mark.parametrize("bad", [BUS_SERVO_POSITION_MIN - 1, BUS_SERVO_POSITION_MAX + 1, 5000, -1])
def test_bus_servo_position_out_of_range_raises(board, bad):
    with pytest.raises(ValueError):
        board.bus_servo_set_position(0.5, [[13, bad]])


@pytest.mark.parametrize("edge", [BUS_SERVO_POSITION_MIN, BUS_SERVO_POSITION_MAX])
def test_bus_servo_position_edges_are_allowed(board, edge):
    board.bus_servo_set_position(0.5, [[13, edge]])


@pytest.mark.parametrize("bad", [PWM_SERVO_POSITION_MIN - 1, PWM_SERVO_POSITION_MAX + 1])
def test_pwm_servo_position_out_of_range_raises(board, bad):
    with pytest.raises(ValueError):
        board.pwm_servo_set_position(0.5, [[1, bad]])


@pytest.mark.parametrize("bad_duration", [70.0, -1.0])
def test_duration_that_would_wrap_the_field_raises(board, bad_duration):
    with pytest.raises(ValueError):
        board.bus_servo_set_position(bad_duration, [[13, 500]])


def test_position_cache_is_not_updated_by_a_rejected_write(board):
    board.bus_servo_set_position(0.5, [[13, 600]])
    with pytest.raises(ValueError):
        board.bus_servo_set_position(0.5, [[13, 5000]])
    assert board.bus_servo_read_position(13, use_cache=True) == 600


@pytest.mark.parametrize("lo,hi", [(-1, 900), (100, 1001), (900, 100)])
def test_invalid_angle_limits_raise(board, lo, hi):
    with pytest.raises(ValueError):
        board.bus_servo_set_angle_limit(13, lo, hi)


def test_equal_angle_limits_are_allowed(board):
    board.bus_servo_set_angle_limit(13, 500, 500)


@pytest.mark.parametrize("bad", [MOTOR_SPEED_MIN - 0.01, MOTOR_SPEED_MAX + 0.01, 5.0])
def test_motor_speed_out_of_range_raises(board, bad):
    with pytest.raises(ValueError):
        board.set_motor_speed([(1, bad)])


@pytest.mark.parametrize("edge", [MOTOR_SPEED_MIN, 0.0, MOTOR_SPEED_MAX])
def test_motor_speed_edges_are_allowed(board, edge):
    board.set_motor_speed([(1, edge)])


@pytest.mark.parametrize("bad_id", [256, -1])
def test_servo_id_outside_a_byte_raises(board, bad_id):
    with pytest.raises(ValueError):
        board.bus_servo_set_position(0.5, [[bad_id, 500]])


def test_field_width_helpers():
    assert _u8("x", 0) == 0 and _u8("x", 255) == 255
    assert _u16("x", 65535) == 65535
    assert _i8("x", -128) == -128 and _i8("x", 127) == 127
    for fn, value in ((_u8, 256), (_u8, -1), (_u16, 65536), (_i8, 128), (_i8, -129)):
        with pytest.raises(ValueError):
            fn("x", value)


# ------------------------------------------------------------- concurrency

def test_concurrent_writers_never_interleave_one_packet(fake_serial):
    """Regression for the unlocked _write shared by the motion and main threads."""
    class OverlapDetector(FakeSerial):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.active = 0
            self.overlapped = False
            self.guard = threading.Lock()

        def write(self, buf):
            with self.guard:
                self.active += 1
                if self.active > 1:
                    self.overlapped = True
            time.sleep(0.002)
            with self.guard:
                self.active -= 1
            return super().write(buf)

    port = OverlapDetector()
    import ainex_api.board as board_module
    original = board_module.serial.Serial
    board_module.serial.Serial = lambda *a, **kw: port
    try:
        b = Board(device="/dev/fake")
    finally:
        board_module.serial.Serial = original

    def spam(servo_id):
        for _ in range(15):
            b.bus_servo_set_position(0.1, [[servo_id, 500]])

    threads = [threading.Thread(target=spam, args=(i,)) for i in (1, 13, 23)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    b.close()

    assert not port.overlapped
    assert len(port.writes) == 45
    for packet in port.writes:
        assert packet[:2] == b"\xAA\x55"
        assert packet[-1] == crc8(packet[2:-1])
        assert len(packet) == 4 + packet[3] + 1


# --------------------------------------------------------- sensor decoders

def test_imu_nine_axis_decodes(board):
    values = tuple(float(i) for i in range(9))
    feed_and_settle(board, frame(int(PacketFunction.IMU), struct.pack("<9f", *values)))
    assert board.get_imu() == pytest.approx(values)


def test_imu_six_axis_decodes(board):
    values = tuple(float(i) for i in range(6))
    feed_and_settle(board, frame(int(PacketFunction.IMU), struct.pack("<6f", *values)))
    assert board.get_imu() == pytest.approx(values)


@pytest.mark.parametrize("payload", [b"", b"\x00" * 12, b"\x00" * 40])
def test_imu_with_an_unexpected_length_returns_none(board, payload):
    feed_and_settle(board, frame(int(PacketFunction.IMU), payload))
    assert board.get_imu() is None


def test_battery_decodes_millivolts(board):
    feed_and_settle(board, frame(int(PacketFunction.SYS),
                                 bytes([SYS_BATTERY]) + struct.pack("<H", 11888)))
    assert board.get_battery() == 11888


def test_battery_ignores_other_sys_subcommands(board):
    feed_and_settle(board, frame(int(PacketFunction.SYS), b"\x01\x00\x00"))
    assert board.get_battery() is None


def test_battery_short_payload_returns_none(board):
    feed_and_settle(board, frame(int(PacketFunction.SYS), bytes([SYS_BATTERY])))
    assert board.get_battery() is None


@pytest.mark.parametrize("event,expected", [
    (KeyEvent.CLICK, 0),
    (KeyEvent.PRESSED, 1),
])
def test_button_click_and_press_are_surfaced(board, event, expected):
    feed_and_settle(board, frame(int(PacketFunction.KEY), bytes([2, int(event)])))
    assert board.get_button() == (2, expected)


@pytest.mark.parametrize("event", [KeyEvent.LONGPRESS, KeyEvent.DOUBLE_CLICK,
                                   KeyEvent.TRIPLE_CLICK, KeyEvent.RELEASE_FROM_SP])
def test_other_button_events_are_dropped(board, event):
    feed_and_settle(board, frame(int(PacketFunction.KEY), bytes([1, int(event)])))
    assert board.get_button() is None


def test_unknown_button_event_does_not_kill_the_receiver(board):
    feed_and_settle(board, frame(int(PacketFunction.KEY), b"\x01\x99"))
    assert board.get_button() is None
    assert board._recv_thread.is_alive()


def test_gamepad_axes_and_buttons_decode(board):
    # lx, ly, rx, ry at full scale; the x axes are inverted by the decoder.
    payload = struct.pack("<HB4b", 0x0100, 9, 127, 127, -128, -128)
    feed_and_settle(board, frame(int(PacketFunction.GAMEPAD), payload))
    axes, buttons = board.get_gamepad()
    assert buttons[0] == 1
    assert axes[0] == pytest.approx(-1.0)
    assert axes[1] == pytest.approx(1.0)
    assert axes[2] == pytest.approx(1.0)
    assert axes[3] == pytest.approx(-1.0)
    assert axes[6] == 1


@pytest.mark.parametrize("mask,index", [
    (0x0100, 0), (0x0200, 1), (0x0800, 3), (0x1000, 4), (0x4000, 6),
    (0x8000, 7), (0x0001, 8), (0x0002, 9), (0x0004, 10), (0x0008, 11),
    (0x0020, 13), (0x0040, 14),
])
def test_every_gamepad_button_maps_to_its_documented_slot(board, mask, index):
    payload = struct.pack("<HB4b", mask, 0, 0, 0, 0, 0)
    feed_and_settle(board, frame(int(PacketFunction.GAMEPAD), payload))
    _axes, buttons = board.get_gamepad()
    assert buttons[index] == 1
    assert sum(buttons) == 1


@pytest.mark.parametrize("hat,axis,value", [(9, 6, 1), (13, 6, -1), (11, 7, -1), (15, 7, 1)])
def test_gamepad_hat_directions(board, hat, axis, value):
    payload = struct.pack("<HB4b", 0, hat, 0, 0, 0, 0)
    feed_and_settle(board, frame(int(PacketFunction.GAMEPAD), payload))
    axes, _buttons = board.get_gamepad()
    assert axes[axis] == value


def test_gamepad_wrong_length_returns_none(board):
    feed_and_settle(board, frame(int(PacketFunction.GAMEPAD), b"\x00" * 4))
    assert board.get_gamepad() is None


def test_reads_return_none_while_reception_is_disabled(board):
    feed_and_settle(board, frame(int(PacketFunction.SYS),
                                 bytes([SYS_BATTERY]) + struct.pack("<H", 11888)))
    board.enable_reception(False)
    assert board.get_battery() is None
    assert board.get_imu() is None
    assert board.get_button() is None
    assert board.get_gamepad() is None
