"""Servo map integrity and the travel limits that keep joints inside their range."""

import pytest

from ainex_api.servos import (
    ServoController, SERVO_MAP, NAME_TO_ID, SERVO_LIMITS, SERVO_CENTER,
    BODY_SERVO_IDS, ALL_SERVO_IDS, SERVO_POSITION_MIN, SERVO_POSITION_MAX,
    check_position, servo_limits,
)


def test_servo_map_covers_ids_one_to_twentyfour():
    assert sorted(SERVO_MAP) == list(range(1, 25))


def test_servo_names_are_unique():
    assert len(set(SERVO_MAP.values())) == len(SERVO_MAP)
    assert len(NAME_TO_ID) == len(SERVO_MAP)


def test_odd_ids_are_left_and_even_ids_are_right():
    for servo_id, name in SERVO_MAP.items():
        if servo_id > 22:
            continue
        assert name.startswith("l_" if servo_id % 2 else "r_"), (servo_id, name)


def test_body_and_all_id_sets_match_the_map():
    assert BODY_SERVO_IDS == list(range(1, 23))
    assert ALL_SERVO_IDS == sorted(SERVO_MAP)
    assert set(ALL_SERVO_IDS) - set(BODY_SERVO_IDS) == {23, 24}


def test_only_the_head_servos_narrow_their_travel():
    assert set(SERVO_LIMITS) == {23, 24}
    for servo_id, (lo, hi) in SERVO_LIMITS.items():
        assert SERVO_POSITION_MIN <= lo < hi <= SERVO_POSITION_MAX


def test_centre_is_inside_every_servo_travel():
    for servo_id in ALL_SERVO_IDS:
        lo, hi = servo_limits(servo_id)
        assert lo <= SERVO_CENTER <= hi


def test_unlimited_servos_default_to_the_command_range():
    assert servo_limits(1) == (SERVO_POSITION_MIN, SERVO_POSITION_MAX)


@pytest.mark.parametrize("servo_id,position", [
    (1, 0), (1, 1000), (1, 500), (23, 125), (23, 875), (24, 315), (24, 625),
])
def test_positions_inside_travel_are_accepted(servo_id, position):
    assert check_position(servo_id, position) == position


@pytest.mark.parametrize("servo_id,position", [
    (1, -1), (1, 1001), (23, 124), (23, 876), (24, 314), (24, 626),
])
def test_positions_outside_travel_raise(servo_id, position):
    with pytest.raises(ValueError):
        check_position(servo_id, position)


@pytest.mark.parametrize("servo_id", [0, 25, 254, -1])
def test_unknown_servo_id_raises(servo_id):
    with pytest.raises(ValueError):
        check_position(servo_id, 500)


def test_head_travel_is_enforced_through_the_named_api(board):
    controller = ServoController(board)
    with pytest.raises(ValueError):
        controller.set_by_name("head_pan", 9000)
    assert board.port_fake.writes == []


def test_set_positions_rejects_the_batch_before_writing_anything(board):
    controller = ServoController(board)
    with pytest.raises(ValueError):
        controller.set_positions({13: 500, 23: 9000})
    assert board.port_fake.writes == []


def test_set_body_requires_exactly_twentytwo_positions(board):
    controller = ServoController(board)
    for count in (21, 23, 0):
        with pytest.raises(ValueError):
            controller.set_body([SERVO_CENTER] * count)


def test_set_body_maps_positions_onto_ids_in_order(board):
    controller = ServoController(board)
    controller.set_body([SERVO_CENTER] * len(BODY_SERVO_IDS))
    data = board.port_fake.writes[-1][4:-1]
    assert data[3] == len(BODY_SERVO_IDS)
    ids = [data[4 + i * 3] for i in range(len(BODY_SERVO_IDS))]
    assert ids == BODY_SERVO_IDS


def test_unknown_servo_name_raises(board):
    controller = ServoController(board)
    with pytest.raises(ValueError):
        controller.set_by_name("nose_wiggle", 500)


def test_name_and_id_lookups_round_trip():
    controller = ServoController(None)
    for servo_id, name in SERVO_MAP.items():
        assert controller.get_servo_name(servo_id) == name
        assert controller.get_servo_id(name) == servo_id


def test_unknown_lookups_degrade_predictably():
    controller = ServoController(None)
    assert controller.get_servo_name(99) == "servo_99"
    assert controller.get_servo_id("nope") is None


def test_stop_defaults_to_every_servo(board):
    controller = ServoController(board)
    controller.stop()
    data = board.port_fake.writes[-1][4:-1]
    assert data[0] == 0x03
    assert list(data[2:]) == ALL_SERVO_IDS
