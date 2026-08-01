"""Battery maths, staleness, smoothing and the button callback lifecycle."""

import threading
import time

import pytest

from ainex_api.sensors import (
    SensorReader, IMUData, BatteryState, DISCHARGE_CURVE, CELL_COUNT,
    LOW_CELL_MV, CRITICAL_CELL_MV, SMOOTHING_SAMPLES, SENSOR_TTL_S,
    _interpolate_percent, _status_for,
)


class StubBoard:
    """Board stand-in that hands out queued sensor readings."""

    def __init__(self):
        self.imu = None
        self.battery = None
        self.button = None
        self.gamepad = None

    def get_imu(self):
        return self.imu

    def get_battery(self):
        return self.battery

    def get_button(self):
        return self.button

    def get_gamepad(self):
        return self.gamepad


@pytest.fixture
def reader():
    return SensorReader(StubBoard())


# ------------------------------------------------------------ battery maths

def test_discharge_curve_descends_in_both_columns():
    voltages = [mv for mv, _ in DISCHARGE_CURVE]
    percents = [pct for _, pct in DISCHARGE_CURVE]
    assert voltages == sorted(voltages, reverse=True)
    assert percents == sorted(percents, reverse=True)
    assert len(set(voltages)) == len(voltages)


def test_curve_spans_a_full_charge_range():
    assert DISCHARGE_CURVE[0][1] == 100
    assert DISCHARGE_CURVE[-1][1] == 0


def test_critical_is_below_low():
    assert CRITICAL_CELL_MV < LOW_CELL_MV


def test_thresholds_sit_inside_the_curve():
    assert DISCHARGE_CURVE[-1][0] < CRITICAL_CELL_MV < DISCHARGE_CURVE[0][0]
    assert DISCHARGE_CURVE[-1][0] < LOW_CELL_MV < DISCHARGE_CURVE[0][0]


@pytest.mark.parametrize("cell_mv,expected", [(5000, 100.0), (4200, 100.0),
                                              (3270, 0.0), (2000, 0.0)])
def test_interpolation_saturates_outside_the_curve(cell_mv, expected):
    assert _interpolate_percent(cell_mv) == expected


@pytest.mark.parametrize("mv,pct", DISCHARGE_CURVE)
def test_interpolation_reproduces_every_curve_point(mv, pct):
    assert _interpolate_percent(mv) == pytest.approx(pct, abs=1e-6)


def test_interpolation_is_monotonic_across_the_range():
    values = [_interpolate_percent(mv) for mv in range(3200, 4300, 5)]
    assert values == sorted(values)


def test_interpolation_lands_between_its_neighbours():
    assert 45 < _interpolate_percent(3830) < 50


@pytest.mark.parametrize("cell_mv,status", [
    (4200, 'ok'), (3600, 'ok'), (LOW_CELL_MV + 1, 'ok'),
    (LOW_CELL_MV, 'low'), (3400, 'low'), (CRITICAL_CELL_MV + 1, 'low'),
    (CRITICAL_CELL_MV, 'critical'), (3000, 'critical'),
])
def test_status_thresholds(cell_mv, status):
    assert _status_for(cell_mv) == status


# --------------------------------------------------------------- staleness

def test_battery_reading_is_reported_once_it_arrives(reader):
    reader.board.battery = 11888
    assert reader.get_battery_voltage() == 11888


def test_last_battery_reading_is_held_briefly_when_the_link_pauses(reader):
    reader.board.battery = 11888
    reader.get_battery_voltage()
    reader.board.battery = None
    assert reader.get_battery_voltage() == 11888


def test_battery_goes_none_once_the_reading_is_stale(reader, monkeypatch):
    reader.board.battery = 11888
    reader.get_battery_voltage()
    reader.board.battery = None

    base = time.monotonic()
    monkeypatch.setattr("ainex_api.sensors.time.monotonic",
                        lambda: base + SENSOR_TTL_S + 1)
    assert reader.get_battery_voltage() is None
    assert reader.get_battery_state() is None


def test_imu_goes_none_once_the_reading_is_stale(reader, monkeypatch):
    reader.board.imu = tuple(float(i) for i in range(9))
    assert reader.get_imu() is not None
    reader.board.imu = None

    base = time.monotonic()
    monkeypatch.setattr("ainex_api.sensors.time.monotonic",
                        lambda: base + SENSOR_TTL_S + 1)
    assert reader.get_imu() is None


def test_stale_battery_also_empties_the_smoothing_window(reader, monkeypatch):
    reader.board.battery = 11888
    for _ in range(SMOOTHING_SAMPLES):
        reader.get_battery_voltage()
    assert reader.get_battery_voltage_smoothed() is not None

    reader.board.battery = None
    base = time.monotonic()
    monkeypatch.setattr("ainex_api.sensors.time.monotonic",
                        lambda: base + SENSOR_TTL_S + 1)
    assert reader.get_battery_voltage() is None
    assert reader.get_battery_voltage_smoothed() is None


# --------------------------------------------------------------- smoothing

def test_smoothed_voltage_needs_a_full_window(reader):
    reader.board.battery = 11888
    for _ in range(SMOOTHING_SAMPLES - 1):
        assert reader.get_battery_voltage_smoothed() is None
    assert reader.get_battery_voltage_smoothed() == 11888


def test_median_rejects_a_load_sag(reader):
    """A single servo-load dip must not drag the reported voltage down."""
    for mv in (11900, 11880, 9000, 11890, 11870):
        reader.board.battery = mv
        reader.get_battery_voltage()
    # Silence the link so the smoothed getter's own read adds no sixth sample.
    reader.board.battery = None
    assert reader.get_battery_voltage_smoothed() == 11880


def test_the_smoothed_getter_takes_a_sample_of_its_own(reader):
    reader.board.battery = 11888
    for _ in range(SMOOTHING_SAMPLES - 1):
        reader.get_battery_voltage()
    assert len(reader._battery_window) == SMOOTHING_SAMPLES - 1
    assert reader.get_battery_voltage_smoothed() == 11888
    assert len(reader._battery_window) == SMOOTHING_SAMPLES


def test_battery_state_is_internally_consistent(reader):
    reader.board.battery = 11400
    for _ in range(SMOOTHING_SAMPLES):
        reader.get_battery_voltage()
    state = reader.get_battery_state()
    assert isinstance(state, BatteryState)
    assert state.percent == _interpolate_percent(state.voltage_mv / CELL_COUNT)
    assert state.status == _status_for(state.voltage_mv / CELL_COUNT)
    assert reader.get_battery_percent() == state.percent
    assert reader.get_battery_status() == state.status


def test_battery_helpers_return_none_before_any_reading(reader):
    assert reader.get_battery_state() is None
    assert reader.get_battery_percent() is None
    assert reader.get_battery_status() is None


# ------------------------------------------------------------------ IMU

def test_imu_from_nine_values_keeps_the_magnetometer():
    imu = IMUData.from_tuple(tuple(float(i) for i in range(9)))
    assert (imu.accel_x, imu.gyro_x, imu.mag_z) == (0.0, 3.0, 8.0)


def test_imu_from_six_values_zeroes_the_magnetometer():
    imu = IMUData.from_tuple(tuple(float(i) for i in range(6)))
    assert (imu.mag_x, imu.mag_y, imu.mag_z) == (0.0, 0.0, 0.0)


def test_imu_from_too_few_values_raises():
    with pytest.raises(ValueError):
        IMUData.from_tuple((1.0, 2.0))


# ---------------------------------------------------------------- buttons

@pytest.mark.parametrize("code,event_type", [(0, 'click'), (1, 'press')])
def test_button_events_are_typed(reader, code, event_type):
    reader.board.button = (2, code)
    event = reader.get_button()
    assert (event.button_id, event.event_type) == (2, event_type)


def test_no_button_event_returns_none(reader):
    assert reader.get_button() is None


def test_callback_registered_before_start_fires(reader):
    seen = threading.Event()
    reader.on_button(lambda _e: seen.set())
    reader.board.button = (1, 0)
    reader.start()
    try:
        assert seen.wait(timeout=2.0)
    finally:
        reader.stop()


def test_callback_registered_after_start_still_fires(reader):
    """Registering a callback after start() used to silently never poll."""
    reader.start()
    seen = threading.Event()
    reader.board.button = (1, 0)
    reader.on_button(lambda _e: seen.set())
    try:
        assert seen.wait(timeout=2.0)
    finally:
        reader.stop()


def test_registering_twice_does_not_start_a_second_thread(reader):
    reader.start()
    reader.on_button(lambda _e: None)
    first = reader._button_thread
    reader.on_button(lambda _e: None)
    try:
        assert reader._button_thread is first
    finally:
        reader.stop()


def test_a_failing_callback_does_not_kill_the_poll_thread(reader):
    calls = []

    def boom(event):
        calls.append(event)
        raise RuntimeError("callback blew up")

    reader.board.button = (1, 0)
    reader.on_button(boom)
    reader.start()
    try:
        deadline = time.monotonic() + 2.0
        while len(calls) < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(calls) >= 3
        assert reader._button_thread.is_alive()
    finally:
        reader.stop()


def test_stop_ends_the_poll_thread(reader):
    reader.on_button(lambda _e: None)
    reader.start()
    thread = reader._button_thread
    reader.stop()
    thread.join(timeout=2.0)
    assert not thread.is_alive()


def test_gamepad_is_a_passthrough(reader):
    reader.board.gamepad = ([0.0] * 8, [0] * 16)
    assert reader.get_gamepad() == reader.board.gamepad


def test_concurrent_battery_reads_do_not_corrupt_the_window(reader):
    """The cached state is shared, so the getters take a lock."""
    reader.board.battery = 11888
    errors = []

    def hammer():
        try:
            for _ in range(200):
                reader.get_battery_voltage_smoothed()
                reader.get_battery_state()
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=hammer) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert len(reader._battery_window) == SMOOTHING_SAMPLES
