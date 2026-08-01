"""
Sensor Reader
=============

High-level interface for robot sensors.
"""

import time
import statistics
import threading
from collections import deque
from typing import Optional, Tuple, List, Callable
from dataclasses import dataclass
from .board import Board


# AINEX ships an 11.1V 3500mAh 5C LiPo; its HX-35H bus servos run 9.0-12.6V
CELL_COUNT = 3

# Resting per-cell mV -> state of charge, descending. LiPo discharge is flat
# between 3950 and 3770 mV, so a linear voltage map is wrong across most of it.
DISCHARGE_CURVE = [
    (4200, 100), (4150, 95), (4110, 90), (4080, 85), (4020, 80),
    (3980, 75), (3950, 70), (3910, 65), (3870, 60), (3850, 55),
    (3840, 50), (3820, 45), (3800, 40), (3790, 35), (3770, 30),
    (3750, 25), (3730, 20), (3710, 15), (3690, 10), (3610, 5),
    (3270, 0),
]

LOW_CELL_MV = 3500       # recharge now
CRITICAL_CELL_MV = 3300  # stop and power down; 3000 damages cells
SMOOTHING_SAMPLES = 5    # median window, rejects servo load sag

# The board pushes battery and IMU packets unsolicited. Past this age a reading
# is reported as unavailable rather than served stale, so a dead link cannot
# masquerade as a healthy pack.
SENSOR_TTL_S = 15.0


def _status_for(cell_mv: float) -> str:
    """Battery level for a per-cell voltage."""
    if cell_mv <= CRITICAL_CELL_MV:
        return 'critical'
    if cell_mv <= LOW_CELL_MV:
        return 'low'
    return 'ok'


def _interpolate_percent(cell_mv: float) -> float:
    """State of charge for a per-cell voltage, interpolated over DISCHARGE_CURVE."""
    if cell_mv >= DISCHARGE_CURVE[0][0]:
        return 100.0
    if cell_mv <= DISCHARGE_CURVE[-1][0]:
        return 0.0

    for (hi_mv, hi_pct), (lo_mv, lo_pct) in zip(DISCHARGE_CURVE, DISCHARGE_CURVE[1:]):
        if cell_mv >= lo_mv:
            ratio = (cell_mv - lo_mv) / (hi_mv - lo_mv)
            return lo_pct + ratio * (hi_pct - lo_pct)
    return 0.0


@dataclass
class IMUData:
    """IMU sensor data"""
    accel_x: float
    accel_y: float
    accel_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float
    mag_x: float = 0.0
    mag_y: float = 0.0
    mag_z: float = 0.0

    @classmethod
    def from_tuple(cls, data: Tuple[float, ...]) -> 'IMUData':
        if len(data) >= 9:
            return cls(*data[:9])
        elif len(data) >= 6:
            return cls(*data[:6])
        raise ValueError("Invalid IMU data")


@dataclass
class ButtonEvent:
    """Button press event"""
    button_id: int
    event_type: str  # 'click' or 'press'
    timestamp: float


@dataclass
class BatteryState:
    """Battery voltage, charge and level, all derived from one reading."""
    voltage_mv: float
    percent: float
    status: str  # 'ok', 'low' or 'critical'


class SensorReader:
    """
    High-level sensor interface.

    Provides:
    - IMU data (accelerometer, gyroscope, magnetometer)
    - Button events with callbacks
    - Battery voltage monitoring
    - Gamepad input
    """

    def __init__(self, board: Board):
        self.board = board

        # Button callback
        self._button_callback: Optional[Callable[[ButtonEvent], None]] = None
        self._button_thread: Optional[threading.Thread] = None
        self._thread_lock = threading.Lock()
        self._running = False

        # Last values, with the monotonic time they arrived. Reentrant because
        # the smoothed getter reads the window through the plain one.
        self._state_lock = threading.RLock()
        self._last_imu: Optional[IMUData] = None
        self._last_imu_time: Optional[float] = None
        self._last_battery: Optional[int] = None
        self._last_battery_time: Optional[float] = None
        self._battery_window = deque(maxlen=SMOOTHING_SAMPLES)

    def start(self):
        """
        Start button polling.

        Polling only runs once a callback exists, so calling this before
        on_button() is fine: registering the callback starts the thread.
        """
        self._running = True
        if self._button_callback:
            self._ensure_button_thread()

    def stop(self):
        """Stop button polling"""
        self._running = False

    def _ensure_button_thread(self):
        """Start the poll thread unless one is already running"""
        with self._thread_lock:
            if self._button_thread is not None and self._button_thread.is_alive():
                return
            self._button_thread = threading.Thread(target=self._poll_buttons, daemon=True)
            self._button_thread.start()

    def get_imu(self) -> Optional[IMUData]:
        """
        Get latest IMU data.

        Returns:
            IMUData, or None if nothing has arrived within SENSOR_TTL_S
        """
        data = self.board.get_imu()
        now = time.monotonic()
        with self._state_lock:
            if data:
                self._last_imu = IMUData.from_tuple(data)
                self._last_imu_time = now
            elif self._last_imu_time is not None and now - self._last_imu_time > SENSOR_TTL_S:
                self._last_imu = None
                self._last_imu_time = None
            return self._last_imu

    def get_battery_voltage(self) -> Optional[int]:
        """
        Get battery voltage in millivolts.

        Returns:
            Voltage in mV, or None if nothing has arrived within SENSOR_TTL_S
        """
        voltage = self.board.get_battery()
        now = time.monotonic()
        with self._state_lock:
            if voltage is not None:
                self._last_battery = voltage
                self._last_battery_time = now
                self._battery_window.append(voltage)
            elif self._last_battery_time is not None and now - self._last_battery_time > SENSOR_TTL_S:
                self._last_battery = None
                self._last_battery_time = None
                self._battery_window.clear()
            return self._last_battery

    def get_battery_voltage_smoothed(self) -> Optional[float]:
        """
        Median of the last SMOOTHING_SAMPLES readings, in millivolts.

        Servos under load sag the pack by a volt or more; the median rejects
        those transients. It needs a full window to do so, so this returns None
        until SMOOTHING_SAMPLES readings have arrived.
        """
        self.get_battery_voltage()
        with self._state_lock:
            if len(self._battery_window) < SMOOTHING_SAMPLES:
                return None
            return statistics.median(self._battery_window)

    def get_battery_state(self) -> Optional[BatteryState]:
        """
        Voltage, charge and level from a single reading.

        Use this instead of the individual getters when reporting more than one
        of them, so the three cannot describe different instants.
        """
        voltage = self.get_battery_voltage_smoothed()
        if voltage is None:
            return None
        cell_mv = voltage / CELL_COUNT
        return BatteryState(voltage_mv=voltage,
                            percent=_interpolate_percent(cell_mv),
                            status=_status_for(cell_mv))

    def get_battery_percent(self) -> Optional[float]:
        """
        Get estimated state of charge for the 3S LiPo pack.

        Interpolates the resting discharge curve from the smoothed voltage.
        Reads pessimistically while the robot is walking.
        """
        state = self.get_battery_state()
        return state.percent if state else None

    def get_battery_status(self) -> Optional[str]:
        """
        Battery level as 'ok', 'low' (recharge now) or 'critical' (power down).

        Thresholds are 3.50 and 3.30 V per cell, leaving margin above the
        3.00 V cell damage floor and the 9.0 V servo minimum.
        """
        state = self.get_battery_state()
        return state.status if state else None

    def get_button(self) -> Optional[ButtonEvent]:
        """
        Get button event (non-blocking).

        Returns:
            ButtonEvent or None
        """
        data = self.board.get_button()
        if data:
            button_id, event_code = data
            event_type = 'click' if event_code == 0 else 'press'
            return ButtonEvent(
                button_id=button_id,
                event_type=event_type,
                timestamp=time.time()
            )
        return None

    def on_button(self, callback: Callable[[ButtonEvent], None]):
        """
        Register button callback.

        Args:
            callback: Function called with ButtonEvent
        """
        self._button_callback = callback
        if self._running:
            self._ensure_button_thread()

    def _poll_buttons(self):
        """Background thread for button polling; no fault may kill it"""
        while self._running:
            try:
                event = self.get_button()
                if event and self._button_callback:
                    self._button_callback(event)
            except Exception:
                pass
            time.sleep(0.02)

    def get_gamepad(self) -> Optional[Tuple[List[float], List[int]]]:
        """
        Get gamepad state.

        Returns:
            Tuple of (axes, buttons) or None
            axes: [lx, ly, rx, ry, r2, l2, hat_x, hat_y]
            buttons: [cross, circle, _, square, triangle, _, l1, r1, l2, r2, select, start, _, l3, r3, _]
        """
        return self.board.get_gamepad()
