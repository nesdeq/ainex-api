"""
Sensor Reader
=============

High-level interface for robot sensors.
"""

import time
import threading
from typing import Optional, Tuple, List, Callable
from dataclasses import dataclass
from .board import Board


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
        self._running = False

        # Last values
        self._last_imu: Optional[IMUData] = None
        self._last_battery: Optional[int] = None

    def start(self):
        """Start sensor polling"""
        self.board.enable_reception(True)
        self._running = True

        if self._button_callback:
            self._button_thread = threading.Thread(target=self._poll_buttons, daemon=True)
            self._button_thread.start()

    def stop(self):
        """Stop sensor polling"""
        self._running = False
        self.board.enable_reception(False)

    def get_imu(self) -> Optional[IMUData]:
        """
        Get latest IMU data.

        Returns:
            IMUData or None if not available
        """
        data = self.board.get_imu()
        if data:
            self._last_imu = IMUData.from_tuple(data)
        return self._last_imu

    def get_battery_voltage(self) -> Optional[int]:
        """
        Get battery voltage in millivolts.

        Returns:
            Voltage in mV or None
        """
        voltage = self.board.get_battery()
        if voltage is not None:
            self._last_battery = voltage
        return self._last_battery

    def get_battery_percent(self) -> Optional[float]:
        """
        Get estimated battery percentage.

        Based on typical 2S LiPo: 6.4V (0%) to 8.4V (100%)
        """
        voltage = self.get_battery_voltage()
        if voltage is None:
            return None

        # Convert to percentage (2S LiPo range)
        min_v = 6400  # 6.4V empty
        max_v = 8400  # 8.4V full
        percent = (voltage - min_v) / (max_v - min_v) * 100
        return max(0, min(100, percent))

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

    def _poll_buttons(self):
        """Background thread for button polling"""
        while self._running:
            event = self.get_button()
            if event and self._button_callback:
                try:
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
