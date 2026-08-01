"""
Peripherals Controller
======================

Controls buzzer and LED.
"""

import time
from .board import Board


class Peripherals:
    """
    Control robot peripherals.

    - Buzzer: Sound feedback
    - LED: Status indicator
    """

    def __init__(self, board: Board):
        self.board = board

    # ========== Buzzer ==========

    def beep(self, freq: int = 2000, duration: float = 0.1):
        """
        Play a single beep.

        Args:
            freq: Frequency in Hz; only the field width is checked, so values
                  outside the buzzer's usable band simply produce nothing
            duration: Duration in seconds
        """
        self.board.set_buzzer(freq, duration, 0, 1)

    def beep_pattern(self, freq: int, on_time: float, off_time: float, repeat: int = 1):
        """
        Play a beep pattern.

        Args:
            freq: Frequency in Hz
            on_time: On duration in seconds
            off_time: Off duration in seconds
            repeat: Number of repetitions
        """
        self.board.set_buzzer(freq, on_time, off_time, repeat)

    def chirp(self):
        """Quick chirp sound"""
        self.beep(2400, 0.05)

    def confirm(self):
        """Confirmation sound (two beeps)"""
        self.beep_pattern(2000, 0.1, 0.1, 2)

    def error_sound(self):
        """Error sound (low tone)"""
        self.beep(500, 0.3)

    def startup_sound(self):
        """Startup melody"""
        for freq in [1000, 1500, 2000]:
            self.beep(freq, 0.1)
            time.sleep(0.12)

    # ========== LED ==========

    def led_on(self, led_id: int = 1):
        """Turn LED on"""
        self.board.set_led(1.0, 0, 1, led_id)

    def led_off(self, led_id: int = 1):
        """Turn LED off"""
        self.board.set_led(0, 1.0, 1, led_id)

    def led_blink(self, on_time: float = 0.5, off_time: float = 0.5,
                  repeat: int = 3, led_id: int = 1):
        """
        Blink LED.

        Args:
            on_time: On duration in seconds
            off_time: Off duration in seconds
            repeat: Number of blinks
            led_id: LED ID (default 1)
        """
        self.board.set_led(on_time, off_time, repeat, led_id)
