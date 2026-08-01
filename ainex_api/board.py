"""
Board Communication Layer
=========================

Low-level serial communication with the STM32 robot controller.
Protocol: 0xAA 0x55 Function Length Data CRC8
"""

import enum
import time
import queue
import struct
import serial
import threading
from typing import Optional, List, Tuple

# CRC8 lookup table
CRC8_TABLE = [
    0, 94, 188, 226, 97, 63, 221, 131, 194, 156, 126, 32, 163, 253, 31, 65,
    157, 195, 33, 127, 252, 162, 64, 30, 95, 1, 227, 189, 62, 96, 130, 220,
    35, 125, 159, 193, 66, 28, 254, 160, 225, 191, 93, 3, 128, 222, 60, 98,
    190, 224, 2, 92, 223, 129, 99, 61, 124, 34, 192, 158, 29, 67, 161, 255,
    70, 24, 250, 164, 39, 121, 155, 197, 132, 218, 56, 102, 229, 187, 89, 7,
    219, 133, 103, 57, 186, 228, 6, 88, 25, 71, 165, 251, 120, 38, 196, 154,
    101, 59, 217, 135, 4, 90, 184, 230, 167, 249, 27, 69, 198, 152, 122, 36,
    248, 166, 68, 26, 153, 199, 37, 123, 58, 100, 134, 216, 91, 5, 231, 185,
    140, 210, 48, 110, 237, 179, 81, 15, 78, 16, 242, 172, 47, 113, 147, 205,
    17, 79, 173, 243, 112, 46, 204, 146, 211, 141, 111, 49, 178, 236, 14, 80,
    175, 241, 19, 77, 206, 144, 114, 44, 109, 51, 209, 143, 12, 82, 176, 238,
    50, 108, 142, 208, 83, 13, 239, 177, 240, 174, 76, 18, 145, 207, 45, 115,
    202, 148, 118, 40, 171, 245, 23, 73, 8, 86, 180, 234, 105, 55, 213, 139,
    87, 9, 235, 181, 54, 104, 138, 212, 149, 203, 41, 119, 244, 170, 72, 22,
    233, 183, 85, 11, 136, 214, 52, 106, 43, 117, 151, 201, 74, 20, 246, 168,
    116, 42, 200, 150, 21, 75, 169, 247, 182, 232, 10, 84, 215, 137, 107, 53
]


class PacketFunction(enum.IntEnum):
    """Packet function codes for serial protocol"""
    SYS = 0
    LED = 1
    BUZZER = 2
    MOTOR = 3
    PWM_SERVO = 4
    BUS_SERVO = 5
    KEY = 6
    IMU = 7
    GAMEPAD = 8
    SBUS = 9


class KeyEvent(enum.IntEnum):
    """Button event types"""
    PRESSED = 0x01
    LONGPRESS = 0x02
    LONGPRESS_REPEAT = 0x04
    RELEASE_FROM_LP = 0x08
    RELEASE_FROM_SP = 0x10
    CLICK = 0x20
    DOUBLE_CLICK = 0x40
    TRIPLE_CLICK = 0x80


# Function codes the parser will accept; anything else is line noise.
FUNCTION_CODES = frozenset(int(f) for f in PacketFunction)

BROADCAST_SERVO_ID = 254

# Command range of the bus servos. Per-servo mechanical travel is narrower and
# is enforced by servos.check_position.
BUS_SERVO_POSITION_MIN = 0
BUS_SERVO_POSITION_MAX = 1000

# PWM servos take a pulse width in microseconds, not the bus servo scale.
PWM_SERVO_POSITION_MIN = 500
PWM_SERVO_POSITION_MAX = 2500

_GAMEPAD_STRUCT = struct.Struct("<HB4b")
_IMU_9AXIS = struct.Struct("<9f")
_IMU_6AXIS = struct.Struct("<6f")

SYS_BATTERY = 0x04


def crc8(data: bytes) -> int:
    """Calculate CRC8 checksum"""
    check = 0
    for b in data:
        check = CRC8_TABLE[check ^ b]
    return check & 0xFF


def _u8(name: str, value: int) -> int:
    """Reject values that do not fit an unsigned byte field."""
    if not 0 <= value <= 0xFF:
        raise ValueError(f"{name}={value} does not fit an 8-bit field")
    return value


def _u16(name: str, value: int) -> int:
    """Reject values that do not fit an unsigned 16-bit field."""
    if not 0 <= value <= 0xFFFF:
        raise ValueError(f"{name}={value} does not fit a 16-bit field")
    return value


def _i8(name: str, value: int) -> int:
    """Reject values that do not fit a signed byte field."""
    if not -128 <= value <= 127:
        raise ValueError(f"{name}={value} does not fit a signed 8-bit field")
    return value


class Board:
    """
    Low-level board communication.

    Handles serial protocol with STM32 controller.
    Thread-safe with background receiver thread.
    """

    def __init__(self, device: str = "/dev/ttyAMA0", baudrate: int = 1000000, timeout: float = 5.0,
                 recv_enabled: bool = True):
        self.device = device
        self.baudrate = baudrate
        self._recv_enabled = recv_enabled
        self._running = True
        self._retry_count = 10

        # Servo position cache
        self._servo_positions = {i: 500 for i in range(1, 25)}

        # Serial port
        self.port = serial.Serial(None, baudrate, timeout=timeout)
        self.port.rts = False
        self.port.dtr = False
        self.port.setPort(device)
        self.port.open()

        # Response queues
        self._queues = {
            PacketFunction.SYS: queue.Queue(maxsize=2),
            PacketFunction.KEY: queue.Queue(maxsize=2),
            PacketFunction.IMU: queue.Queue(maxsize=2),
            PacketFunction.GAMEPAD: queue.Queue(maxsize=2),
            PacketFunction.BUS_SERVO: queue.Queue(maxsize=2),
            PacketFunction.PWM_SERVO: queue.Queue(maxsize=2),
        }

        # Read lock
        self._servo_lock = threading.Lock()

        # Parser state
        self._state = 0
        self._frame = []
        self._recv_count = 0

        # Start receiver thread
        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._recv_thread.start()
        time.sleep(0.1)

    def close(self):
        """Close the serial port"""
        self._running = False
        if self.port.is_open:
            self.port.close()

    def enable_reception(self, enable: bool = True):
        """Enable/disable data reception from board"""
        self._recv_enabled = enable

    def _write(self, func: PacketFunction, data: bytes):
        """Write packet to serial port"""
        buf = bytes([0xAA, 0x55, int(func), len(data)]) + data
        buf += bytes([crc8(buf[2:])])
        self.port.write(buf)

    def _recv_loop(self):
        """Background thread for receiving packets"""
        while self._running:
            if not self._recv_enabled:
                time.sleep(0.01)
                continue

            try:
                data = self.port.read(self.port.in_waiting or 1)
                if not data:
                    continue

                for byte in data:
                    self._parse_byte(byte)
            except Exception:
                self._state = 0   # never leave the parser mid-frame

    def _parse_byte(self, byte: int):
        """Parse incoming byte through state machine"""
        if self._state == 0:  # Start byte 1
            if byte == 0xAA:
                self._state = 1
        elif self._state == 1:  # Start byte 2
            self._state = 2 if byte == 0x55 else 0
        elif self._state == 2:  # Function
            if byte in FUNCTION_CODES:
                self._frame = [byte, 0]
                self._state = 3
            else:
                self._state = 0
        elif self._state == 3:  # Length
            self._frame[1] = byte
            self._recv_count = 0
            self._state = 5 if byte == 0 else 4
        elif self._state == 4:  # Data
            self._frame.append(byte)
            self._recv_count += 1
            if self._recv_count >= self._frame[1]:
                self._state = 5
        elif self._state == 5:  # Checksum
            if crc8(bytes(self._frame)) == byte:
                func = PacketFunction(self._frame[0])
                payload = bytes(self._frame[2:])
                if func in self._queues:
                    try:
                        self._queues[func].put_nowait(payload)
                    except queue.Full:
                        pass
            self._state = 0

    # ========== Bus Servo Commands ==========

    def bus_servo_set_position(self, duration: float, positions: List[List[int]]):
        """
        Set bus servo positions.

        Args:
            duration: Movement duration in seconds
            positions: List of [servo_id, position] pairs (position 0-1000)
        """
        duration_ms = _u16("duration_ms", int(duration * 1000))
        _u8("servo count", len(positions))
        data = bytes([0x01, duration_ms & 0xFF, (duration_ms >> 8) & 0xFF, len(positions)])
        for servo_id, pos in positions:
            _u8("servo_id", servo_id)
            if not BUS_SERVO_POSITION_MIN <= pos <= BUS_SERVO_POSITION_MAX:
                raise ValueError(f"Position {pos} outside "
                                 f"{BUS_SERVO_POSITION_MIN}-{BUS_SERVO_POSITION_MAX} "
                                 f"for servo {servo_id}")
            self._servo_positions[servo_id] = pos
            data += struct.pack("<BH", servo_id, pos)
        self._write(PacketFunction.BUS_SERVO, data)

    def bus_servo_stop(self, servo_ids: List[int]):
        """Stop specified servos"""
        _u8("servo count", len(servo_ids))
        data = bytes([0x03, len(servo_ids)]) + bytes(_u8("servo_id", s) for s in servo_ids)
        self._write(PacketFunction.BUS_SERVO, data)

    def bus_servo_enable_torque(self, servo_id: int, enable: bool):
        """Enable/disable servo torque"""
        cmd = 0x0B if enable else 0x0C
        _u8("servo_id", servo_id)
        self._write(PacketFunction.BUS_SERVO, struct.pack("<BB", cmd, servo_id))
        time.sleep(0.02)

    def bus_servo_set_offset(self, servo_id: int, offset: int):
        """Set servo offset (-128 to 127)"""
        _u8("servo_id", servo_id)
        _i8("offset", offset)
        self._write(PacketFunction.BUS_SERVO, struct.pack("<BBb", 0x20, servo_id, offset))
        time.sleep(0.02)

    def bus_servo_save_offset(self, servo_id: int):
        """Save servo offset to EEPROM"""
        _u8("servo_id", servo_id)
        self._write(PacketFunction.BUS_SERVO, struct.pack("<BB", 0x24, servo_id))
        time.sleep(0.02)

    def bus_servo_set_angle_limit(self, servo_id: int, min_angle: int, max_angle: int):
        """Set servo angle limits (0-1000)"""
        _u8("servo_id", servo_id)
        _u16("min_angle", min_angle)
        _u16("max_angle", max_angle)
        self._write(PacketFunction.BUS_SERVO, struct.pack("<BBHH", 0x30, servo_id, min_angle, max_angle))
        time.sleep(0.02)

    def _bus_servo_read(self, servo_id: int, cmd: int, fmt: str) -> Optional[Tuple]:
        """
        Read bus servo data with retry.

        Replies are [servo id, command, status, value]; anything that does not
        match this request is a straggler from an earlier one and is discarded.
        """
        _u8("servo_id", servo_id)
        replies = self._queues[PacketFunction.BUS_SERVO]
        with self._servo_lock:
            for _ in range(self._retry_count):
                while not replies.empty():
                    try:
                        replies.get_nowait()
                    except queue.Empty:
                        break
                self._write(PacketFunction.BUS_SERVO, bytes([cmd, servo_id]))
                try:
                    data = replies.get(timeout=0.1)
                    result = struct.unpack(fmt, data)
                except (queue.Empty, struct.error):
                    continue
                if result[1] != cmd:
                    continue
                if servo_id != BROADCAST_SERVO_ID and result[0] != servo_id:
                    continue
                if result[2] == 0:  # Success
                    return result[3:]
            return None

    def bus_servo_read_position(self, servo_id: int, use_cache: bool = False) -> Optional[int]:
        """Read servo position (returns position 0-1000)"""
        if use_cache:
            return self._servo_positions.get(servo_id)
        result = self._bus_servo_read(servo_id, 0x05, "<BBbh")
        return result[0] if result else None

    def bus_servo_read_id(self, servo_id: int = BROADCAST_SERVO_ID) -> Optional[int]:
        """Read servo ID (254 = broadcast)"""
        result = self._bus_servo_read(servo_id, 0x12, "<BBbB")
        return result[0] if result else None

    def bus_servo_read_offset(self, servo_id: int) -> Optional[int]:
        """Read servo offset"""
        result = self._bus_servo_read(servo_id, 0x22, "<BBbb")
        return result[0] if result else None

    def bus_servo_read_voltage(self, servo_id: int) -> Optional[int]:
        """Read servo voltage in mV"""
        result = self._bus_servo_read(servo_id, 0x07, "<BBbH")
        return result[0] if result else None

    def bus_servo_read_temperature(self, servo_id: int) -> Optional[int]:
        """Read servo temperature in Celsius"""
        result = self._bus_servo_read(servo_id, 0x09, "<BBbB")
        return result[0] if result else None

    # ========== PWM Servo Commands ==========

    def pwm_servo_set_position(self, duration: float, positions: List[List[int]]):
        """Set PWM servo positions (position 500-2500)"""
        duration_ms = _u16("duration_ms", int(duration * 1000))
        _u8("servo count", len(positions))
        data = bytes([0x01, duration_ms & 0xFF, (duration_ms >> 8) & 0xFF, len(positions)])
        for servo_id, pos in positions:
            _u8("servo_id", servo_id)
            if not PWM_SERVO_POSITION_MIN <= pos <= PWM_SERVO_POSITION_MAX:
                raise ValueError(f"Position {pos} outside "
                                 f"{PWM_SERVO_POSITION_MIN}-{PWM_SERVO_POSITION_MAX} "
                                 f"for PWM servo {servo_id}")
            data += struct.pack("<BH", servo_id, pos)
        self._write(PacketFunction.PWM_SERVO, data)

    # ========== Peripherals ==========

    def set_buzzer(self, freq: int, on_time: float, off_time: float = 0, repeat: int = 1):
        """
        Play buzzer tone.

        Note: Buzzer may not be present on all AINEX models.

        Args:
            freq: Frequency in Hz
            on_time: On duration in seconds
            off_time: Off duration in seconds
            repeat: Number of repetitions
        """
        data = struct.pack("<HHHH",
                           _u16("freq", freq),
                           _u16("on_ms", int(on_time * 1000)),
                           _u16("off_ms", int(off_time * 1000)),
                           _u16("repeat", repeat))
        self._write(PacketFunction.BUZZER, data)

    def set_led(self, on_time: float, off_time: float, repeat: int = 1, led_id: int = 1):
        """Blink LED"""
        on_ms = _u16("on_ms", int(on_time * 1000))
        off_ms = _u16("off_ms", int(off_time * 1000))
        _u8("led_id", led_id)
        _u16("repeat", repeat)
        self._write(PacketFunction.LED, struct.pack("<BHHH", led_id, on_ms, off_ms, repeat))

    def set_motor_speed(self, speeds: List[Tuple[int, float]]):
        """Set motor speeds [(motor_id, speed), ...] speed -1.0 to 1.0"""
        _u8("motor count", len(speeds))
        data = bytes([0x01, len(speeds)])
        for motor_id, speed in speeds:
            data += struct.pack("<Bf", _u8("motor_id", motor_id - 1), speed)
        self._write(PacketFunction.MOTOR, data)

    # ========== Sensors ==========

    def get_imu(self) -> Optional[Tuple[float, ...]]:
        """Get IMU data (ax, ay, az, gx, gy, gz[, mx, my, mz])"""
        if not self._recv_enabled:
            return None
        try:
            data = self._queues[PacketFunction.IMU].get_nowait()
        except queue.Empty:
            return None
        if len(data) == _IMU_9AXIS.size:
            return _IMU_9AXIS.unpack(data)
        if len(data) == _IMU_6AXIS.size:
            return _IMU_6AXIS.unpack(data)
        return None

    def get_button(self) -> Optional[Tuple[int, int]]:
        """Get button event (button_id, event_type) 0=click, 1=pressed"""
        if not self._recv_enabled:
            return None
        try:
            data = self._queues[PacketFunction.KEY].get_nowait()
        except queue.Empty:
            return None
        if len(data) < 2:
            return None
        try:
            event = KeyEvent(data[1])
        except ValueError:
            return None   # event types this API does not surface
        if event == KeyEvent.CLICK:
            return data[0], 0
        if event == KeyEvent.PRESSED:
            return data[0], 1
        return None

    def get_battery(self) -> Optional[int]:
        """Get battery voltage in mV"""
        if not self._recv_enabled:
            return None
        try:
            data = self._queues[PacketFunction.SYS].get_nowait()
        except queue.Empty:
            return None
        if len(data) >= 3 and data[0] == SYS_BATTERY:
            return struct.unpack('<H', data[1:3])[0]
        return None

    def get_gamepad(self) -> Optional[Tuple[List[float], List[int]]]:
        """Get gamepad data (axes, buttons)"""
        if not self._recv_enabled:
            return None
        try:
            data = self._queues[PacketFunction.GAMEPAD].get_nowait()
        except queue.Empty:
            return None
        if len(data) != _GAMEPAD_STRUCT.size:
            return None

        buttons_raw, hat, lx, ly, rx, ry = _GAMEPAD_STRUCT.unpack(data)

        axes = [0.0] * 8
        buttons = [0] * 16

        # Parse buttons
        button_masks = [
            (0x0100, 0), (0x0200, 1), (0x0800, 3), (0x1000, 4),
            (0x4000, 6), (0x8000, 7), (0x0004, 10), (0x0008, 11),
            (0x0001, 8), (0x0002, 9), (0x0020, 13), (0x0040, 14)
        ]
        for mask, idx in button_masks:
            if buttons_raw & mask:
                buttons[idx] = 1

        # Parse axes
        axes[0] = -lx / 127 if lx > 0 else -lx / 128
        axes[1] = ly / 127 if ly > 0 else ly / 128
        axes[2] = -rx / 127 if rx > 0 else -rx / 128
        axes[3] = ry / 127 if ry > 0 else ry / 128
        axes[4] = 1 if buttons_raw & 0x0002 else 0
        axes[5] = 1 if buttons_raw & 0x0001 else 0

        if hat == 9: axes[6] = 1
        elif hat == 13: axes[6] = -1
        if hat == 11: axes[7] = -1
        elif hat == 15: axes[7] = 1

        return axes, buttons
