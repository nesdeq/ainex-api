"""Shared fixtures.

Most tests drive the real Board class through a fake serial port, so the framing,
CRC and reply-correlation code under test is the same code that ships. Tests
marked `hardware` or `camera` talk to the actual robot and skip when it is absent.
"""

import os
import sys
import queue
import struct
import threading

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ainex_api.board import Board, PacketFunction, crc8  # noqa: E402

SERIAL_DEVICE = "/dev/ttyAMA0"
CAMERA_DEVICE = "/dev/usb_cam"

HAS_SERIAL = os.path.exists(SERIAL_DEVICE) and os.access(SERIAL_DEVICE, os.R_OK | os.W_OK)
HAS_CAMERA = os.path.exists(CAMERA_DEVICE)

requires_hardware = pytest.mark.skipif(not HAS_SERIAL,
                                       reason=f"no writable {SERIAL_DEVICE}")
requires_camera = pytest.mark.skipif(not HAS_CAMERA,
                                     reason=f"no {CAMERA_DEVICE}")


def frame(func: int, payload: bytes) -> bytes:
    """Build a wire frame the way the board would send it."""
    body = bytes([func, len(payload)]) + payload
    return b"\xAA\x55" + body + bytes([crc8(body)])


class FakeSerial:
    """Serial stand-in that records writes and replays programmed replies."""

    def __init__(self, *args, **kwargs):
        self.is_open = False
        self.rts = None
        self.dtr = None
        self.port = None
        self.writes = []
        self._inbox = bytearray()
        self._lock = threading.Lock()
        # Called with each written packet; returns bytes for the robot to read.
        self.responder = None

    def setPort(self, device):
        self.port = device

    def open(self):
        self.is_open = True

    def close(self):
        self.is_open = False

    def write(self, buf):
        with self._lock:
            self.writes.append(bytes(buf))
            reply = self.responder(bytes(buf)) if self.responder else None
            if reply:
                self._inbox.extend(reply)
        return len(buf)

    def feed(self, data: bytes):
        """Queue bytes as if the board had sent them."""
        with self._lock:
            self._inbox.extend(data)

    @property
    def in_waiting(self):
        with self._lock:
            return len(self._inbox)

    def read(self, size=1):
        with self._lock:
            if not self._inbox:
                return b""
            take = min(size, len(self._inbox))
            out = bytes(self._inbox[:take])
            del self._inbox[:take]
            return out


@pytest.fixture
def fake_serial(monkeypatch):
    """Patch pyserial so Board opens a FakeSerial, and hand it back."""
    holder = {}

    def factory(*args, **kwargs):
        holder["port"] = FakeSerial(*args, **kwargs)
        return holder["port"]

    monkeypatch.setattr("ainex_api.board.serial.Serial", factory)
    return holder


@pytest.fixture
def board(fake_serial):
    """A real Board driving a fake port, so the shipping code is what runs."""
    b = Board(device="/dev/fake")
    b.port_fake = fake_serial["port"]
    yield b
    b.close()


class ServoModel:
    """Minimal servo bus that answers reads the way the real firmware does."""

    REPLY_FORMATS = {
        0x05: "<h",   # position
        0x07: "<H",   # voltage mV
        0x09: "<B",   # temperature C
        0x12: "<B",   # id
        0x22: "<b",   # offset
    }

    def __init__(self, values=None, present=None):
        self.values = values or {}
        self.present = present
        self.requests = []

    def __call__(self, packet: bytes):
        # packet is AA 55 func len data... crc
        if len(packet) < 6 or packet[2] != int(PacketFunction.BUS_SERVO):
            return None
        payload = packet[4:-1]
        if len(payload) != 2:
            return None
        cmd, servo_id = payload[0], payload[1]
        if cmd not in self.REPLY_FORMATS:
            return None
        self.requests.append((cmd, servo_id))
        if self.present is not None and servo_id not in self.present:
            return None
        value = self.values.get((servo_id, cmd), 0)
        body = struct.pack("<BBb", servo_id, cmd, 0)
        body += struct.pack(self.REPLY_FORMATS[cmd], value)
        return frame(int(PacketFunction.BUS_SERVO), body)


def drain(q: queue.Queue):
    """Empty a queue without blocking."""
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            return
