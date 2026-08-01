"""The remote vision wire protocol, over real loopback sockets."""

import socket
import struct
import threading
import time

import pytest

from ainex_api.remote_vision import (
    RemoteVisionClient, RemoteVisionServer, FRAME_HEADER, RESULT_STRUCT,
    MAGIC_FRAME, MAGIC_RESULT, GESTURE_IDS, ID_TO_GESTURE, MAX_JPEG_BYTES,
    SOCKET_TIMEOUT_S, ACCEPT_POLL_S, _I16_MAX, _I16_MIN, _U16_MAX,
    _CONFIDENCE_SCALE,
)
from ainex_api.vision import FaceData, GestureData

import numpy as np
import cv2


def jpeg_of(width=64, height=48):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    ok, buf = cv2.imencode('.jpg', frame)
    assert ok
    return buf.tobytes()


class StubCamera:
    def __init__(self, frame=None):
        self._frame = frame if frame is not None else np.zeros((48, 64, 3), dtype=np.uint8)
        self.started = False

    def start(self):
        self.started = True
        return True

    def stop(self):
        self.started = False

    def read(self):
        return self._frame


# ------------------------------------------------------------ wire format

def test_struct_sizes_match_the_documented_layout():
    assert FRAME_HEADER.size == 14
    assert RESULT_STRUCT.size == 16


def test_magics_are_distinct_and_two_bytes():
    assert MAGIC_FRAME != MAGIC_RESULT
    assert len(MAGIC_FRAME) == len(MAGIC_RESULT) == 2


def test_gesture_ids_round_trip():
    for name, gid in GESTURE_IDS.items():
        assert ID_TO_GESTURE[gid] == name
    assert len(set(GESTURE_IDS.values())) == len(GESTURE_IDS)
    assert GESTURE_IDS['none'] == 0


def test_every_gesture_id_fits_the_byte_field():
    for gid in GESTURE_IDS.values():
        assert 0 <= gid <= 255


def test_frame_header_round_trips():
    packed = FRAME_HEADER.pack(MAGIC_FRAME, 42, 640, 480, 12345)
    assert FRAME_HEADER.unpack(packed) == (MAGIC_FRAME, 42, 640, 480, 12345)


# ------------------------------------------------------- result clamping

def make_server():
    server = RemoteVisionServer(port=0)
    sent = []
    server._client = type("Sink", (), {"sendall": lambda _s, data: sent.append(data)})()
    return server, sent


@pytest.mark.parametrize("x,y,w,h", [
    (0, 0, 0, 0),
    (_I16_MAX + 10, _I16_MIN - 10, _U16_MAX + 10, -5),
    (-40000, 40000, 100000, 100000),
])
def test_out_of_domain_face_boxes_are_clamped_not_raised(x, y, w, h):
    server, sent = make_server()
    face = FaceData(x=x, y=y, width=w, height=h, timestamp=0.0)
    server._send_result(1, face, None)
    _magic, _fid, fx, fy, fw, fh, _g, _c = RESULT_STRUCT.unpack(sent[-1])
    assert _I16_MIN <= fx <= _I16_MAX
    assert _I16_MIN <= fy <= _I16_MAX
    assert 0 <= fw <= _U16_MAX
    assert 0 <= fh <= _U16_MAX


@pytest.mark.parametrize("confidence", [-1.0, 0.0, 0.5, 1.0, 2.5, 99.0])
def test_confidence_stays_inside_the_percentage_domain(confidence):
    server, sent = make_server()
    gesture = GestureData(gesture='waving', confidence=confidence, timestamp=0.0)
    server._send_result(1, None, gesture)
    *_, encoded = RESULT_STRUCT.unpack(sent[-1])
    assert 0 <= encoded <= _CONFIDENCE_SCALE


def test_absent_face_and_gesture_encode_as_empty():
    server, sent = make_server()
    server._send_result(7, None, None)
    magic, fid, fx, fy, fw, fh, gid, conf = RESULT_STRUCT.unpack(sent[-1])
    assert magic == MAGIC_RESULT
    assert fid == 7
    assert (fx, fy, fw, fh, gid, conf) == (0, 0, 0, 0, 0, 0)


def test_unknown_gesture_name_falls_back_to_none():
    server, sent = make_server()
    gesture = GestureData(gesture='moonwalk', confidence=1.0, timestamp=0.0)
    server._send_result(1, None, gesture)
    *_, gid, _conf = RESULT_STRUCT.unpack(sent[-1])
    assert gid == GESTURE_IDS['none']


# ------------------------------------------------------------ live server

@pytest.fixture
def live_server(monkeypatch):
    """A real server on a real port, with detection stubbed out."""
    server = RemoteVisionServer(port=0)
    monkeypatch.setattr(server, "_init_vision", lambda w, h: None)
    monkeypatch.setattr(
        server, "_detect",
        lambda frame: (FaceData(x=100, y=50, width=40, height=40, timestamp=0.0),
                       GestureData(gesture='waving', confidence=1.0, timestamp=0.0)))

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(('127.0.0.1', 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    server._socket = listener
    server._running = True

    def serve():
        listener.settimeout(ACCEPT_POLL_S)
        while server._running:
            try:
                server._client, _addr = listener.accept()
                server._client.settimeout(SOCKET_TIMEOUT_S)
            except socket.timeout:
                continue
            except OSError:
                return
            server._start_time = time.monotonic()
            server._process_client()
            try:
                server._client.close()
            except OSError:
                pass
            server._client = None

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    yield server, port
    server._running = False
    listener.close()
    thread.join(timeout=3.0)


def send_frame(sock, payload, frame_id=1, width=64, height=48, claimed_len=None):
    header = FRAME_HEADER.pack(MAGIC_FRAME, frame_id, width, height,
                               claimed_len if claimed_len is not None else len(payload))
    sock.sendall(header + payload)


def recv_result(sock):
    """Read one result, or None if the server hung up (FIN or RST)."""
    data = b''
    while len(data) < RESULT_STRUCT.size:
        try:
            chunk = sock.recv(RESULT_STRUCT.size - len(data))
        except (ConnectionResetError, socket.timeout):
            return None
        if not chunk:
            return None
        data += chunk
    return RESULT_STRUCT.unpack(data)


def test_a_frame_gets_a_result_back(live_server):
    _server, port = live_server
    sock = socket.create_connection(('127.0.0.1', port), timeout=5)
    sock.settimeout(5)
    try:
        send_frame(sock, jpeg_of())
        magic, fid, fx, fy, fw, fh, gid, _conf = recv_result(sock)
        assert magic == MAGIC_RESULT
        assert fid == 1
        assert (fx, fy, fw, fh) == (100, 50, 40, 40)
        assert ID_TO_GESTURE[gid] == 'waving'
    finally:
        sock.close()


def test_several_frames_stay_in_lockstep(live_server):
    _server, port = live_server
    sock = socket.create_connection(('127.0.0.1', port), timeout=5)
    sock.settimeout(5)
    try:
        for frame_id in range(1, 6):
            send_frame(sock, jpeg_of(), frame_id=frame_id)
            _magic, fid, *_rest = recv_result(sock)
            assert fid == frame_id
    finally:
        sock.close()


def test_an_undecodable_frame_still_gets_an_answer(live_server):
    """Otherwise the client blocks until its socket timeout."""
    _server, port = live_server
    sock = socket.create_connection(('127.0.0.1', port), timeout=5)
    sock.settimeout(5)
    try:
        send_frame(sock, b"not a jpeg at all", frame_id=9)
        magic, fid, fx, fy, fw, fh, _gid, _conf = recv_result(sock)
        assert magic == MAGIC_RESULT
        assert fid == 9
        assert (fw, fh) == (0, 0)
    finally:
        sock.close()


def test_an_oversized_length_claim_is_refused(live_server):
    _server, port = live_server
    sock = socket.create_connection(('127.0.0.1', port), timeout=5)
    sock.settimeout(5)
    try:
        header = FRAME_HEADER.pack(MAGIC_FRAME, 1, 64, 48, MAX_JPEG_BYTES + 1)
        sock.sendall(header)
        assert recv_result(sock) is None
    finally:
        sock.close()


def test_a_zero_length_claim_is_refused(live_server):
    _server, port = live_server
    sock = socket.create_connection(('127.0.0.1', port), timeout=5)
    sock.settimeout(5)
    try:
        sock.sendall(FRAME_HEADER.pack(MAGIC_FRAME, 1, 64, 48, 0))
        assert recv_result(sock) is None
    finally:
        sock.close()


def test_a_bad_magic_drops_the_connection(live_server):
    _server, port = live_server
    sock = socket.create_connection(('127.0.0.1', port), timeout=5)
    sock.settimeout(5)
    try:
        sock.sendall(FRAME_HEADER.pack(b'\x00\x00', 1, 64, 48, 10) + b"0123456789")
        assert recv_result(sock) is None
    finally:
        sock.close()


def test_a_stalled_client_does_not_lock_out_the_next_one(live_server):
    """Regression: the accepted socket had no timeout, so this wedged the server."""
    _server, port = live_server
    staller = socket.create_connection(('127.0.0.1', port), timeout=5)
    staller.sendall(FRAME_HEADER.pack(MAGIC_FRAME, 1, 64, 48, 5000))  # never sends the body

    try:
        deadline = time.monotonic() + SOCKET_TIMEOUT_S + 5
        while time.monotonic() < deadline:
            try:
                nxt = socket.create_connection(('127.0.0.1', port), timeout=1)
            except OSError:
                time.sleep(0.2)
                continue
            nxt.settimeout(5)
            try:
                send_frame(nxt, jpeg_of(), frame_id=77)
                result = recv_result(nxt)
                if result is not None:
                    assert result[1] == 77
                    return
            except (socket.timeout, OSError):
                pass
            finally:
                nxt.close()
            time.sleep(0.2)
        pytest.fail("server never served a second client")
    finally:
        staller.close()


# ------------------------------------------------------------ live client

def test_client_reports_failure_and_clears_state_when_offline():
    """A dead link must not leave a stale face for the control loop to chase."""
    camera = StubCamera()
    client = RemoteVisionClient('127.0.0.1', 1, camera, reconnect_interval=0.0)
    client._cached_face = FaceData(x=1, y=2, width=3, height=4, timestamp=0.0)
    assert client.update() is False
    assert client.get_face() is None
    assert client.get_gesture().gesture == 'none'


def test_client_round_trip_against_the_real_server(live_server):
    _server, port = live_server
    camera = StubCamera()
    client = RemoteVisionClient('127.0.0.1', port, camera, reconnect_interval=0.0)
    try:
        assert client.start() is True
        assert client.update() is True
        face = client.get_face()
        assert (face.x, face.y, face.width, face.height) == (100, 50, 40, 40)
        assert client.get_gesture().gesture == 'waving'
        assert 0.0 <= client.get_gesture().confidence <= 1.0
        assert client.get_frame() is not None
    finally:
        client.stop()


def test_client_recovers_after_the_server_drops_it(live_server):
    _server, port = live_server
    camera = StubCamera()
    client = RemoteVisionClient('127.0.0.1', port, camera, reconnect_interval=0.0)
    try:
        assert client.start() is True
        assert client.update() is True
        client._socket.close()
        client.update()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if client.update():
                assert client.get_face() is not None
                return
            time.sleep(0.1)
        pytest.fail("client never reconnected")
    finally:
        client.stop()


def test_client_disconnects_on_a_desynced_reply():
    """A misaligned stream cannot resync, so the connection must be dropped."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(('127.0.0.1', 0))
    listener.listen(1)
    port = listener.getsockname()[1]

    def liar():
        conn, _ = listener.accept()
        conn.recv(65536)
        conn.sendall(struct.pack('<2sIhhHHBB', b'\xDE\xAD', 1, 0, 0, 0, 0, 0, 0))
        time.sleep(1.0)
        conn.close()

    thread = threading.Thread(target=liar, daemon=True)
    thread.start()

    client = RemoteVisionClient('127.0.0.1', port, StubCamera(), reconnect_interval=999)
    try:
        assert client._connect() is True
        assert client.update() is False
        assert client._connected is False
    finally:
        client._disconnect()
        listener.close()
        thread.join(timeout=3)
