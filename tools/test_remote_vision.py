#!/usr/bin/env python3
"""
Test remote vision connection.

Usage:
    python tools/test_remote_vision.py 192.168.0.3:9999
    python tools/test_remote_vision.py 192.168.0.3        # uses port 9999
"""

import sys
import os
import time
import socket
import struct

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ainex_api.remote_vision import MAGIC_FRAME, MAGIC_RESULT, RESULT_STRUCT, ID_TO_GESTURE

RESULT_SIZE = RESULT_STRUCT.size


def test_connection(host: str, port: int) -> bool:
    """Test TCP connection."""
    print(f"[1] Testing connection to {host}:{port}...")
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(5)
        s.connect((host, port))
        s.close()
        print(f"    OK")
        return True
    except socket.timeout:
        print(f"    FAIL - Timeout")
        return False
    except ConnectionRefusedError:
        print(f"    FAIL - Connection refused (server not running?)")
        return False
    except OSError as e:
        print(f"    FAIL - {e}")
        return False


def test_camera() -> tuple:
    """Test camera, return (success, capture, frame)."""
    print(f"[2] Testing camera...")
    try:
        from ainex_api.camera import open_camera
        cap, w, h = open_camera()

        if cap is None:
            print(f"    FAIL - No camera")
            return False, None, None

        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"    OK - {w}x{h}")
            return True, cap, frame
        else:
            print(f"    FAIL - No frame")
            cap.release()
            return False, None, None

    except ImportError:
        print(f"    FAIL - OpenCV not installed")
        return False, None, None
    except Exception as e:
        print(f"    FAIL - {e}")
        return False, None, None


def test_roundtrip(host: str, port: int, cap, frame) -> bool:
    """Test sending frame and receiving result."""
    print(f"[3] Testing round-trip...")
    try:
        import cv2

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.settimeout(10)
        s.connect((host, port))

        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        jpeg_bytes = jpeg.tobytes()
        h, w = frame.shape[:2]

        header = struct.pack('<2sIHHI', MAGIC_FRAME, 1, w, h, len(jpeg_bytes))

        start = time.monotonic()
        s.sendall(header + jpeg_bytes)

        data = b''
        while len(data) < RESULT_SIZE:
            chunk = s.recv(RESULT_SIZE - len(data))
            if not chunk:
                raise ConnectionError("Server closed")
            data += chunk

        elapsed = (time.monotonic() - start) * 1000

        magic, frame_id, face_x, face_y, face_w, face_h, gesture_id, conf = \
            RESULT_STRUCT.unpack(data)

        if magic != MAGIC_RESULT:
            print(f"    FAIL - Bad response")
            return False

        print(f"    OK - {elapsed:.1f}ms")
        if face_w > 0:
            print(f"    Face: ({face_x},{face_y}) {face_w}x{face_h}")
        else:
            print(f"    Face: none")
        print(f"    Gesture: {ID_TO_GESTURE.get(gesture_id, '?')}")

        s.close()
        return True

    except Exception as e:
        print(f"    FAIL - {e}")
        return False


def test_latency(host: str, port: int, cap, count: int = 20) -> bool:
    """Test latency over multiple frames."""
    print(f"[4] Testing latency ({count} frames)...")
    try:
        import cv2

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.settimeout(10)
        s.connect((host, port))

        latencies = []
        for i in range(count):
            ret, frame = cap.read()
            if not ret:
                continue

            _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            jpeg_bytes = jpeg.tobytes()
            h, w = frame.shape[:2]

            header = struct.pack('<2sIHHI', MAGIC_FRAME, i, w, h, len(jpeg_bytes))

            start = time.monotonic()
            s.sendall(header + jpeg_bytes)

            data = b''
            while len(data) < RESULT_SIZE:
                chunk = s.recv(RESULT_SIZE - len(data))
                if not chunk:
                    break
                data += chunk

            if len(data) == RESULT_SIZE:
                latencies.append((time.monotonic() - start) * 1000)

        s.close()

        if latencies:
            avg = sum(latencies) / len(latencies)
            print(f"    OK - Avg: {avg:.1f}ms, Min: {min(latencies):.1f}ms, Max: {max(latencies):.1f}ms")
            return True
        else:
            print(f"    FAIL - No successful frames")
            return False

    except Exception as e:
        print(f"    FAIL - {e}")
        return False


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    addr = sys.argv[1]
    if ':' in addr:
        host, port = addr.split(':')
        port = int(port)
    else:
        host = addr
        port = 9999

    print()
    print("=" * 40)
    print(f"Remote Vision Test")
    print(f"Server: {host}:{port}")
    print("=" * 40)
    print()

    # Test 1: Connection
    if not test_connection(host, port):
        return 1

    # Test 2: Camera
    ok, cap, frame = test_camera()
    if not ok:
        return 1

    # Test 3: Round-trip
    if not test_roundtrip(host, port, cap, frame):
        cap.release()
        return 1

    # Test 4: Latency
    test_latency(host, port, cap)

    cap.release()

    print()
    print("=" * 40)
    print("All tests passed!")
    print("=" * 40)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
