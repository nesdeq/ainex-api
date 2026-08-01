"""
Remote Vision System
====================

Split vision processing: camera on robot, MediaPipe on PC.

Usage:
    # On PC (run server):
    python -m ainex_api.remote_vision --port 9999

    # On Robot:
    python run.py --remote-vision PC_IP:9999

    # Or programmatically:
    robot = Robot(remote_vision="PC_IP:9999")
"""

import cv2
import time
import socket
import struct
import numpy as np
from typing import Optional, Tuple

from .vision import FaceData, GestureData, VisionSystem, HAS_MEDIAPIPE
from .camera import Camera


# Protocol constants
MAGIC_FRAME = b'\xAA\x55'
MAGIC_RESULT = b'\x55\xAA'

# Frames arrive from the network. A 640x480 JPEG at quality 80 runs 30-60 KB;
# this caps the buffer a client can make the server allocate.
MAX_JPEG_BYTES = 4 * 1024 * 1024

# Domain of the result struct's coordinate fields.
_I16_MIN, _I16_MAX = -32768, 32767
_U16_MAX = 65535

# Gesture ID mapping
GESTURE_IDS = {
    'none': 0,
    'left_hand_raised': 1,
    'right_hand_raised': 2,
    'both_hands_raised': 3,
    'waving': 4,
}
ID_TO_GESTURE = {v: k for k, v in GESTURE_IDS.items()}

# Struct formats (little-endian)
# Frame header: magic(2) + frame_id(4) + width(2) + height(2) + jpeg_len(4) = 14 bytes
FRAME_HEADER = struct.Struct('<2sIHHI')

# Result: magic(2) + frame_id(4) + face_x(2) + face_y(2) + face_w(2) + face_h(2) + gesture(1) + confidence(1) = 16 bytes
RESULT_STRUCT = struct.Struct('<2sIhhHHBB')


class RemoteVisionClient:
    """
    Runs on robot. Captures frames, sends to server, receives results.
    Same interface as VisionSystem - drop-in replacement.
    """

    def __init__(self, host: str, port: int, camera: Camera,
                 reconnect_interval: float = 2.0):
        self.camera = camera
        self._host = host
        self._port = port
        self._reconnect_interval = reconnect_interval

        self._socket: Optional[socket.socket] = None
        self._connected = False
        self._running = False

        # Cached results (same interface as VisionSystem)
        self._last_frame: Optional[np.ndarray] = None
        self._last_frame_hw = (480, 640)
        self._cached_face: Optional[FaceData] = None
        self._cached_gesture: Optional[GestureData] = None

        # Stats
        self._frame_id = 0
        self._last_reconnect = 0

    def start(self) -> bool:
        """Start camera and connect to server."""
        if not self.camera.start():
            print(f"[RemoteVision] ERROR: Camera failed to start")
            return False

        self._running = True

        if not self._connect():
            print(f"[RemoteVision] ERROR: Could not connect to {self._host}:{self._port}")
            print(f"[RemoteVision] Make sure server is running: python -m ainex_api.remote_vision --port {self._port}")
            self.camera.stop()
            self._running = False
            return False

        return True

    def stop(self):
        """Stop camera and disconnect."""
        self._running = False
        self._disconnect()
        self.camera.stop()

    def _connect(self) -> bool:
        """Establish connection to vision server."""
        if self._connected:
            return True

        now = time.monotonic()
        if now - self._last_reconnect < self._reconnect_interval:
            return False

        self._last_reconnect = now

        print(f"[RemoteVision] Connecting to {self._host}:{self._port}...")

        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._socket.settimeout(5.0)
            self._socket.connect((self._host, self._port))
            self._socket.settimeout(10.0)  # Longer timeout for recv after connected
            self._connected = True
            print(f"[RemoteVision] Connected!")
            return True
        except socket.timeout:
            print(f"[RemoteVision] Connection timed out")
            self._socket = None
            self._connected = False
            return False
        except ConnectionRefusedError:
            print(f"[RemoteVision] Connection refused - is server running?")
            self._socket = None
            self._connected = False
            return False
        except OSError as e:
            print(f"[RemoteVision] Connection failed: {e}")
            self._socket = None
            self._connected = False
            return False

    def _disconnect(self):
        """Close connection."""
        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass
        self._socket = None
        self._connected = False

    def _send_frame(self, frame: np.ndarray) -> bool:
        """Encode and send frame to server."""
        if not self._connected:
            return False

        try:
            # JPEG encode (quality 80 balances size/quality)
            _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            jpeg_bytes = jpeg.tobytes()

            h, w = frame.shape[:2]
            header = FRAME_HEADER.pack(MAGIC_FRAME, self._frame_id, w, h, len(jpeg_bytes))

            # Send header + data
            self._socket.sendall(header + jpeg_bytes)
            return True

        except (socket.error, BrokenPipeError) as e:
            print(f"[RemoteVision] Send error: {e}")
            self._disconnect()
            return False

    def _recv_result(self) -> bool:
        """Receive detection result from server."""
        if not self._connected:
            return False

        try:
            # Receive fixed-size result
            data = b''
            while len(data) < RESULT_STRUCT.size:
                chunk = self._socket.recv(RESULT_STRUCT.size - len(data))
                if not chunk:
                    raise ConnectionError("Server closed connection")
                data += chunk

            # Unpack result
            magic, frame_id, face_x, face_y, face_w, face_h, gesture_id, confidence = \
                RESULT_STRUCT.unpack(data)

            if magic != MAGIC_RESULT:
                print(f"[RemoteVision] Bad magic: {magic}")
                return False

            now = time.time()

            # Update cached face
            if face_w > 0 and face_h > 0:
                self._cached_face = FaceData(
                    x=face_x, y=face_y,
                    width=face_w, height=face_h,
                    timestamp=now
                )
            else:
                self._cached_face = None

            # Update cached gesture
            gesture_name = ID_TO_GESTURE.get(gesture_id, 'none')
            self._cached_gesture = GestureData(
                gesture=gesture_name,
                confidence=confidence / 100.0,
                timestamp=now
            )

            return True

        except (socket.error, socket.timeout, ConnectionError) as e:
            print(f"[RemoteVision] Recv error: {e}")
            self._disconnect()
            return False

    def update(self) -> bool:
        """
        Capture frame, send to server, receive results.
        Same interface as VisionSystem.update().
        """
        # Try reconnect if disconnected
        if not self._connected:
            self._connect()

        # Capture frame
        self._last_frame = self.camera.read()
        if self._last_frame is None:
            return False

        self._last_frame_hw = self._last_frame.shape[:2]
        self._frame_id = (self._frame_id + 1) & 0xFFFFFFFF

        # If not connected, clear results and return
        if not self._connected:
            self._cached_face = None
            self._cached_gesture = GestureData(gesture='none', confidence=0, timestamp=time.time())
            return True  # Frame captured, just no detection

        # Send frame and receive result
        if self._send_frame(self._last_frame):
            self._recv_result()

        return True

    def get_face(self) -> Optional[FaceData]:
        """Get cached face detection result."""
        return self._cached_face

    def get_gesture(self) -> Optional[GestureData]:
        """Get cached gesture detection result."""
        return self._cached_gesture

    def get_frame(self) -> Optional[np.ndarray]:
        """Get raw BGR frame from last update()."""
        return self._last_frame

    @property
    def has_mediapipe(self) -> bool:
        """Remote server has MediaPipe."""
        return True


class RemoteVisionServer:
    """
    Runs on PC. Receives frames, runs MediaPipe, sends results.

    Usage:
        server = RemoteVisionServer(port=9999)
        server.run()  # Blocks, Ctrl+C to stop
    """

    def __init__(self, port: int = 9999, pose_model: int = 2):
        """
        Args:
            port: Listen port
            pose_model: 0=Lite, 1=Full, 2=Heavy (default: 2 for PC)
        """
        self._port = port
        self._pose_model = pose_model

        self._socket: Optional[socket.socket] = None
        self._client: Optional[socket.socket] = None
        self._running = False

        # Vision system (will use DummyCamera since we receive frames)
        self._vision: Optional[VisionSystem] = None

        # Stats
        self._frames_processed = 0
        self._start_time = None

    def _init_vision(self, width: int, height: int):
        """Initialize vision system for given frame size."""
        dummy_camera = _DummyCamera(width, height)
        self._vision = VisionSystem(dummy_camera, pose_model=self._pose_model)

    def run(self):
        """Run server (blocking)."""
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._socket.bind(('0.0.0.0', self._port))
        self._socket.listen(1)

        pose_models = {0: 'Lite', 1: 'Full', 2: 'Heavy'}

        print(f"[RemoteVisionServer] Listening on port {self._port}")
        print(f"[RemoteVisionServer] Pose model: {pose_models.get(self._pose_model, '?')}")
        print("[RemoteVisionServer] Waiting for connection...")

        self._running = True

        try:
            while self._running:
                # Accept connection
                self._socket.settimeout(1.0)
                try:
                    if self._client:
                        self._client.close()
                    self._client, addr = self._socket.accept()
                    self._client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    print(f"[RemoteVisionServer] Client connected: {addr}")
                except socket.timeout:
                    continue

                self._start_time = time.monotonic()
                self._frames_processed = 0

                # Process frames from this client
                self._process_client()

                print(f"[RemoteVisionServer] Client disconnected. Processed {self._frames_processed} frames")

        except KeyboardInterrupt:
            print("\n[RemoteVisionServer] Shutting down...")
        finally:
            self._running = False
            if self._client:
                self._client.close()
            if self._socket:
                self._socket.close()

    def _process_client(self):
        """Process frames from connected client."""
        while self._running:
            try:
                # Receive frame header
                header_data = self._recv_exact(FRAME_HEADER.size)
                if not header_data:
                    break

                magic, frame_id, width, height, jpeg_len = FRAME_HEADER.unpack(header_data)

                if magic != MAGIC_FRAME:
                    print(f"[RemoteVisionServer] Bad magic: {magic}")
                    break

                if not 0 < jpeg_len <= MAX_JPEG_BYTES:
                    print(f"[RemoteVisionServer] Rejecting frame: jpeg_len={jpeg_len} "
                          f"(limit {MAX_JPEG_BYTES})")
                    break

                # Receive JPEG data
                jpeg_data = self._recv_exact(jpeg_len)
                if not jpeg_data:
                    break

                # Decode frame
                frame = cv2.imdecode(np.frombuffer(jpeg_data, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    print("[RemoteVisionServer] Failed to decode frame")
                    continue

                # Initialize vision system on first frame
                if self._vision is None:
                    self._init_vision(width, height)

                # Run detection
                face, gesture = self._detect(frame)

                # Send result
                self._send_result(frame_id, face, gesture)

                self._frames_processed += 1

                # Stats every 100 frames
                if self._frames_processed % 100 == 0:
                    elapsed = time.monotonic() - self._start_time
                    fps = self._frames_processed / elapsed
                    print(f"[RemoteVisionServer] {self._frames_processed} frames, {fps:.1f} FPS")

            except (socket.error, ConnectionError, struct.error) as e:
                print(f"[RemoteVisionServer] Client error: {e}")
                break

    def _recv_exact(self, size: int) -> Optional[bytes]:
        """Receive exactly size bytes."""
        data = b''
        while len(data) < size:
            try:
                chunk = self._client.recv(size - len(data))
                if not chunk:
                    return None
                data += chunk
            except socket.timeout:
                continue
            except socket.error:
                return None
        return data

    def _detect(self, frame: np.ndarray) -> Tuple[Optional[FaceData], Optional[GestureData]]:
        """Run detection on frame."""
        # Feed frame to dummy camera
        self._vision.camera._frame = frame
        self._vision.camera._frame_count += 1

        # Run detection
        self._vision.update()

        return self._vision.get_face(), self._vision.get_gesture()

    def _send_result(self, frame_id: int, face: Optional[FaceData], gesture: Optional[GestureData]):
        """Send detection result to client."""
        # Pack face data; the client controls the frame size, so pin every
        # coordinate to the field domain rather than letting the pack raise.
        if face:
            face_x = min(max(face.x, _I16_MIN), _I16_MAX)
            face_y = min(max(face.y, _I16_MIN), _I16_MAX)
            face_w = min(max(face.width, 0), _U16_MAX)
            face_h = min(max(face.height, 0), _U16_MAX)
        else:
            face_x, face_y, face_w, face_h = 0, 0, 0, 0

        # Pack gesture data
        if gesture:
            gesture_id = GESTURE_IDS.get(gesture.gesture, 0)
            confidence = min(max(int(gesture.confidence * 100), 0), 0xFF)
        else:
            gesture_id = 0
            confidence = 0

        result = RESULT_STRUCT.pack(
            MAGIC_RESULT, frame_id,
            face_x, face_y, face_w, face_h,
            gesture_id, confidence
        )

        try:
            self._client.sendall(result)
        except socket.error:
            pass


class _DummyCamera:
    """Fake camera for server-side VisionSystem."""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self._frame = None
        self._frame_count = 0
        self._running = True

    def start(self) -> bool:
        return True

    def stop(self):
        self._running = False

    def read(self) -> Optional[np.ndarray]:
        return self._frame

    @property
    def is_open(self) -> bool:
        return self._running


def main():
    """CLI entry point for server."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Remote Vision Server - run on PC for fast MediaPipe processing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m ainex_api.remote_vision                    # Max quality (default)
  python -m ainex_api.remote_vision --pose-model 1    # Full pose model (faster)
"""
    )
    parser.add_argument('--port', type=int, default=9999, help='Listen port (default: 9999)')
    parser.add_argument('--pose-model', type=int, default=2, choices=[0, 1, 2],
                        help='Pose model: 0=Lite, 1=Full, 2=Heavy (default: 2)')
    args = parser.parse_args()

    if not HAS_MEDIAPIPE:
        print("ERROR: MediaPipe not installed. Run: pip install mediapipe")
        return 1

    server = RemoteVisionServer(port=args.port, pose_model=args.pose_model)
    server.run()
    return 0


if __name__ == "__main__":
    exit(main())
