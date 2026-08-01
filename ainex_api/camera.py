"""
AINEX Camera Module
===================

Camera capture from the robot's USB camera.
Hardcoded to /dev/usb_cam - the standard AINEX camera device.

Features:
- Background thread capture for non-blocking operation
- Optional lens distortion correction (barrel/fisheye)
- Configurable resolution
"""

import cv2
import time
import threading
import numpy as np
from typing import Optional


# Hardware default - AINEX USB camera
DEFAULT_CAMERA_DEVICE = "/dev/usb_cam"
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
CAMERA_FALLBACKS = ["/dev/video0", "/dev/video1", 0, 1]


def open_camera(device: Optional[str] = None, width: Optional[int] = None,
                height: Optional[int] = None):
    """
    Open camera with fallback support.

    Args:
        device: Primary device path (default: /dev/usb_cam)
        width: Frame width (default: 640)
        height: Frame height (default: 480)

    Returns:
        Tuple of (VideoCapture, actual_width, actual_height) or (None, 0, 0) if failed
    """
    device = device or DEFAULT_CAMERA_DEVICE
    width = width or DEFAULT_WIDTH
    height = height or DEFAULT_HEIGHT

    cap = cv2.VideoCapture(device)

    if not cap.isOpened():
        for fb in CAMERA_FALLBACKS:
            if fb == device:
                continue
            cap = cv2.VideoCapture(fb)
            if cap.isOpened():
                print(f"[Camera] Using fallback: {fb}")
                break

    if not cap.isOpened():
        return None, 0, 0

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    return cap, actual_w, actual_h


class Camera:
    """
    Camera capture from AINEX USB camera.

    Runs capture in background thread for non-blocking operation.
    Default device: /dev/usb_cam (hardcoded for AINEX)
    Default resolution: 640x480

    Lens Correction:
        Many USB cameras have slight barrel distortion. Enable correction with:
            camera = Camera(undistort=True, distortion_k1=-0.15)

        The k1 parameter controls radial distortion:
        - k1 < 0: corrects barrel distortion (edges curve inward)
        - k1 > 0: corrects pincushion distortion (edges curve outward)
        - Typical values for USB webcams: -0.1 to -0.25
    """

    def __init__(self, device: Optional[str] = None, width: Optional[int] = None,
                 height: Optional[int] = None,
                 undistort: bool = False, distortion_k1: float = -0.15):
        """
        Initialize camera.

        Args:
            device: Camera device path (default: /dev/usb_cam)
            width: Frame width (default: 640)
            height: Frame height (default: 480)
            undistort: Enable lens distortion correction
            distortion_k1: Primary radial distortion coefficient
                          Negative for barrel, positive for pincushion
        """
        self.device = device or DEFAULT_CAMERA_DEVICE
        self.width = width or DEFAULT_WIDTH
        self.height = height or DEFAULT_HEIGHT

        self._cap = None
        self._frame = None
        self._frame_lock = threading.Lock()
        self._running = False
        self._thread = None
        self._frame_count = 0
        self._start_time = None

        # Lens distortion correction
        self._undistort = undistort
        self._distortion_k1 = distortion_k1
        self._map1 = None
        self._map2 = None

    def _init_undistort_maps(self):
        """
        Precompute undistortion maps for efficiency.

        Uses standard radial distortion model with single k1 coefficient.
        Maps are computed once and reused for every frame.
        """
        h, w = self.height, self.width

        # Camera matrix (intrinsic parameters)
        # Assume principal point at center, focal length ~500 for typical USB cam
        fx = fy = w * 0.8  # Approximate focal length
        cx, cy = w / 2, h / 2

        camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        # Distortion coefficients: [k1, k2, p1, p2, k3]
        # Only use k1 for simple barrel correction
        dist_coeffs = np.array([
            self._distortion_k1,  # k1: primary radial
            0.0,                  # k2: secondary radial
            0.0,                  # p1: tangential
            0.0,                  # p2: tangential
            0.0                   # k3: tertiary radial
        ], dtype=np.float32)

        # alpha=0.5 balances cropping against keeping edge pixels
        new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), alpha=0.5
        )

        # Precompute undistortion maps
        self._map1, self._map2 = cv2.initUndistortRectifyMap(
            camera_matrix, dist_coeffs, None, new_camera_matrix,
            (w, h), cv2.CV_16SC2
        )

        print(f"[Camera] Lens correction enabled (k1={self._distortion_k1})")

    def start(self) -> bool:
        """Start camera capture"""
        if self._running:
            return True

        self._cap, actual_w, actual_h = open_camera(self.device, self.width, self.height)

        if self._cap is None:
            print(f"[Camera] FAILED to open {self.device}")
            return False

        self.width = actual_w
        self.height = actual_h

        # Initialize undistortion maps if enabled
        if self._undistort:
            self._init_undistort_maps()

        self._running = True
        self._start_time = time.monotonic()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        print(f"[Camera] Started: {self.device} @ {actual_w}x{actual_h}")
        return True

    def stop(self):
        """Stop camera capture"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._cap:
            self._cap.release()
        self._cap = None
        print("[Camera] Stopped")

    def _capture_loop(self):
        """Background capture thread"""
        while self._running and self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            if ret:
                # Apply lens correction if enabled
                if self._undistort and self._map1 is not None:
                    frame = cv2.remap(frame, self._map1, self._map2, cv2.INTER_LINEAR)

                with self._frame_lock:
                    self._frame = frame
                    self._frame_count += 1
            else:
                time.sleep(0.001)

    def read(self) -> Optional[np.ndarray]:
        """
        Get latest frame (thread-safe).

        Returns:
            BGR frame as numpy array, or None if no frame available
        """
        with self._frame_lock:
            if self._frame is not None:
                return self._frame.copy()
        return None

    @property
    def is_open(self) -> bool:
        """Check if camera is running"""
        return self._running and self._cap is not None and self._cap.isOpened()

    @property
    def fps(self) -> float:
        """Get current capture FPS"""
        if self._start_time and self._frame_count > 0:
            elapsed = time.monotonic() - self._start_time
            return self._frame_count / elapsed if elapsed > 0 else 0
        return 0

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
