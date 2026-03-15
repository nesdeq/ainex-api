"""
Motion Player
=============

Plays motion sequences from d6a database files.
Supports both synchronous and asynchronous playback.
"""

import os
import time
import sqlite3
import threading
from typing import Optional, List, Dict, Callable
from dataclasses import dataclass
from .board import Board


@dataclass
class MotionFrame:
    """Single frame of a motion sequence"""
    frame_id: int
    duration_ms: int
    positions: List[int]  # 22 servo positions (servos 1-22)


@dataclass
class Motion:
    """Complete motion sequence"""
    name: str
    frames: List[MotionFrame]

    @property
    def total_duration_ms(self) -> int:
        return sum(f.duration_ms for f in self.frames)

    @property
    def frame_count(self) -> int:
        return len(self.frames)


class MotionPlayer:
    """
    Plays motion sequences from d6a database files.

    Supports:
    - Synchronous playback (blocking)
    - Asynchronous playback (non-blocking)
    - Motion caching
    - Custom timing multiplier
    """

    def __init__(self, board: Board, motion_dir: Optional[str] = None):
        self.board = board
        self.motion_dir = motion_dir or os.path.join(
            os.path.dirname(__file__), 'motions'
        )

        # Playback state
        self._playing = False
        self._stop_requested = False
        self._current_motion: Optional[str] = None

        # Motion cache
        self._cache: Dict[str, Motion] = {}

        # Timing (0.8 = wait 80% of duration, faster transitions)
        self.timing_multiplier = 0.8

    @property
    def is_playing(self) -> bool:
        """Check if motion is currently playing"""
        return self._playing

    @property
    def current_motion(self) -> Optional[str]:
        """Get name of currently playing motion"""
        return self._current_motion if self._playing else None

    def load(self, name: str) -> Optional[Motion]:
        """
        Load motion from d6a file.

        Args:
            name: Motion name (without .d6a extension)

        Returns:
            Motion object or None if not found
        """
        # Check cache
        if name in self._cache:
            return self._cache[name]

        # Find file
        path = os.path.join(self.motion_dir, f"{name}.d6a")
        if not os.path.exists(path):
            return None

        # Load from database
        try:
            conn = sqlite3.connect(path)
            cursor = conn.execute("SELECT * FROM ActionGroup")
            frames = []

            for row in cursor:
                frame_id = row[0]
                duration_ms = row[1]
                positions = list(row[2:24])  # Columns 2-23 = servos 1-22

                frames.append(MotionFrame(
                    frame_id=frame_id,
                    duration_ms=duration_ms,
                    positions=positions
                ))

            conn.close()

            motion = Motion(name=name, frames=frames)
            self._cache[name] = motion
            return motion

        except Exception as e:
            print(f"[Motion] Failed to load {name}: {e}")
            return None

    def play(self, name: str, blocking: bool = True,
             on_complete: Optional[Callable] = None) -> bool:
        """
        Play a motion sequence.

        Args:
            name: Motion name (without .d6a extension)
            blocking: Wait for completion if True
            on_complete: Callback when motion completes (async only)

        Returns:
            True if motion started, False if not found or already playing
        """
        if self._playing:
            return False

        motion = self.load(name)
        if motion is None:
            return False

        if blocking:
            self._play_sync(motion)
        else:
            thread = threading.Thread(
                target=self._play_async,
                args=(motion, on_complete),
                daemon=True
            )
            thread.start()

        return True

    def _play_sync(self, motion: Motion):
        """Synchronous playback"""
        self._playing = True
        self._current_motion = motion.name
        self._stop_requested = False

        try:
            for frame in motion.frames:
                if self._stop_requested:
                    break

                positions = [[i + 1, pos] for i, pos in enumerate(frame.positions)]

                # Send command
                self.board.bus_servo_set_position(frame.duration_ms / 1000.0, positions)

                # Wait (with timing multiplier for faster transitions)
                wait_time = (frame.duration_ms / 1000.0) * self.timing_multiplier
                time.sleep(wait_time)

        finally:
            self._playing = False
            self._current_motion = None

    def _play_async(self, motion: Motion, on_complete: Optional[Callable]):
        """Asynchronous playback"""
        self._play_sync(motion)
        if on_complete:
            on_complete()

    def stop(self):
        """Stop current motion"""
        self._stop_requested = True

    def wait(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for current motion to complete.

        Args:
            timeout: Maximum wait time in seconds (None = infinite)

        Returns:
            True if motion completed, False if timeout
        """
        start = time.time()
        while self._playing:
            if timeout and (time.time() - start) > timeout:
                return False
            time.sleep(0.01)
        return True

    def list_motions(self) -> List[str]:
        """List available motion names"""
        if not os.path.exists(self.motion_dir):
            return []
        return [f[:-4] for f in os.listdir(self.motion_dir) if f.endswith('.d6a')]

    def clear_cache(self):
        """Clear motion cache"""
        self._cache.clear()

    def set_servos_direct(self, duration_ms: int, positions: List[List[int]]):
        """
        Set servo positions directly (bypass motion system).

        Args:
            duration_ms: Movement duration in milliseconds
            positions: List of [servo_id, position] pairs
        """
        self.board.bus_servo_set_position(duration_ms / 1000.0, positions)
