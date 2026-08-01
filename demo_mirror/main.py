"""
Demo Mirror
===========

Face tracking + gesture mirroring demo.
Uses ainex_api for all hardware/perception.
"""

import sys
import time
import signal
import argparse
from typing import Optional

from ainex_api import Robot
from ainex_api.motion import MOTION_STOP_TIMEOUT_S

from .behavior import BehaviorController


class DemoMirror:
    """
    Face tracking + gesture mirroring demo.

    - Tracks faces with head movement
    - Mirrors arm movements
    - Responds to waving with greeting

    All hardware/perception handled by ainex_api.
    """

    def __init__(self, loop_rate: int = 10, action_cooldown: float = 4.0,
                 undistort: bool = False, distortion_k1: float = -0.15,
                 remote_vision: Optional[str] = None):
        print("=" * 60)
        print("Demo Mirror - Face Tracking + Gesture Mirroring")
        print("=" * 60)

        # Checked before the robot exists, so a bad rate cannot strand an open
        # serial port on the way to a ZeroDivisionError.
        if loop_rate <= 0:
            raise ValueError(f"loop_rate={loop_rate} must be positive")

        # Initialize robot with vision enabled
        self.robot = Robot(
            undistort_camera=undistort,
            distortion_k1=distortion_k1,
            remote_vision=remote_vision
        )
        vision_mode = "REMOTE → " + remote_vision if remote_vision else "LOCAL"
        print(f"[OK] Robot initialized (vision: {vision_mode})")

        # The robot owns the serial port from here, so nothing below may fail
        # without handing it back.
        try:
            self.behavior = BehaviorController(self.robot, action_cooldown)
            print("[OK] Behavior controller initialized")

            self.loop_rate = loop_rate
            self._loop_period = 1.0 / loop_rate

            self.running = True
            self._shutdown_requested = False

            signal.signal(signal.SIGINT, self._signal_handler)

            print("=" * 60)
            print(f"  Camera:     {self.robot._camera.width}x{self.robot._camera.height} "
                  f"@ {self.robot._camera.device}")
            print(f"  MediaPipe:  {'AVAILABLE' if self.robot.vision.has_mediapipe else 'NOT INSTALLED'}")
            print(f"  Loop Rate:  {loop_rate} Hz")
            print("=" * 60)
        except Exception:
            self.robot.close()
            raise

    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C"""
        if self._shutdown_requested:
            print("\n[FORCE SHUTDOWN]")
            sys.exit(1)

        self._shutdown_requested = True
        self.running = False
        print("\n[Shutdown requested...]")

    def _init_hardware(self):
        """Initialize hardware to safe state"""
        print("[Init] Centering head...")
        self.robot.head.center(duration=1.0, blocking=True)

        print("[Init] Standing...")
        self.robot.stand()

        print("[OK] Hardware ready")

    def run(self):
        """Run main control loop"""
        try:
            self._run()
        finally:
            self._cleanup()

    def _run(self):
        """Main control loop; run() guarantees cleanup around this."""
        self._init_hardware()

        # Start vision
        if not self.robot.vision.start():
            print("[ERROR] Failed to start camera!")
            print("Check that /dev/usb_cam exists")
            return

        print()
        print("=" * 60)
        print("MAIN LOOP ACTIVE")
        print(f"Loop rate: {self.loop_rate} Hz ({1000/self.loop_rate:.0f}ms)")
        print("Robot will:")
        print("  - Track your face with head movement")
        print("  - Mirror your arm movements (raise left/right/both)")
        print("  - Greet when you wave one hand")
        print("Press Ctrl+C to shutdown")
        print("=" * 60)
        print()

        frame_count = 0
        start_time = time.monotonic()

        while self.running:
            loop_start = time.monotonic()
            frame_count += 1

            # Update vision (captures frame, runs detection)
            self.robot.vision.update()

            # Debug first 5 frames
            if frame_count <= 5:
                face = self.robot.vision.get_face()
                if face:
                    print(f"[F{frame_count}] FACE at ({face.x}, {face.y})")
                else:
                    print(f"[F{frame_count}] no face")

            # Run behavior (reads from vision, controls robot)
            self.behavior.update()

            # Status output every 3 seconds
            if frame_count % (self.loop_rate * 3) == 0:
                elapsed = time.monotonic() - start_time
                fps = frame_count / elapsed
                state = self.behavior.current_state.name
                gesture = self.robot.vision.get_gesture()
                g = gesture.gesture if gesture else 'none'
                face = self.robot.vision.get_face()
                print(f"[{elapsed:.0f}s] FPS:{fps:.1f} State:{state} Face:{face is not None} Gesture:{g}")

            # Rate limiting
            elapsed = time.monotonic() - loop_start
            if elapsed < self._loop_period:
                time.sleep(self._loop_period - elapsed)

    def _cleanup(self):
        """Cleanup on shutdown"""
        print()
        print("=" * 60)
        print("SHUTTING DOWN")
        print("=" * 60)

        # Print stats
        stats = self.behavior.stats
        print(f"Frames processed: {stats['frames']}")
        print(f"Gestures mirrored: {stats['gestures_mirrored']}")
        print(f"State transitions: {stats['state_transitions']}")

        # Reset hardware. Stop first: play() refuses while a motion is in flight.
        print("[Cleanup] Returning to stand pose...")
        self.robot.motion.stop()
        self.robot.motion.wait(timeout=MOTION_STOP_TIMEOUT_S)
        if not self.robot.stand():
            print("[Cleanup] WARNING: stand pose refused")
        time.sleep(1.0)

        print("[Cleanup] Centering head...")
        self.robot.head.center(duration=1.0, blocking=True)

        self.robot.close()

        print("=" * 60)
        print("SHUTDOWN COMPLETE")
        print("=" * 60)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Demo Mirror - Face Tracking + Gesture Mirroring')
    parser.add_argument('--loop-rate', type=int, default=10, help='Loop rate in Hz')
    parser.add_argument('--action-cooldown', type=float, default=4.0,
                        help='Cooldown between greet/wave actions')
    parser.add_argument('--undistort', action='store_true',
                        help='Enable lens distortion correction')
    parser.add_argument('--distortion-k1', type=float, default=-0.15,
                        help='Radial distortion coefficient (negative=barrel)')
    parser.add_argument('--remote-vision', type=str, default=None,
                        help='Remote vision server "host:port" (run server on PC first)')
    args = parser.parse_args()

    try:
        demo = DemoMirror(
            loop_rate=args.loop_rate,
            action_cooldown=args.action_cooldown,
            undistort=args.undistort,
            distortion_k1=args.distortion_k1,
            remote_vision=args.remote_vision
        )
        demo.run()

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
