#!/usr/bin/env python3
"""
AINEX Control Panel - NiceGUI version

Usage: python tools/ainexCP.py

Based on NiceGUI's recommended video streaming pattern:
https://github.com/zauberzeug/nicegui/blob/main/examples/opencv_webcam/main.py
"""

import sys
import os

# Suppress TensorFlow/MediaPipe noise - must be before any imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import time
import cv2
import atexit
import base64

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nicegui import ui, app, run
from ainex_api import Robot
from ainex_api.motion import MOTION_STOP_TIMEOUT_S
from ainex_api.head import PAN_MIN, PAN_MAX, TILT_MIN, TILT_MAX

# Global state
robot: Robot = None
shutdown_done = False

# Debug overlay flags - module level, shared by all
debug_overlay = {'face': False, 'gesture': False, 'pose': False}


def get_robot() -> Robot:
    global robot
    if robot is None:
        robot = Robot()
        robot.vision.start()
    return robot


def graceful_shutdown():
    """Shutdown robot gracefully."""
    global shutdown_done
    if robot is None or shutdown_done:
        return
    shutdown_done = True

    print("\n[Shutdown] Centering head...")
    try:
        robot.motion.stop()   # play() refuses while a motion is in flight
        robot.motion.wait(timeout=MOTION_STOP_TIMEOUT_S)
        robot.head.center(duration=1.0)
        time.sleep(1.0)
        print("[Shutdown] Standing...")
        robot.stand()
        time.sleep(1.5)
        print("[Shutdown] Closing...")
        robot.close()  # This stops vision internally
        print("[Shutdown] Complete")
    except Exception:
        pass  # Ignore errors during shutdown


atexit.register(graceful_shutdown)


def grab_frame() -> bytes:
    """Capture frame, apply overlays, return as JPEG bytes."""
    r = get_robot()

    # Update vision (capture + detection)
    r.vision.update()

    # Get frame with or without debug overlays
    if debug_overlay['face'] or debug_overlay['gesture'] or debug_overlay['pose']:
        frame = r.vision.draw_debug(
            face=debug_overlay['face'],
            gesture=debug_overlay['gesture'],
            pose=debug_overlay['pose']
        )
    else:
        frame = r.vision.get_frame()

    if frame is None:
        return None

    # Encode to JPEG
    _, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return jpg.tobytes()


# Initialize robot at startup
@app.on_startup
async def startup():
    get_robot()
    print("[OK] Robot initialized")


# UI Page
@ui.page('/')
async def main_page():
    r = get_robot()

    ui.dark_mode().enable()

    with ui.column().classes('w-full items-center p-4'):
        ui.label('🤖 AINEX').classes('text-2xl font-bold mb-2')

        # Debug toggles
        def toggle(key):
            def handler(e):
                debug_overlay[key] = e.value
            return handler

        with ui.row().classes('w-full max-w-4xl gap-4 mb-2'):
            ui.switch('Face', on_change=toggle('face'))
            ui.switch('Gesture', on_change=toggle('gesture'))
            ui.switch('Pose', on_change=toggle('pose'))

        # Camera + controls
        with ui.row().classes('w-full max-w-4xl'):
            with ui.column().classes('flex-grow'):
                # Use interactive_image for efficient updates
                video_image = ui.interactive_image().classes('w-full rounded-lg')

                # Timer updates frame - runs grab_frame in thread to avoid blocking
                async def update_frame():
                    if shutdown_done:
                        return
                    jpg_bytes = await run.io_bound(grab_frame)
                    if jpg_bytes:
                        b64 = base64.b64encode(jpg_bytes).decode('utf-8')
                        video_image.set_source(f'data:image/jpeg;base64,{b64}')

                ui.timer(0.05, update_frame)  # ~20fps

            with ui.column().classes('ml-4'):
                ui.label('Movement').classes('font-bold')
                with ui.column().classes('items-center'):
                    with ui.row():
                        ui.label().classes('w-12')
                        ui.button('⬆️', on_click=r.walk_forward).classes('w-12 h-12')
                        ui.label().classes('w-12')
                    with ui.row():
                        ui.button('↩️', on_click=r.turn_left).classes('w-12 h-12')
                        ui.label().classes('w-12')
                        ui.button('↪️', on_click=r.turn_right).classes('w-12 h-12')
                    with ui.row():
                        ui.label().classes('w-12')
                        ui.button('⬇️', on_click=r.walk_backward).classes('w-12 h-12')
                        ui.label().classes('w-12')

                ui.label('Head').classes('font-bold mt-4')
                with ui.column().classes('items-center'):
                    with ui.row():
                        ui.label().classes('w-12')
                        ui.button('⬆️', on_click=lambda: r.head.move(tilt=min(TILT_MAX, r.head.tilt + 50))).classes('w-12 h-12')
                        ui.label().classes('w-12')
                    with ui.row():
                        ui.button('👈', on_click=lambda: r.head.move(pan=min(PAN_MAX, r.head.pan + 75))).classes('w-12 h-12')
                        ui.button('⏺️', on_click=lambda: r.head.center(duration=0.5)).classes('w-12 h-12')
                        ui.button('👉', on_click=lambda: r.head.move(pan=max(PAN_MIN, r.head.pan - 75))).classes('w-12 h-12')
                    with ui.row():
                        ui.label().classes('w-12')
                        ui.button('⬇️', on_click=lambda: r.head.move(tilt=max(TILT_MIN, r.head.tilt - 50))).classes('w-12 h-12')
                        ui.label().classes('w-12')

        # Actions
        with ui.row().classes('w-full max-w-4xl mt-4'):
            ui.label('Actions').classes('font-bold')
        with ui.row().classes('w-full max-w-4xl gap-2 flex-wrap'):
            ui.button('🧍 Stand', on_click=r.stand)
            ui.button('🧎 Crouch', on_click=r.stand_low)
            ui.button('👋 Greet', on_click=r.greet)
            ui.button('🙌 Wave', on_click=r.wave)
            ui.button('⬅️ Step L', on_click=r.step_left)
            ui.button('➡️ Step R', on_click=r.step_right)
            ui.button('🔄 Twist', on_click=lambda: r.play('twist'))
            ui.button('⚽ Left Shot', on_click=lambda: r.play('left_shot'))
            ui.button('⚽ Right Shot', on_click=lambda: r.play('right_shot'))


ui.run(title='AINEX Control', port=8080, reload=False)
