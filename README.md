# AINEX MK2

Pure Python API for the AINEX humanoid robot. No ROS.

Direct serial control of the Rosman5 board - works on any Pi with any Linux.

## Install

```bash
# 1. Install udev rules (serial + camera permissions)
sudo cp tools/99-ainex.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger

# 2. Install Python dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# Run demo (face tracking + gesture mirroring)
python run.py

# With remote vision (faster - run server on PC first)
python -m ainex_api.remote_vision --port 9999  # On PC
python run.py --remote-vision <PC_IP>:9999     # On robot

# Web control panel
python tools/ainexCP.py  # Opens http://localhost:8080
```

## API

```python
from ainex_api import Robot

robot = Robot()

# Poses
robot.stand()           # Standing pose
robot.stand_low()       # Crouched
robot.zero()            # All servos center
robot.relax()           # Disable torque
robot.enable()          # Enable torque

# Motions (48 .d6a files)
robot.greet()
robot.wave()
robot.walk_forward()
robot.walk_backward()
robot.turn_left()
robot.turn_right()
robot.step_left()
robot.step_right()
robot.climb_stairs()
robot.descend_stairs()
robot.get_up_from_front()
robot.get_up_from_back()
robot.play('motion_name')   # Any .d6a file
robot.list_motions()        # List all

# Arms (direct servo, fast)
robot.raise_left_arm()
robot.raise_right_arm()
robot.raise_both_arms()
robot.lower_arms()
robot.lower_left_arm()
robot.lower_right_arm()

# Head
robot.head.move(pan=500, tilt=500)  # Absolute position
robot.head.center()                 # Center position
robot.head.track_point(x, y)       # PID tracking
robot.head.nod()
robot.head.shake()
robot.head.pan                      # Current pan position
robot.head.tilt                     # Current tilt position

# Vision
robot.vision.start()
robot.vision.update()               # Capture + detect
face = robot.vision.get_face()      # FaceData or None
gesture = robot.vision.get_gesture() # GestureData
frame = robot.vision.get_frame()    # BGR numpy array
debug = robot.vision.draw_debug(face=True, gesture=True, pose=True)
robot.vision.landmarks              # Raw pose landmarks
robot.vision.has_mediapipe          # MediaPipe available?
robot.vision.stop()

# Servos (direct control)
robot.servos.set_position(13, 500, duration=0.5)
robot.servos.set_positions({13: 360, 14: 640}, duration=0.5)
robot.servos.set_by_name('l_shoulder_pitch', 500)
robot.servos.set_body([500] * 22, duration=1.0)
robot.servos.get_position(13)
robot.servos.get_positions([13, 14])
robot.servos.enable_torque(13, True)
robot.servos.enable_all_torque(True)
robot.servos.stop()
robot.servos.read_temperature(13)   # Celsius
robot.servos.read_voltage(13)       # mV
robot.servos.set_offset(13, 0)
robot.servos.save_offset(13)
robot.servos.get_servo_name(13)     # 'l_shoulder_pitch'
robot.servos.get_servo_id('l_shoulder_pitch')  # 13

# Sensors
robot.get_battery()                 # Percentage, or None
robot.get_battery_status()          # 'ok' | 'low' | 'critical', or None
robot.sensors.get_battery_state()   # BatteryState(voltage_mv, percent, status) from one reading
robot.sensors.get_imu()             # IMUData or None
robot.sensors.get_battery_voltage() # mV or None
robot.sensors.get_battery_voltage_smoothed()  # mV, median of last 5; None until 5 arrive
robot.sensors.get_button()          # ButtonEvent or None
robot.sensors.on_button(callback)   # Register callback
robot.start_sensors()               # Only needed to run on_button callbacks
robot.stop_sensors()
robot.sensors.get_gamepad()         # (axes, buttons) or None

# Peripherals
robot.beep(freq=2000, duration=0.1)
robot.chirp()
robot.peripherals.beep_pattern(2000, 0.1, 0.1, repeat=3)
robot.peripherals.confirm()         # Two beeps
robot.peripherals.error_sound()     # Low tone
robot.peripherals.startup_sound()   # Rising melody
robot.peripherals.led_on()
robot.peripherals.led_off()
robot.peripherals.led_blink(on_time=0.5, off_time=0.5, repeat=3)

# Status
robot.status()                      # Dict of all state
robot.print_status()                # Print formatted

robot.close()
```

### Gestures Detected

- `none` - No gesture
- `left_hand_raised` - User's left hand up
- `right_hand_raised` - User's right hand up
- `both_hands_raised` - Both hands up
- `waving` - Hand in wave position with oscillating motion

### Remote Vision

Offload MediaPipe to PC for better performance (~66ms vs ~150ms).

```bash
# PC: Start server
python -m ainex_api.remote_vision --port 9999

# Robot: Connect
robot = Robot(remote_vision="192.168.0.3:9999")
```

### Camera (standalone)

```python
from ainex_api import Camera, open_camera

# Background capture with optional lens correction
camera = Camera(undistort=True, distortion_k1=-0.15)
camera.start()
frame = camera.read()       # BGR numpy array or None
camera.fps                  # Current FPS
camera.is_open              # Running?
camera.stop()

# One-shot open (no background thread)
cap, width, height = open_camera()
```

### Demo Mirror

`demo_mirror/` is a standalone demo that tracks faces with the head and mirrors arm gestures. Wave at the robot and it greets back. Run it with `python run.py` (see Quick Start).

## Hardware

| Resource | Path | Details |
|----------|------|---------|
| Serial | `/dev/ttyAMA0` | 1M baud, Pi GPIO UART to STM32 |
| Camera | `/dev/usb_cam` | 640x480 (symlink from udev) |
| Servos | Bus servos | Position 0-1000, center=500 |
| Head | Servos 23/24 | Pan 125-875, tilt 315-625; enforced on every write path |
| Battery | 3S LiPo | 11.1V 3500mAh, 12.6V full, recharge at 10.5V, stop at 9.9V |

Sensor reads older than 15 s report `None` rather than a stale value. Servo writes outside a
servo's travel raise `ValueError`; percentage is an estimate from a resting discharge curve and
reads low under load.

## Servo Map

| ID | Joint | ID | Joint |
|----|-------|----|-------|
| 1 | l_ankle_roll | 2 | r_ankle_roll |
| 3 | l_ankle_pitch | 4 | r_ankle_pitch |
| 5 | l_knee | 6 | r_knee |
| 7 | l_hip_pitch | 8 | r_hip_pitch |
| 9 | l_hip_roll | 10 | r_hip_roll |
| 11 | l_hip_yaw | 12 | r_hip_yaw |
| 13 | l_shoulder_pitch | 14 | r_shoulder_pitch |
| 15 | l_shoulder_roll | 16 | r_shoulder_roll |
| 17 | l_elbow_pitch | 18 | r_elbow_pitch |
| 19 | l_elbow_yaw | 20 | r_elbow_yaw |
| 21 | l_gripper | 22 | r_gripper |
| 23 | head_pan | 24 | head_tilt |

**Pattern:** Odd = Left, Even = Right

## Structure

```
ainex-api/
├── ainex_api/           # Hardware API
│   ├── robot.py         # Main entry point
│   ├── vision.py        # Face/gesture detection
│   ├── remote_vision.py # PC offload server/client
│   ├── head.py          # Pan/tilt + PID tracking
│   ├── motion.py        # D6A motion player
│   ├── servos.py        # 24 servo control
│   ├── sensors.py       # IMU, battery, buttons
│   ├── peripherals.py   # Buzzer, LED
│   ├── board.py         # Serial protocol
│   ├── camera.py        # USB camera
│   └── motions/         # 48 .d6a files
├── demo_mirror/         # Demo: face tracking + gesture mirroring
├── tools/
│   ├── ainexCP.py       # Web control panel
│   ├── 99-ainex.rules   # udev rules
│   └── test_remote_vision.py
├── run.py               # Entry point
└── requirements.txt
```
