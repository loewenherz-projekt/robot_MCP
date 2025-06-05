"""
Configuration for the robot controller.
Update it before using any other script.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Tuple
from lerobot.common.robot_devices.cameras.configs import OpenCVCameraConfig, CameraConfig


@dataclass
class RobotConfig:
    """Configuration for the robot controller."""

    # Serial port
    port: str = "/dev/tty.usbmodem58FD0168731"

    # Provide the absolute path to the calibration file
    calibration_file: str = os.path.join(os.path.dirname(__file__), "main_follower.json")
    
    motors = {
            "shoulder_pan": [1, "sts3215"],
            "shoulder_lift": [2, "sts3215"],
            "elbow_flex": [3, "sts3215"],
            "wrist_flex": [4, "sts3215"],
            "wrist_roll": [5, "sts3215"],
            "gripper": [6, "sts3215"]
        }
    
    # Camera configuration using lerobot format
    cameras: Dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "front": OpenCVCameraConfig(
                camera_index=1,
                fps=30,
                width=640,
                height=360,
            ),
             "wrist": OpenCVCameraConfig(
                camera_index=0,
                fps=30,
                width=640,
                height=360,
            ),          
            "top view": OpenCVCameraConfig(
                camera_index=2,
                fps=30,
                width=640,
                height=360,
            ),    
        }
    )

    robot_description: str = ("""
Follow these instructions precisely. Never deviate.

You control a 3D printed robot with 5 DOF + gripper. Max forward reach ~250 mm.
Shoulder and elbow links are 12 cm and 14 cm. Gripper fingers ~8 cm.
Use these to estimate distances. E.g., if the object is near but not in the gripper, you can safely move 5–10 cm forward.

Robot has 3 cameras:
- front: at the base, looks forward
- wrist: close view of gripper
- top view: shows whole robot

Instructions:
- Move slowly and iteratively
- Close gripper completely to grab objects
- Check results after each move before proceeding
- Split into smaller steps and reanalyze after each one
- Use only the latest images to evaluate success
- Always plan movements to avoid collisions
- Move above object with gripper tilted up (10–15°) to avoid collisions. Stay >25 cm above ground when moving or rotating
- Never move with gripper near the ground
- Drop and restart plan if unsure or failed
"""
    )

    # Kinematic and operational parameters
    L1: float = 117.0  # Length of first lever (shoulder to elbow) in mm
    L2: float = 136.0  # Length of second lever (elbow to wrist_flex axis) in mm
    BASE_HEIGHT_MM: float = 120.0 # Height from ground to shoulder_lift axis in mm
    
    # These offsets describe how the physical linkage is mounted relative to ideal joint axes
    # Used in FK/IK calculations if the kinematic model requires them.
    SHOULDER_MOUNT_OFFSET_MM: float = 32.0 # Example: Offset for shoulder joint from idealized zero
    ELBOW_MOUNT_OFFSET_MM: float = 4.0    # Example: Offset for elbow joint from idealized zero

    OPERATIONAL_DEGREE_LIMITS: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "gripper":         (-10.0, 110.0),      # 0 = closed, 100 = fully open
            "wrist_roll":      (-20.0, 120.0),       # 0 = vertical, 90 = flat
            "wrist_flex":     (-100.0, 100.0),    # -90 = max up, 90 = max down
            "elbow_flex":      (-15.0, 190.0),     # 0 = fully bent, 180 = fully straight
            "shoulder_lift":  (-15.0, 195.0),     # 0 = base position, 180 = fully forward
            "shoulder_pan":    (0.0, 180.0),      # 0 left, 180 right
        }
    )

    PRESET_POSITIONS: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "1": { "gripper": 0.0, "wrist_roll": 90.0, "wrist_flex": 0.0, "elbow_flex": 0.0, "shoulder_lift": 0.0, "shoulder_pan": 90.0 },
            "2": { "gripper": 0.0, "wrist_roll": 90.0, "wrist_flex": 0.0, "elbow_flex": 45.0, "shoulder_lift": 45.0, "shoulder_pan": 90.0 },
            "3": { "gripper": 40.0, "wrist_roll": 90.0, "wrist_flex": 90.0, "elbow_flex": 45.0, "shoulder_lift": 45.0, "shoulder_pan": 90.0 },
            "4": { "gripper": 40.0, "wrist_roll": 90.0, "wrist_flex": -60.0, "elbow_flex": 20.0, "shoulder_lift": 80.0, "shoulder_pan": 90.0 },
        }
    )

    SPATIAL_LIMITS: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "x": (-20.0, 250.0), # Min/Max X coordinate (mm) for wrist_flex origin
            "z": (30.0, 370.0),  # Min/Max Z coordinate (mm) for wrist_flex origin
        }
    )

# Create a global instance
robot_config = RobotConfig()