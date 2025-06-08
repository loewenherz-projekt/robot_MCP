"""
Keyboard controller for the robot.
Used to control the robot manually.
"""

import logging
import time
import os
from datetime import datetime
from typing import Dict, Any
from pynput import keyboard
from robot_controller import RobotController
from PIL import Image

logger = logging.getLogger(__name__)

class KeyboardController:
    def __init__(self, robot_controller: RobotController):
        """Initialize keyboard controller with robot controller."""
        self.robot = robot_controller
        self.running = False
        self.spatial_step_mm = 3.0
        self.angle_step_deg = 3.0
        self.gripper_step_pct = 5.0
        
        # Create snapshots directory if it doesn't exist
        self.snapshots_dir = "camera_snapshots"
        os.makedirs(self.snapshots_dir, exist_ok=True)
        
        # Key mappings using intuitive command names
        self.key_mappings = {
            # Cartesian movements
            keyboard.KeyCode.from_char('w'): ("intuitive_move", {"move_gripper_forward_mm": self.spatial_step_mm}),
            keyboard.KeyCode.from_char('s'): ("intuitive_move", {"move_gripper_forward_mm": -self.spatial_step_mm}),
            keyboard.Key.up:                 ("intuitive_move", {"move_gripper_up_mm": self.spatial_step_mm}),
            keyboard.Key.down:               ("intuitive_move", {"move_gripper_up_mm": -self.spatial_step_mm}),
            
            # Rotations and Tilts (as direct joint deltas within intuitive_move)
            keyboard.Key.left:  ("intuitive_move", {"rotate_robot_clockwise_angle": -self.angle_step_deg}), # Counter-clockwise
            keyboard.Key.right: ("intuitive_move", {"rotate_robot_clockwise_angle": self.angle_step_deg}),  # Clockwise
            
            keyboard.KeyCode.from_char('r'): ("intuitive_move", {"tilt_gripper_down_angle": -self.angle_step_deg}), # Tilt Up
            keyboard.KeyCode.from_char('f'): ("intuitive_move", {"tilt_gripper_down_angle": self.angle_step_deg}),  # Tilt Down
            
            keyboard.KeyCode.from_char('a'): ("intuitive_move", {"rotate_gripper_clockwise_angle": -self.angle_step_deg}), # Counter-clockwise
            keyboard.KeyCode.from_char('d'): ("intuitive_move", {"rotate_gripper_clockwise_angle": self.angle_step_deg}),  # Clockwise
            
            # Gripper control (direct joint set, not part of intuitive_move structure)
            keyboard.KeyCode.from_char('q'): ("gripper_delta", self.gripper_step_pct),  # Open incrementally
            keyboard.KeyCode.from_char('e'): ("gripper_delta", -self.gripper_step_pct), # Close incrementally

            # Camera snapshot key
            keyboard.KeyCode.from_char('c'): ("camera_snapshot", None),
            
            # Preset positions using number keys
            keyboard.KeyCode.from_char('1'): ("preset", "1"),
            keyboard.KeyCode.from_char('2'): ("preset", "2"),
            keyboard.KeyCode.from_char('3'): ("preset", "3"),
            keyboard.KeyCode.from_char('4'): ("preset", "4"),
            # Add more presets if defined in robot_config

        }

    def on_press(self, key: Any) -> bool:
        if key == keyboard.Key.esc:
            self.stop()
            return False # Stop listener

        if key in self.key_mappings:
            action_type, params = self.key_mappings[key]
            
            if action_type == "intuitive_move":
                self.robot.execute_intuitive_move(**params, use_interpolation_for_move=False).to_json() 
            elif action_type == "gripper_delta":
                delta = params
                self.robot.increment_joints_by_delta({'gripper': delta}).to_json()
            elif action_type == "preset":
                preset_key = params
                self.robot.apply_named_preset(preset_key).to_json()
            elif action_type == "camera_snapshot":
                self.take_camera_snapshot()
        return True

    def take_camera_snapshot(self) -> None:
        try:
            images = self.robot.get_camera_images()
            if not images: return
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            saved_count = 0
            for camera_name, img_array in images.items():
                try:
                    pil_img = Image.fromarray(img_array)
                    filename = os.path.join(self.snapshots_dir, f"{camera_name}_{timestamp}.jpg")
                    pil_img.save(filename)
                    saved_count += 1
                except Exception as e_save: logger.error(f"Snapshot save error for '{camera_name}': {e_save}", exc_info=True)
        except Exception as e_snap: logger.error(f"Take snapshot error: {e_snap}", exc_info=True)

    def start(self) -> None:
        # Print controls once at startup
        print("--- Keyboard Controller Active ---")
        print(f" W/S: Gripper Forward/Backward ({self.spatial_step_mm} mm)")
        print(f" UP/DOWN: Gripper Up/Down ({self.spatial_step_mm} mm)")
        print(f" LEFT/RIGHT: Rotate Robot CCW/CW ({self.angle_step_deg} deg)")
        print(f" R/F: Tilt Gripper Up/Down ({self.angle_step_deg} deg)")
        print(f" A/D: Rotate Gripper CCW/CW ({self.angle_step_deg} deg)")
        print(f" Q/E: Gripper Open/Close by {self.gripper_step_pct}%")
        print(" C: Camera Snapshot")
        print(" 1-4: Presets")

        print(" ESC: Exit")
        print("--------------------------------")
        self.running = True
        try:
            self.listener = keyboard.Listener(on_press=self.on_press)
            self.listener.start()
        except Exception as e:
            # Critical error if listener cannot start
            logger.error(f"Keyboard listener FAILED to start: {e}", exc_info=True)
            self.running = False # Ensure controller knows it's not running

    def stop(self) -> None:
        if self.running:
            # print("Stopping Keyboard Controller...") # User feedback, not logging
            self.running = False
            if hasattr(self, 'listener') and self.listener.is_alive():
                try: self.listener.stop()
                except Exception as e_stop: logger.error(f"Listener stop error: {e_stop}", exc_info=True)

# Add this block at the end of the file if it's not there or is incomplete
if __name__ == '__main__':
    # Setup minimal logging for standalone test/run
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s KBD_Ctrl %(levelname)s: %(message)s")
    logger_main = logging.getLogger("KeyboardCtrlApp") 

    logger_main.info("Attempting to start KeyboardController directly...")
    
    robot_instance = None
    kb_controller = None
    try:                    
        robot_instance = RobotController()
        logger_main.info("RobotController initialized for KeyboardController.")
        
        kb_controller = KeyboardController(robot_instance)
        kb_controller.start()
        
        # Keep main thread alive while keyboard listener runs
        # Sleep prevents excessive CPU usage while still allowing clean shutdown
        while kb_controller.running:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        logger_main.info("KeyboardInterrupt received by main thread, initiating shutdown.")
    except Exception as e_main:
        logger_main.error(f"CRITICAL error in KeyboardController main execution: {e_main}", exc_info=True)
    finally:
        if kb_controller and kb_controller.running:
            logger_main.info("Ensuring KeyboardController is stopped...")
            kb_controller.stop()
        if robot_instance:
            logger_main.info("Disconnecting robot from KeyboardController app...")
            robot_instance.disconnect()
        logger_main.info("KeyboardController application finished.")
