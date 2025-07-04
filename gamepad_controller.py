#!/usr/bin/env python3
"""Gamepad controller based on keyboard_controller.py."""

import sys
import time
import logging

from lerobot.teleoperators.gamepad.gamepad_utils import GamepadController
from robot_controller import RobotController

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

class XboxGamepadController:
    """Control robot using an Xbox Series X Bluetooth gamepad."""

    def __init__(self, robot_controller: RobotController):
        self.robot = robot_controller
        self.gamepad = GamepadController()
        self.running = False
        self.base_speed_scale = 0.3    # m/s for linear x/y
        self.base_rot_speed_deg = 60.0 # deg/s for base rotation
        self.spatial_step_mm = 2.0
        self.angle_step_deg = 2.0
        self.gripper_step_pct = 3.0

    def start(self) -> None:
        self.gamepad.start()
        self.running = True
        print("\n" + "="*50)
        print("🎮 GAMEPAD CONTROLLER ACTIVE")
        print("Left stick:    Base forward/back & rotate (Y/X)")
        print("Triggers:      Base strafe left/right (LT/RT)")
        print("D-Pad:         Arm up/down")
        print("Right stick:   Arm forward/back & rotate")
        print("X/Y buttons:   Rotate gripper left/right")
        print("A/B buttons:   Close/Open gripper")
        print("BACK button:   Exit")
        print("="*50)

    def stop(self) -> None:
        if self.running:
            self.running = False
            self.gamepad.stop()
            print("\n🛑 Gamepad controller stopped")

    def _handle_inputs(self) -> None:
        self.gamepad.update()
        lx, ly, rx, ry, lt, rt = self.gamepad.get_axis_values()

        # Normalize triggers from [-1, 1] -> [0, 1]
        lt = (lt + 1.0) / 2.0
        rt = (rt + 1.0) / 2.0

        # D-Pad (Hat) für Arm hoch/runter
        hat_x, hat_y = 0, 0
        if hasattr(self.gamepad.joystick, "get_hat"):
            try:
                hat_x, hat_y = self.gamepad.joystick.get_hat(0)
            except Exception:
                hat_x, hat_y = 0, 0

        # --- Arm-Steuerung ---
        rotate_delta   = rx * self.angle_step_deg
        forward_delta  = -ry * self.spatial_step_mm
        up_delta       = hat_y * self.spatial_step_mm  # D-Pad up/down

        if abs(rotate_delta) > 1e-3 or abs(forward_delta) > 1e-3 or hat_y != 0:
            result = self.robot.execute_intuitive_move(
                move_gripper_forward_mm=forward_delta,
                move_gripper_up_mm=up_delta,
                rotate_robot_clockwise_angle=rotate_delta,
                use_interpolation=False,
            )
            if not result.ok:
                logger.warning(f"Arm move failed: {result.msg}")

        # --- Base-Steuerung ---
        base_action = {
            "x.vel": -ly * self.base_speed_scale,
            "y.vel": (lt - rt) * self.base_speed_scale,
            "theta.vel": -lx * self.base_rot_speed_deg,
        }
        try:
            self.robot.robot.send_action(base_action)
        except Exception as e:
            logger.error(f"Base move failed: {e}")

        # --- Gripper/Rotation über Buttons ---
        # X button -> rotate gripper left
        if self.gamepad.joystick.get_button(2):
            self.robot.execute_intuitive_move(
                rotate_gripper_clockwise_angle=-self.angle_step_deg,
                use_interpolation=False,
            )
        # Y button -> rotate gripper right
        if self.gamepad.joystick.get_button(3):
            self.robot.execute_intuitive_move(
                rotate_gripper_clockwise_angle=self.angle_step_deg,
                use_interpolation=False,
            )
        # A button -> close gripper
        if self.gamepad.joystick.get_button(0):
            self.robot.increment_joints_by_delta({"gripper": -self.gripper_step_pct})
        # B button -> open gripper
        if self.gamepad.joystick.get_button(1):
            self.robot.increment_joints_by_delta({"gripper": self.gripper_step_pct})

        # BACK button (6) to exit
        if self.gamepad.joystick.get_button(6):
            self.stop()

    def run(self) -> None:
        self.start()
        try:
            while self.running:
                self._handle_inputs()
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("\n⚠️  KeyboardInterrupt received, shutting down...")
        finally:
            self.stop()


def main() -> int:
    print("🔌 Connecting to robot...")
    robot = RobotController()
    gp_controller = XboxGamepadController(robot)
    gp_controller.run()
    robot.disconnect(reset_pos=True)
    print("👋 Gamepad controller finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
