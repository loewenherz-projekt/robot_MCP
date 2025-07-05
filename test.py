#!/usr/bin/env python3
"""Gamepad controller based on keyboard_controller.py."""

import sys
import time
import logging

from lerobot.teleoperators.gamepad.gamepad_utils import GamepadController
from robot_controller import RobotController

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Mapping von Button-Nummern zu Namen (ggf. anpassen falls dein Mapping anders ist)
BUTTON_NAMES = {
    0: "A",
    1: "B",
    2: "X",
    3: "Y",
    4: "LB",
    5: "RB",
    6: "BACK",
    7: "START",
    8: "Guide",
    9: "Left Stick",
    10: "Right Stick"
    # ggf. erweitern je nach Controller
}

class XboxGamepadController:
    """Control robot using an Xbox Series X Bluetooth gamepad."""

    def __init__(self, robot_controller: RobotController):
        self.robot = robot_controller
        self.gamepad = GamepadController()
        self.running = False
        self.base_speed_scale = 0.3
        self.spatial_step_mm = 2.0
        self.angle_step_deg = 2.0
        self.gripper_step_pct = 3.0
        # Neu: Merke dir alten Button-Status
        self.last_button_state = [0] * 12  # Passe die Länge ggf. an deine Anzahl Buttons an

    def start(self) -> None:
        self.gamepad.start()
        self.running = True
        print("\n" + "="*50)
        print("🎮 GAMEPAD CONTROLLER ACTIVE")
        print("Left stick:    Base X/Y")
        print("Right stick:   Arm rotation/forward")
        print("X/Y buttons:   Rotate gripper left/right")
        print("A/B buttons:   Close/Open gripper")
        print("BACK button:   Exit")
        print("="*50)

    def stop(self) -> None:
        if self.running:
            self.running = False
            self.gamepad.stop()
            print("\n🛑 Gamepad controller stopped")

    def _print_button_events(self):
        # Prüfe alle Buttons und gib Statusänderung aus
        for i in range(len(self.last_button_state)):
            pressed = self.gamepad.joystick.get_button(i)
            if pressed != self.last_button_state[i]:
                if pressed:
                    print(f"Controller Button pressed: {BUTTON_NAMES.get(i, f'Button {i}')}")
                else:
                    print(f"Controller Button released: {BUTTON_NAMES.get(i, f'Button {i}')}")
                self.last_button_state[i] = pressed

    def _handle_inputs(self) -> None:
        self.gamepad.update()
        self._print_button_events()  # NEU: Button-Änderungen immer anzeigen

        lx, ly, rx, ry, _, _ = self.gamepad.get_axis_values()

        base_action = {
            "x.vel": -ly * self.base_speed_scale,
            "y.vel": -lx * self.base_speed_scale,
            "theta.vel": 0.0,
        }
        try:
            self.robot.robot.send_action(base_action)
        except Exception as e:
            logger.error(f"Base move failed: {e}")

        arm_move = False
        rotate_delta = rx * self.angle_step_deg
        forward_delta = -ry * self.spatial_step_mm
        if abs(rotate_delta) > 1e-3 or abs(forward_delta) > 1e-3:
            arm_move = True
            result = self.robot.execute_intuitive_move(
                move_gripper_forward_mm=forward_delta,
                rotate_robot_clockwise_angle=rotate_delta,
                use_interpolation=False,
            )
            if not result.ok:
                logger.warning(f"Arm move failed: {result.msg}")

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
