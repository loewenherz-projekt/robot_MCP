#!/usr/bin/env python3
"""Gamepad controller with full button, axis, and D-pad/hat logging."""

import sys
import time
import logging

from lerobot.teleoperators.gamepad.gamepad_utils import GamepadController
from robot_controller import RobotController

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Xbox Standard Mapping (anpassen nach Testausgabe!)
BUTTON_NAMES = {
    0: "A",
    1: "B",
    2: "",
    3: "X",
    4: "Y",
    5: "RB",
    6: "LB",
    7: "LB",
    8: "Xbox",
    9: "",
    10: "Back",
    11: "Start",
    12: "XBox",
    13: "Left Stick Button",
    14: "Right Stick Button",
    15: "Take Pic Btn"
    # ggf. erweitern!
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

        # Zustand der Buttons/Achsen/Hats für Event-Tracking
        self.last_button_state = []
        self.last_axis_state = []
        self.last_hat_state = []

    def start(self) -> None:
        self.gamepad.start()
        self.running = True
        # Initialisiere Status-Arrays anhand echter Anzahl Buttons/Achsen/Hats
        joy = self.gamepad.joystick
        self.last_button_state = [0] * joy.get_numbuttons()
        self.last_axis_state = [0.0] * joy.get_numaxes()
        self.last_hat_state = [(0, 0)] * joy.get_numhats()

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

    def _print_input_events(self):
        joy = self.gamepad.joystick

        # BUTTONS
        for i in range(joy.get_numbuttons()):
            pressed = joy.get_button(i)
            if pressed != self.last_button_state[i]:
                state = "pressed" if pressed else "released"
                print(f"Button {i} ({BUTTON_NAMES.get(i, f'Button {i}')}) {state}")
                self.last_button_state[i] = pressed

        # AXES (Sticks + Trigger)
        for i in range(joy.get_numaxes()):
            value = joy.get_axis(i)
            # Nur relevante Änderungen anzeigen (Toleranzschwelle)
            if abs(value - self.last_axis_state[i]) > 0.07:
                print(f"Axis {i} changed: {value:.2f}")
                self.last_axis_state[i] = value

        # HATS (D-Pad)
        for i in range(joy.get_numhats()):
            hat = joy.get_hat(i)
            if hat != self.last_hat_state[i]:
                print(f"D-Pad Hat {i} changed: {hat}")
                self.last_hat_state[i] = hat

    def _handle_inputs(self) -> None:
        self.gamepad.update()
        self._print_input_events()  # Zeige alle Änderungen an!

        # Restlicher Steuerungs-Code wie gehabt:
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

        rotate_delta = rx * self.angle_step_deg
        forward_delta = -ry * self.spatial_step_mm
        if abs(rotate_delta) > 1e-3 or abs(forward_delta) > 1e-3:
            result = self.robot.execute_intuitive_move(
                move_gripper_forward_mm=forward_delta,
                rotate_robot_clockwise_angle=rotate_delta,
                use_interpolation=False,
            )
            if not result.ok:
                logger.warning(f"Arm move failed: {result.msg}")

        # Button-Aktionen wie gehabt:
        if self.gamepad.joystick.get_button(3):  # X
            self.robot.execute_intuitive_move(
                rotate_gripper_clockwise_angle=-self.angle_step_deg,
                use_interpolation=False,
            )
        if self.gamepad.joystick.get_button(4):  # Y
            self.robot.execute_intuitive_move(
                rotate_gripper_clockwise_angle=self.angle_step_deg,
                use_interpolation=False,
            )
        if self.gamepad.joystick.get_button(0):  # A
            self.robot.increment_joints_by_delta({"gripper": -self.gripper_step_pct})
        if self.gamepad.joystick.get_button(1):  # B
            self.robot.increment_joints_by_delta({"gripper": self.gripper_step_pct})
        if self.gamepad.joystick.get_button(10):  # Back
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
