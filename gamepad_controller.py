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
        self.base_speed_scale = 0.25  # m/s for linear x/y (reduced to prevent servo saturation)
        self.base_rot_speed_deg = 45.0  # deg/s for base rotation (reduced to prevent servo saturation)
        self.spatial_step_mm = 1.5  # mm per step (reduced to prevent servo saturation)
        self.vertical_step_mm = 0.5  # mm per step for D-Pad vertical movement (smaller for wrist flex sensitivity)
        self.angle_step_deg = 1.5  # deg per step (reduced to prevent servo saturation)
        self.gripper_step_pct = 2.5  # % per step (reduced to prevent servo saturation)
        
        # Gripper rotation speed control
        self.gripper_rotation_slow_factor = 0.25  # 25% of normal speed
        self.gripper_rotation_speed_up_time = 1.0  # seconds to reach full speed
        self.x_button_press_time = None
        self.y_button_press_time = None
        
    def _apply_exponential_curve(self, value: float, deadzone: float = 0.05, exponent: float = 2.0) -> float:
        """Apply exponential curve to joystick input for more precise control.
        
        Args:
            value: Raw joystick input (-1.0 to 1.0)
            deadzone: Deadzone threshold (0.0 to 1.0)
            exponent: Exponential power (higher = slower at low deflection)
            
        Returns:
            Curved value for more precise control
        """
        # Apply deadzone
        if abs(value) < deadzone:
            return 0.0
        
        # Remove deadzone from calculation
        sign = 1.0 if value > 0 else -1.0
        abs_value = abs(value)
        
        # Scale to remove deadzone effect
        scaled_value = (abs_value - deadzone) / (1.0 - deadzone)
        
        # Apply exponential curve: x^exponent
        curved_value = pow(scaled_value, exponent)
        
        return sign * curved_value

    def start(self) -> None:
        self.gamepad.start()
        self.running = True
        print("\n" + "="*50)
        print("🎮 GAMEPAD CONTROLLER ACTIVE (Exponential Control)")
        print("Left stick:    Base forward/back & rotate")
        print("Triggers:      Gripper tilt (LT=up, RT=down)")
        print("D-Pad:         Arm up/down (fine control)")
        print("Right stick:   Arm forward/back & rotate")
        print("X/Y buttons:   Rotate gripper right/left")
        print("A/B buttons:   Close/Open gripper")
        print("BACK button:   Exit")
        print("Note: Exponential curve - very slow at low deflection, full speed at maximum")
        print("Gripper rotation: 25% speed initially, 100% speed after 1s hold")
        print("="*50)

    def stop(self) -> None:
        if self.running:
            self.running = False
            self.gamepad.stop()
            print("\n🛑 Gamepad controller stopped")

    def _handle_inputs(self) -> None:
        self.gamepad.update()
        lx_raw, ly_raw, rx_raw, ry_raw, lt_raw, rt_raw = self.gamepad.get_axis_values()
        
        # Apply exponential curve to all axes for more precise control
        # Higher exponent: much slower at low deflection, full speed at maximum
        lx = self._apply_exponential_curve(lx_raw, deadzone=0.05, exponent=3.0)
        ly = self._apply_exponential_curve(ly_raw, deadzone=0.05, exponent=3.0)
        rx = self._apply_exponential_curve(rx_raw, deadzone=0.05, exponent=3.0)
        ry = self._apply_exponential_curve(ry_raw, deadzone=0.05, exponent=3.0)
        
        # Triggers use different processing (they go from -1 to +1)
        lt = lt_raw
        rt = rt_raw
        
        # Debug output showing stick deflection vs speed relationship
        if abs(lx) > 0.01 or abs(ly) > 0.01 or abs(rx) > 0.01 or abs(ry) > 0.01:
            # Show raw vs curved values with visual bars
            def make_bar(value, max_width=20):
                width = int(abs(value) * max_width)
                return ('█' * width).ljust(max_width)
            
            print(f"\\nStick Deflection vs Speed:")
            if abs(lx_raw) > 0.01:
                print(f"LX: Raw={lx_raw:+.3f} |{make_bar(lx_raw)}| → Curved={lx:+.3f} |{make_bar(lx)}| ({abs(lx)/max(abs(lx_raw), 0.001)*100:.1f}%)")
            if abs(ly_raw) > 0.01:
                print(f"LY: Raw={ly_raw:+.3f} |{make_bar(ly_raw)}| → Curved={ly:+.3f} |{make_bar(ly)}| ({abs(ly)/max(abs(ly_raw), 0.001)*100:.1f}%)")
            if abs(rx_raw) > 0.01:
                print(f"RX: Raw={rx_raw:+.3f} |{make_bar(rx_raw)}| → Curved={rx:+.3f} |{make_bar(rx)}| ({abs(rx)/max(abs(rx_raw), 0.001)*100:.1f}%)")
            if abs(ry_raw) > 0.01:
                print(f"RY: Raw={ry_raw:+.3f} |{make_bar(ry_raw)}| → Curved={ry:+.3f} |{make_bar(ry)}| ({abs(ry)/max(abs(ry_raw), 0.001)*100:.1f}%)")
        
        # Store trigger debug for later (will be shown after trigger processing)

        hat_x, hat_y = 0, 0
        if hasattr(self.gamepad.joystick, "get_hat"):
            try:
                hat_x, hat_y = self.gamepad.joystick.get_hat(0)
            except Exception:
                hat_x, hat_y = 0, 0

        # Arm control first so base command isn't overwritten
        rotate_delta = rx * self.angle_step_deg
        forward_delta = -ry * self.spatial_step_mm
        up_delta = hat_y * self.vertical_step_mm

        # Lower threshold for better stick sensitivity
        if abs(rotate_delta) > 0.01 or abs(forward_delta) > 0.01 or hat_y != 0:
            result = self.robot.execute_intuitive_move(
                move_gripper_forward_mm=forward_delta,
                move_gripper_up_mm=up_delta,
                rotate_robot_right_angle=rotate_delta,
                use_interpolation=False,
            )
            if not result.ok:
                logger.warning(f"Arm move failed: {result.msg}")

        # Fix trigger values: Xbox triggers go from -1.0 (rest) to +1.0 (pressed)
        # Convert to 0.0 (rest) to 1.0 (pressed)
        lt_normalized_raw = (lt + 1.0) / 2.0
        rt_normalized_raw = (rt + 1.0) / 2.0
        
        # Apply exponential curve to triggers for more precise control
        lt_normalized = self._apply_exponential_curve(lt_normalized_raw, deadzone=0.03, exponent=3.0)
        rt_normalized = self._apply_exponential_curve(rt_normalized_raw, deadzone=0.03, exponent=3.0)
        
        # Show trigger values with visualization
        if abs(lt_normalized_raw) > 0.01 or abs(rt_normalized_raw) > 0.01:
            def make_bar(value, max_width=20):
                width = int(abs(value) * max_width)
                return ('█' * width).ljust(max_width)
            
            print(f"\\nTrigger Deflection vs Speed:")
            if abs(lt_normalized_raw) > 0.01:
                print(f"LT: Raw={lt_normalized_raw:.3f} |{make_bar(lt_normalized_raw)}| → Curved={lt_normalized:.3f} |{make_bar(lt_normalized)}| ({lt_normalized/max(lt_normalized_raw, 0.001)*100:.1f}%)")
            if abs(rt_normalized_raw) > 0.01:
                print(f"RT: Raw={rt_normalized_raw:.3f} |{make_bar(rt_normalized_raw)}| → Curved={rt_normalized:.3f} |{make_bar(rt_normalized)}| ({rt_normalized/max(rt_normalized_raw, 0.001)*100:.1f}%)")
        
        # Trigger control for gripper tilt (wrist_flex) - swapped LT/RT
        tilt_delta = (lt_normalized - rt_normalized) * self.angle_step_deg
        if abs(tilt_delta) > 0.01:
            result = self.robot.execute_intuitive_move(
                tilt_gripper_down_angle=tilt_delta,
                use_interpolation=False,
            )
            if not result.ok:
                logger.warning(f"Gripper tilt failed: {result.msg}")
        
        # Base movement - send continuously, even if no movement to maintain position
        base_x = -ly * self.base_speed_scale
        base_y = 0.0  # Remove trigger strafe, use for gripper tilt instead
        base_theta = -lx * self.base_rot_speed_deg
        
        # Always send base action to maintain responsiveness
        base_action = {}
        
        # Add current arm positions
        for joint_name, position in self.robot.positions_deg.items():
            if joint_name != "gripper":  # Skip gripper for now
                base_action[f"arm_{joint_name}.pos"] = self.robot._deg_to_norm(joint_name, position)
        
        # Add base velocities (always send, even if zero)
        base_action.update({
            "x.vel": base_x,
            "y.vel": base_y,
            "theta.vel": base_theta,
        })
        
        try:
            self.robot.robot.send_action(base_action)
            if abs(base_x) > 0.01 or abs(base_y) > 0.01 or abs(base_theta) > 1.0:
                print(f"Base action sent: x={base_x:.2f}, y={base_y:.2f}, theta={base_theta:.2f}")
        except Exception as e:
            logger.error(f"Base move failed: {e}")

        # Check button states for debugging
        buttons_pressed = []
        for i in range(min(10, self.gamepad.joystick.get_numbuttons())):
            if self.gamepad.joystick.get_button(i):
                buttons_pressed.append(i)
        if buttons_pressed:
            print(f"Buttons pressed: {buttons_pressed}")
        
        # Specific Y-Button debugging
        if self.gamepad.joystick.get_button(4):
            print("Y-Button (4) detected as pressed!")
        
        # X button -> rotate gripper right (clockwise) - servo ID 5
        x_button_pressed = self.gamepad.joystick.get_button(3)
        if x_button_pressed:
            current_time = time.time()
            if self.x_button_press_time is None:
                self.x_button_press_time = current_time
            
            # Calculate speed factor based on hold time
            hold_time = current_time - self.x_button_press_time
            if hold_time >= self.gripper_rotation_speed_up_time:
                speed_factor = 1.0  # Full speed after 1 second
            else:
                speed_factor = self.gripper_rotation_slow_factor  # 25% speed initially
            
            rotation_angle = self.angle_step_deg * speed_factor
            result = self.robot.execute_intuitive_move(
                rotate_gripper_clockwise_angle=rotation_angle,
                use_interpolation=False,
            )
            print(f"X button pressed - rotate gripper right (speed: {speed_factor*100:.0f}%): {result.ok}")
            if not result.ok:
                logger.warning(f"Gripper rotate right failed: {result.msg}")
        else:
            self.x_button_press_time = None
            
        # Y button -> rotate gripper left (counter-clockwise) - servo ID 5
        y_button_pressed = self.gamepad.joystick.get_button(4)
        if y_button_pressed:
            current_time = time.time()
            if self.y_button_press_time is None:
                self.y_button_press_time = current_time
            
            # Calculate speed factor based on hold time
            hold_time = current_time - self.y_button_press_time
            if hold_time >= self.gripper_rotation_speed_up_time:
                speed_factor = 1.0  # Full speed after 1 second
            else:
                speed_factor = self.gripper_rotation_slow_factor  # 25% speed initially
            
            rotation_angle = -self.angle_step_deg * speed_factor
            result = self.robot.execute_intuitive_move(
                rotate_gripper_clockwise_angle=rotation_angle,
                use_interpolation=False,
            )
            print(f"Y button pressed - rotate gripper left (speed: {speed_factor*100:.0f}%): {result.ok}")
            if not result.ok:
                logger.warning(f"Gripper rotate left failed: {result.msg}")
        else:
            self.y_button_press_time = None
        # A button -> open gripper (non-blocking)
        if self.gamepad.joystick.get_button(0):
            try:
                self.robot.increment_joints_by_delta({"gripper": self.gripper_step_pct})
                print("A button - gripper open")
            except Exception as e:
                logger.error(f"Gripper open failed: {e}")
        # B button -> close gripper (non-blocking)
        if self.gamepad.joystick.get_button(1):
            try:
                self.robot.increment_joints_by_delta({"gripper": -self.gripper_step_pct})
                print("B button - gripper close")
            except Exception as e:
                logger.error(f"Gripper close failed: {e}")

        # BACK button (6) to exit
        if self.gamepad.joystick.get_button(6):
            self.stop()

    def run(self) -> None:
        self.start()
        try:
            while self.running:
                self._handle_inputs()
                time.sleep(0.02)  # Reduced from 0.05 to 0.02 for better responsiveness
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
