"""
The main logic of the robot controller.
Used by all other scripts.
"""

import logging
import json
from typing import Dict, List, Optional, Tuple, Any 
import numpy as np
import math
import os

from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus
from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig
from config import robot_config
import time
from camera_controller import CameraController
import traceback
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__) # Use a named logger

@dataclass
class MoveResult:
    ok: bool
    msg: str
    warnings: List[str] = field(default_factory=list)
    robot_state: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict:
        final_robot_state = self.robot_state
        if not final_robot_state:
            final_robot_state = {"error": "Robot state not available."}
        
        json_output = {}

        # We put data in the JSON only if it is not empty
        # This is to avoid sending empty data to the LLM
        if not self.ok:
            json_output["status"] = "error"
        if self.msg:
            json_output["message"] = self.msg
        if self.warnings:
            json_output["warnings"] = self.warnings
        if self.robot_state:
            json_output["robot_state"] = self.robot_state

        # Single point of logging for the returned JSON
        logger.info(f"MoveResult JSON: {json.dumps(json_output)}")
        return json_output

class RobotController:
    """High-level controller that works in degrees and enforces per-joint limits.
    All public movement methods return a standardized dictionary:
    {
        "status": "success" | "error",
        "message": "Descriptive message of what happened",
        "warnings": ["List of non-critical issues encountered"],
        "robot_state": {
            "joint_positions_deg": {motor: angle, ...},
            "cartesian_mm": {"x": x_coord, "z": z_coord}
        }
    }
    """

    OPERATIONAL_DEGREE_LIMITS = robot_config.OPERATIONAL_DEGREE_LIMITS
    PRESET_POSITIONS = robot_config.PRESET_POSITIONS
    SPATIAL_LIMITS = robot_config.SPATIAL_LIMITS
    L1 = robot_config.L1
    L2 = robot_config.L2
    BASE_HEIGHT_MM = robot_config.BASE_HEIGHT_MM
    SHOULDER_OFFSET_ANGLE_RAD = math.asin(robot_config.SHOULDER_MOUNT_OFFSET_MM / L1)
    ELBOW_OFFSET_ANGLE_RAD = math.asin(robot_config.ELBOW_MOUNT_OFFSET_MM / L2)

    def __init__(self, update_goal_pos: bool = True):
        self.motor_names = robot_config.motors.keys()

        self.current_positions_deg: Dict[str, float] = {}
        self.current_cartesian_mm: Dict[str, float] = {"x": 0.0, "z": 0.0}

        bus_cfg = FeetechMotorsBusConfig(
            port=robot_config.port,
            motors=robot_config.motors,
        )
        self.motor_bus = FeetechMotorsBus(bus_cfg)
        self.motor_bus.connect()

        # Load the calibration file
        if os.path.exists(robot_config.calibration_file):
            with open(robot_config.calibration_file, "r") as f:
                self.motor_bus.set_calibration(json.load(f))
        else:
            error_msg = f"Calibration file {robot_config.calibration_file} not found"
            logging.error(error_msg)
            raise FileNotFoundError(error_msg)

        self._refresh_robot_state_cache_from_hw(update_goal_pos=update_goal_pos)

        # Initial state logging is good at INFO
        current_state_dict = self.get_current_robot_state().robot_state # Access the inner dict
        logging.info(f"RobotController initialized. State: {current_state_dict}")

        # Connect cameras
        self.camera_controller = CameraController(camera_configs=robot_config.cameras)
        self.camera_controller.connect()

    def _get_current_robot_state_dict_for_result(self) -> Dict[str, Any]:
        """
        Returns a dictionary of the current state of the robot.
        Returns raw motors angles and cartesian positions.
        Returns a human readable state that is more readable by the LLM.
        """

        robot_current_state = {}
        robot_current_state["robot_rotation_clockwise_deg"] = self.current_positions_deg["shoulder_pan"] - 90
        robot_current_state["gripper_heights_mm"] = self.current_cartesian_mm["z"]
        robot_current_state["gripper_linear_position_mm"] = self.current_cartesian_mm["x"]
        
        shoulder_lift_deg = self.current_positions_deg["shoulder_lift"]
        elbow_flex_deg = self.current_positions_deg["elbow_flex"]
        wrist_flex_deg = self.current_positions_deg["wrist_flex"]
        
        gripper_tilt_val = wrist_flex_deg + shoulder_lift_deg - elbow_flex_deg
        robot_current_state["gripper_tilt_deg"] = gripper_tilt_val
        robot_current_state["gripper_rotation_deg"] = self.current_positions_deg["wrist_roll"]
        robot_current_state["gripper_openness_pct"] = self.current_positions_deg["gripper"]

        return {
            "joint_positions_deg": {name: round(pos, 1) for name, pos in self.current_positions_deg.items()},
            "cartesian_mm": {name: round(pos, 1) for name, pos in self.current_cartesian_mm.items()},
            "human_readable_state": {name: round(pos, 1) for name, pos in robot_current_state.items()}
        }

    def _refresh_robot_state_cache_from_hw(self, update_goal_pos: bool = False):
        """Reads all joints from HW, updates cache, and recalculates FK. Returns list of error strings if any."""
        temp_positions: Dict[str, float] = {}
        try:
            raw_values = self.motor_bus.read("Present_Position", self.motor_names)
            for i, name in enumerate(self.motor_names):
                temp_positions[name] = float(np.asarray(raw_values[i]).flatten()[0])
        except Exception as e:
            logging.error(f"Failed to read motors positions ({e})")
            return

        self.current_positions_deg = temp_positions
        
        # Update Cartesian coordinates using FK
        fk_x, fk_z = self._forward_kinematics(
            self.current_positions_deg["shoulder_lift"],
            self.current_positions_deg["elbow_flex"]
        )

        self.current_cartesian_mm = {"x": fk_x, "z": fk_z}

        if update_goal_pos: # Typically only at init or after manual recovery
            try:
                goals = [self.current_positions_deg[name] for name in self.motor_names]
                self.motor_bus.write("Goal_Position", goals, self.motor_names)
            except Exception as e_goal:
                logging.error(f"Failed to update goal positions on HW: {e_goal}")
        
        return

    def _forward_kinematics(self, shoulder_lift_deg: float, elbow_flex_deg: float) -> tuple[float, float]:
        """Calculates x, z position of the wrist flex motor based on shoulder_lift and elbow_flex angles."""

        ang1_fk = math.radians(shoulder_lift_deg) + self.SHOULDER_OFFSET_ANGLE_RAD
        ang2_fk = math.radians(elbow_flex_deg) + self.ELBOW_OFFSET_ANGLE_RAD - math.radians(shoulder_lift_deg)

        x = -self.L1 * math.cos(ang1_fk) + self.L2 * math.cos(ang2_fk)
        z =  self.L1 * math.sin(ang1_fk) + self.L2 * math.sin(ang2_fk) + self.BASE_HEIGHT_MM
        return x, z


    def _inverse_kinematics(self, target_x: float, target_z: float) -> tuple[float, float]:
        """Calculates shoulder_lift and elbow_flex angles (degrees) for a target X, Z."""

        z_adj = target_z - self.BASE_HEIGHT_MM
        d_sq = target_x**2 + z_adj**2
        d = math.sqrt(d_sq)

        phi1 = math.atan2(z_adj, target_x)
        phi2 = math.acos(min(1.0, max(-1.0, (self.L1**2 + d_sq - self.L2**2) / (2 * self.L1 * d))))
        shoulder_lift_deg = 180.0 - math.degrees(phi1 + phi2) - math.degrees(self.SHOULDER_OFFSET_ANGLE_RAD)

        alpha1 = math.radians(shoulder_lift_deg) + self.SHOULDER_OFFSET_ANGLE_RAD
        
        cos2_arg = min(1.0, max(-1.0, (target_x + self.L1 * math.cos(alpha1)) / self.L2))
        sin2_arg = min(1.0, max(-1.0, (z_adj - self.L1 * math.sin(alpha1)) / self.L2))
        ang2 = math.atan2(sin2_arg, cos2_arg)
        
        elbow_flex_deg = math.degrees(ang2 + math.radians(shoulder_lift_deg)) - math.degrees(self.ELBOW_OFFSET_ANGLE_RAD)

        # Simplified formula - not working when elbow_flex_deg goes beyond 180 degrees
        # elbow_flex_deg = math.degrees(math.acos(min(1.0, max(-1.0, (self.L1**2 - d_sq + self.L2**2) / (2 * self.L1 * self.L2)))) - self.SHOULDER_OFFSET_ANGLE_RAD - self.ELBOW_OFFSET_ANGLE_RAD)

        return shoulder_lift_deg, elbow_flex_deg

    def _is_cartesian_target_valid(self, x: float, z: float) -> tuple[bool, str]:
        """
        Checks if the x,z position is valid.
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        if not (self.SPATIAL_LIMITS["x"][0] <= x <= self.SPATIAL_LIMITS["x"][1]):
            return False, f"Target X {x:.1f}mm out of range {self.SPATIAL_LIMITS['x']}"
        if not (self.SPATIAL_LIMITS["z"][0] <= z <= self.SPATIAL_LIMITS["z"][1]):
            return False, f"Target Z {z:.1f}mm out of range {self.SPATIAL_LIMITS['z']}"
        if x < 20 and z < 150:
            # Avoid robot hitting itself
            return False, f"Target ({x:.1f},{z:.1f})mm violates: if x < 20mm, z must be >= 150mm."
        
        # Check if point is within the maximum reachable circle
        z_adj = z - self.BASE_HEIGHT_MM
        distance = math.sqrt(z_adj**2 + x**2)
        max_reach = math.sqrt(self.L1**2) + math.sqrt(self.L2**2)
        
        if distance > max_reach - 1:
            # 1 is just a safety margin
            return False, f"Target ({x:.1f},{z:.1f})mm is beyond maximum reachable distance, max reach is {max_reach - 1:.1f}mm, distance is {distance:.1f}mm"
            
        return True, "Valid"

    def _are_joint_angles_valid(self, joint_positions: Dict[str, float]) -> Tuple[bool, str]:
        """Checks if all joint angles are within their operational limits.
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        for joint_name, angle in joint_positions.items():
            if joint_name not in self.OPERATIONAL_DEGREE_LIMITS:
                return False, f"Unknown joint '{joint_name}'"
            min_limit, max_limit = self.OPERATIONAL_DEGREE_LIMITS[joint_name]
            if angle < min_limit or angle > max_limit:
                return False, f"Joint {joint_name} angle {angle:.1f}deg is outside limits [{min_limit}, {max_limit}]"
        return True, "Valid"

    # --- PUBLIC API METHODS ---
    def get_current_robot_state(self) -> MoveResult:
        """Refreshes and returns the current robot state."""
        self._refresh_robot_state_cache_from_hw()
        return MoveResult(True, "Current robot state retrieved.", robot_state=self._get_current_robot_state_dict_for_result())

    def set_joints_absolute(self, positions_deg: Dict[str, float], use_interpolation: bool = True) -> MoveResult:
        """Sets one or more joints to absolute degree positions.
           If use_interpolation is True (default), uses dynamic steps.
           If False, moves directly to target in one step (for keyboard/testing).
        """
        aggregate_warnings: List[str] = []
        action_messages: List[str] = []
        
        # Filter for valid motor names and joints that actually need to move
        target_joint_moves: Dict[str, float] = {}
        for joint_name, target_angle in positions_deg.items():
            if joint_name not in self.motor_names:
                aggregate_warnings.append(f"Unknown joint '{joint_name}' ignored for absolute set.")
                continue
            
            target_joint_moves[joint_name] = target_angle

        if not target_joint_moves:
            msg = "No actual joint movements required"
            return MoveResult(True, msg, aggregate_warnings, self._get_current_robot_state_dict_for_result())

        # Check if all target positions are within limits
        is_valid, error_msg = self._are_joint_angles_valid(target_joint_moves)
        if not is_valid:
            return MoveResult(False, f"Movement rejected: {error_msg}", aggregate_warnings, self._get_current_robot_state_dict_for_result())

        num_processed = len(target_joint_moves)
        
        try:
            # Prepare for interpolation: get current HW positions for the joints we are about to move
            joints_to_move_list = list(target_joint_moves.keys())

            hw_current_positions_raw = self.motor_bus.read("Present_Position", joints_to_move_list)
            
            hw_current_positions = [float(np.asarray(val).flatten()[0]) for val in hw_current_positions_raw]
            target_positions_list = [target_joint_moves[j] for j in joints_to_move_list]
            
            steps = 1 # Default for non-interpolated move
            if use_interpolation:
                max_angle_change = 0.0
                for current_pos, target_pos in zip(hw_current_positions, target_positions_list):
                    change = abs(target_pos - current_pos)
                    if change > max_angle_change:
                        max_angle_change = change
                
                steps = max(1, min(150, int(max_angle_change / 1.0))) # 1 deg per step, capped 1-150

            for i in range(1, steps + 1):
                interpolated_positions_raw = [
                    current_pos + (target_pos - current_pos) * (i / steps)
                    for current_pos, target_pos in zip(hw_current_positions, target_positions_list)
                ]
                interpolated_positions_for_bus = [round(p, 2) for p in interpolated_positions_raw]

                self.motor_bus.write("Goal_Position", interpolated_positions_for_bus, joints_to_move_list)
                time.sleep(0.01) # 10ms delay
            
            # After loop, update cache to final target positions for moved joints
            for joint_name in joints_to_move_list:
                self.current_positions_deg[joint_name] = target_joint_moves[joint_name]
                action_messages.append(f"Moved {joint_name} to {target_joint_moves[joint_name]:.1f}deg.")
            
            # If shoulder_lift or elbow_flex moved, update Cartesian cache.
            if "shoulder_lift" in target_joint_moves or "elbow_flex" in target_joint_moves:
                self.current_cartesian_mm["x"], self.current_cartesian_mm["z"] = self._forward_kinematics(
                    self.current_positions_deg["shoulder_lift"],
                    self.current_positions_deg["elbow_flex"]
                )
            
        except Exception as e:
            err_msg = f"Hardware error during multi-joint move: {str(e)}"
            action_messages.append(err_msg)
            aggregate_warnings.append(err_msg)
            # Refresh entire cache from HW on error
            refresh_errors = self._refresh_robot_state_cache_from_hw() 
            aggregate_warnings.extend(refresh_errors)
            return MoveResult(False, f"Set {num_processed} joint(s) failed. Actions: {'; '.join(action_messages)}", 
                            aggregate_warnings, self._get_current_robot_state_dict_for_result())
        
        return MoveResult(True, "", aggregate_warnings, self._get_current_robot_state_dict_for_result())

    def increment_joints_by_delta(self, deltas_deg: Dict[str, float]) -> MoveResult:
        """Increments one or more joints by delta degrees.
        """

        target_positions: Dict[str, float] = {}
        local_warnings: List[str] = []

        for joint_name, delta in deltas_deg.items():
            if joint_name not in self.motor_names:
                local_warnings.append(f"Unknown joint '{joint_name}' for delta ignored.")
                continue
            # Ensure fresh current angle from cache before calculating target
            current_angle = self.current_positions_deg[joint_name]
            target_positions[joint_name] = current_angle + delta
        
        if not target_positions:
            return MoveResult(False, "No valid joints provided for increment.", local_warnings, self._get_current_robot_state_dict_for_result())

        result = self.set_joints_absolute(target_positions, use_interpolation=True)
        all_warnings = local_warnings + (result.warnings if result.warnings else [])
        return MoveResult(result.ok, "", all_warnings, result.robot_state)

    def calculate_target_joint_angles_from_cartesian_deltas(
        self, deltas_mm: Dict[str, float]
    ) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]], List[str], Optional[str]]:
        """
        Calculates target joint angles for shoulder_lift, elbow_flex, and optionally wrist_flex
        based on Cartesian deltas, without actually moving the robot.

        Returns:
            Tuple containing:
            - calculated_target_joints (Dict[str, float]): {'shoulder_lift': deg, 'elbow_flex': deg, 'wrist_flex': deg_comp} or None if error.
            - achieved_cartesian_coords (Dict[str, float]): {'x': mm, 'z': mm} or None if error.
            - warnings (List[str]): List of warnings encountered.
            - error_message (Optional[str]): Error message if calculation failed, None otherwise.
        """
        warnings_list: List[str] = []
        
        target_x_mm = self.current_cartesian_mm["x"] + deltas_mm.get("x", 0.0)
        target_z_mm = self.current_cartesian_mm["z"] + deltas_mm.get("z", 0.0)

        is_valid, validity_msg = self._is_cartesian_target_valid(target_x_mm, target_z_mm)
        if not is_valid:
            return None, None, warnings_list, f"Cartesian target invalid: {validity_msg}"

        calculated_target_joints: Dict[str, float] = {}
        achieved_cartesian_coords: Dict[str, float] = {}

        try:
            sl_target_deg, ef_target_deg = self._inverse_kinematics(target_x_mm, target_z_mm)

            calculated_target_joints["shoulder_lift"] = sl_target_deg
            calculated_target_joints["elbow_flex"] = ef_target_deg

            # Compensation based on maintaining gripper_tilt = wrist_flex + shoulder_lift - elbow_flex constant
            old_sl_deg = self.current_positions_deg["shoulder_lift"]
            old_ef_deg = self.current_positions_deg["elbow_flex"]
            old_wf_deg = self.current_positions_deg["wrist_flex"]

            sl_change_deg = sl_target_deg - old_sl_deg
            ef_change_deg = ef_target_deg - old_ef_deg
            wf_target_comp_deg = old_wf_deg - (sl_change_deg - ef_change_deg)

            calculated_target_joints["wrist_flex"] = wf_target_comp_deg
            
            achieved_cartesian_coords["x"] = target_x_mm
            achieved_cartesian_coords["z"] = target_z_mm

            return calculated_target_joints, achieved_cartesian_coords, warnings_list, None

        except ValueError as e_ik: 
            logger.warning(f"IK Error for target ({target_x_mm:.1f},{target_z_mm:.1f}): {e_ik} during calculation.")
            return None, None, warnings_list, f"Cartesian IK Error: {e_ik}"
        except Exception as e_gen: 
            logger.error(f"Unexpected error during Cartesian calculation for ({target_x_mm:.1f},{target_z_mm:.1f}): {e_gen}", exc_info=True)
            return None, None, warnings_list, f"Unexpected error in Cartesian calculation: {e_gen}"
        

    def execute_intuitive_move(
        self,
        move_gripper_up_mm: Optional[float] = None,
        move_gripper_forward_mm: Optional[float] = None,
        tilt_gripper_down_angle: Optional[float] = None,
        rotate_gripper_clockwise_angle: Optional[float] = None,
        rotate_robot_clockwise_angle: Optional[float] = None,
        use_interpolation_for_move: bool = True
    ) -> MoveResult:
        """
        Executes a robot move based on intuitive, high-level commands.
        Calculates target joint positions from Cartesian deltas (if any),
        adds direct joint deltas, and then executes a single multi-joint move.
        This is intended for use by keyboard controller and MCP.
        """
        logger.info(f"EXEC_INTUITIVE_MOVE: Start. Current cache pos: {self.current_positions_deg}")
        logger.info(f"EXEC_INTUITIVE_MOVE: Inputs: up_mm={move_gripper_up_mm}, fwd_mm={move_gripper_forward_mm}, tilt_deg={tilt_gripper_down_angle}, grip_rot_deg={rotate_gripper_clockwise_angle}, robot_rot_deg={rotate_robot_clockwise_angle}")

        overall_warnings: List[str] = []
        final_target_joint_positions = self.current_positions_deg.copy()
        
        # 1. Compute joint positions from cartesian deltas
        if move_gripper_up_mm is not None or move_gripper_forward_mm is not None:
            calculated_cartesian_joints, _, cart_calc_warnings, cart_calc_error = \
                self.calculate_target_joint_angles_from_cartesian_deltas(
                    {
                        'x': move_gripper_forward_mm or 0.0,
                        'z': move_gripper_up_mm or 0.0
                    }
                )
            
            if cart_calc_warnings:
                overall_warnings.extend(cart_calc_warnings)
            
            if cart_calc_error:
                return MoveResult(False, f"Movement rejected: {cart_calc_error}", 
                                overall_warnings, self._get_current_robot_state_dict_for_result())

            if calculated_cartesian_joints:
                for joint_name, target_angle in calculated_cartesian_joints.items():
                    final_target_joint_positions[joint_name] = target_angle

        # 2. Apply direct joint deltas
        if tilt_gripper_down_angle is not None:
            final_target_joint_positions['wrist_flex'] += float(tilt_gripper_down_angle)
        if rotate_gripper_clockwise_angle is not None:
            final_target_joint_positions['wrist_roll'] += float(rotate_gripper_clockwise_angle)
        if rotate_robot_clockwise_angle is not None:
            final_target_joint_positions['shoulder_pan'] += float(rotate_robot_clockwise_angle)

        result = self.set_joints_absolute(final_target_joint_positions, use_interpolation=use_interpolation_for_move)

        return result

    def apply_named_preset(self, preset_key: str) -> MoveResult:
        """Applies a predefined named preset position."""
        if preset_key not in self.PRESET_POSITIONS:
            return MoveResult(False, f"Unknown preset key: '{preset_key}'. Available: {list(self.PRESET_POSITIONS.keys())}", robot_state=self._get_current_robot_state_dict_for_result())
        
        preset_positions = self.PRESET_POSITIONS[preset_key]
        logging.info(f"Applying preset '{preset_key}': {preset_positions}")
        
        result = self.set_joints_absolute(preset_positions)
        return MoveResult(result.ok, "", result.warnings, result.robot_state)

    def get_camera_images(self) -> Dict[str, np.ndarray]:
        if not self.camera_controller.is_connected:
            logging.warning("get_camera_images: Cameras not connected.")
            return {}
        try: 
            return self.camera_controller.get_images()
        except Exception as e: 
            logging.error(f"Error getting images: {e} {traceback.format_exc()}")
            return {}
    
    def disconnect_cameras(self) -> None:
        if hasattr(self, 'camera_controller') and self.camera_controller.is_connected:
            self.camera_controller.disconnect(); logging.info("Cameras disconnected.")

    def disconnect(self, reset_pos: bool = True) -> None:
        if hasattr(self, 'motor_bus') and self.motor_bus.is_connected:
            logging.info("Robot disconnect: Attempting to move to rest preset '1'.")
            try:
                if reset_pos:
                    # Return to the rest preset position
                    rest_result = self.apply_named_preset("1")
                if not rest_result.ok:
                    logging.warning(f"Robot disconnect: Rest preset move not fully successful: {rest_result.msg} Warnings: {rest_result.warnings}")
            except Exception as e_rest:
                 logging.error(f"Robot disconnect: Exception during move to rest: {e_rest}", exc_info=True)
            finally: 
                self.motor_bus.disconnect()
        
        self.disconnect_cameras()
        logging.info("RobotController fully disconnected.")
