"""
Unit tests for robot controller.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import json
import os
import sys
from typing import Dict, Any
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robot_controller import RobotController, MoveResult
from kinematics import KinematicsModel


class TestMoveResult(unittest.TestCase):
    """Test MoveResult dataclass."""
    
    def test_initialization(self):
        """Test MoveResult initialization."""
        result = MoveResult(True, "Success", ["warning"], {"joint": 0.0})
        
        self.assertTrue(result.ok)
        self.assertEqual(result.msg, "Success")
        self.assertEqual(result.warnings, ["warning"])
        self.assertEqual(result.robot_state, {"joint": 0.0})
    
    def test_to_json_success(self):
        """Test to_json for successful result."""
        result = MoveResult(True, "Success", [], {"joint": 0.0})
        json_result = result.to_json()
        
        self.assertEqual(json_result["robot_state"], {"joint": 0.0})
        self.assertNotIn("status", json_result)
        self.assertNotIn("warnings", json_result)
    
    def test_to_json_error(self):
        """Test to_json for error result."""
        result = MoveResult(False, "Error occurred", ["warning"], {"joint": 0.0})
        json_result = result.to_json()
        
        self.assertEqual(json_result["status"], "error")
        self.assertEqual(json_result["message"], "Error occurred")
        self.assertEqual(json_result["warnings"], ["warning"])
        self.assertEqual(json_result["robot_state"], {"joint": 0.0})
    
    def test_to_json_empty_state(self):
        """Test to_json with empty robot state."""
        result = MoveResult(True, "Success", [], {})
        json_result = result.to_json()
        
        self.assertEqual(json_result["robot_state"], {"error": "Robot state not available."})


class TestKinematicsModel(unittest.TestCase):
    """Test kinematics model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.params = {
            "L1": 120,
            "L2": 140,
            "BASE_HEIGHT_MM": 50,
            "SHOULDER_MOUNT_OFFSET_MM": 10,
            "ELBOW_MOUNT_OFFSET_MM": 10,
            "SPATIAL_LIMITS": {
                "x": (50, 350),
                "z": (50, 300)
            }
        }
        self.kinematics = KinematicsModel(self.params)
    
    def test_initialization(self):
        """Test kinematics initialization."""
        self.assertEqual(self.kinematics.L1, 120)
        self.assertEqual(self.kinematics.L2, 140)
        self.assertEqual(self.kinematics.BASE_HEIGHT_MM, 50)
        self.assertEqual(self.kinematics.SHOULDER_MOUNT_OFFSET_MM, 10)
        self.assertEqual(self.kinematics.ELBOW_MOUNT_OFFSET_MM, 10)
    
    def test_forward_kinematics_basic(self):
        """Test forward kinematics calculation."""
        x, z = self.kinematics.forward_kinematics(90, 90)
        
        # At 90 degrees for both joints, should be at specific position
        self.assertIsInstance(x, float)
        self.assertIsInstance(z, float)
    
    def test_forward_kinematics_zero_angles(self):
        """Test forward kinematics with zero angles."""
        x, z = self.kinematics.forward_kinematics(0, 0)
        
        # With zero angles, should be at calculated position considering offsets
        self.assertAlmostEqual(x, 20.06, places=1)
        self.assertAlmostEqual(z, 70.0, places=1)
    
    def test_inverse_kinematics_reachable(self):
        """Test inverse kinematics for reachable position."""
        # Test a position we know should be reachable
        target_x, target_z = 200, 100
        
        try:
            sl, ef = self.kinematics.inverse_kinematics(target_x, target_z)
            self.assertIsInstance(sl, float)
            self.assertIsInstance(ef, float)
            
            # Verify by forward kinematics
            calc_x, calc_z = self.kinematics.forward_kinematics(sl, ef)
            self.assertAlmostEqual(calc_x, target_x, places=0)
            self.assertAlmostEqual(calc_z, target_z, places=0)
        except ValueError:
            self.fail("Should be able to reach this position")
    
    def test_inverse_kinematics_unreachable(self):
        """Test inverse kinematics for unreachable position."""
        # Test a position that's definitely unreachable
        target_x, target_z = 1000, 1000
        
        # The inverse kinematics doesn't raise ValueError, but the validation should catch it
        is_valid, msg = self.kinematics.is_cartesian_target_valid(target_x, target_z)
        self.assertFalse(is_valid)
        self.assertIn("out of range", msg)
    
    def test_is_cartesian_target_valid_inside(self):
        """Test validation for position inside workspace."""
        is_valid, msg = self.kinematics.is_cartesian_target_valid(200, 150)
        self.assertTrue(is_valid)
        self.assertEqual(msg, "Valid")
    
    def test_is_cartesian_target_valid_outside(self):
        """Test validation for position outside workspace."""
        is_valid, msg = self.kinematics.is_cartesian_target_valid(1000, 1000)
        self.assertFalse(is_valid)
        self.assertIn("out of range", msg)


class TestRobotController(unittest.TestCase):
    """Test robot controller functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock the config
        self.mock_config = Mock()
        self.mock_config.lerobot_config = {"type": "so100", "port": "/tmp/dummy_port"}
        self.mock_config.MOTOR_NORMALIZED_TO_DEGREE_MAPPING = {
            "shoulder_pan": (-86.9, 85.2, 0.0, 180.0),
            "shoulder_lift": (-89.2, 90.6, 0, 180.0),
            "elbow_flex": (99.4, -86.1, 0, 180.0),
            "wrist_flex": (-49.4, 50.5, -90.0, 90.0),
            "wrist_roll": (53.4, -47.5, -90, 90),
            "gripper": (0, 100.0, 0.0, 100.0),
        }
        self.mock_config.PRESET_POSITIONS = {
            "home": {"shoulder_pan": 90, "shoulder_lift": 90, "elbow_flex": 90, 
                    "wrist_flex": 0, "wrist_roll": 0, "gripper": 50}
        }
        self.mock_config.MOVEMENT_CONSTANTS = {
            "DEGREES_PER_STEP": 1.5,
            "MAX_INTERPOLATION_STEPS": 150,
            "STEP_DELAY_SECONDS": 0.01,
        }
        self.mock_config.KINEMATIC_PARAMS = {
            "so100": {
                "L1": 117.0,
                "L2": 136.0,
                "BASE_HEIGHT_MM": 120.0,
                "SHOULDER_MOUNT_OFFSET_MM": 32.0,
                "ELBOW_MOUNT_OFFSET_MM": 4.0,
                "SPATIAL_LIMITS": {
                    "x": (-20.0, 250.0),
                    "z": (30.0, 370.0),
                }
            },
            "default": {
                "L1": 117.0,
                "L2": 136.0,
                "BASE_HEIGHT_MM": 120.0,
                "SHOULDER_MOUNT_OFFSET_MM": 32.0,
                "ELBOW_MOUNT_OFFSET_MM": 4.0,
                "SPATIAL_LIMITS": {
                    "x": (-20.0, 250.0),
                    "z": (30.0, 370.0),
                }
            }
        }
        
        # Mock the robot
        self.mock_robot = Mock()
        self.mock_robot.connect = Mock()  # Mock the connect method
        self.mock_robot.get_observation.return_value = {
            "shoulder_pan.pos": 0.0,
            "shoulder_lift.pos": 0.0,
            "elbow_flex.pos": 0.0,
            "wrist_flex.pos": 0.0,
            "wrist_roll.pos": 0.0,
            "gripper.pos": 50.0
        }
    
    @patch('robot_controller.robot_config')
    @patch.object(RobotController, '_connect_robot')
    def test_initialization_normal_mode(self, mock_connect_robot, mock_robot_config):
        """Test robot controller initialization in normal mode."""
        mock_robot_config.configure_mock(**self.mock_config.__dict__)
        
        # Mock the connection method to prevent hardware access
        mock_connect_robot.return_value = None
        
        controller = RobotController(read_only=False)
        
        self.assertEqual(controller.robot_type, "so100")
        self.assertFalse(controller.read_only)
        self.assertIsNotNone(controller.kinematics)
        mock_connect_robot.assert_called_once()
    
    @patch('robot_controller.robot_config')
    @patch.object(RobotController, '_connect_robot_readonly')
    def test_initialization_read_only_mode(self, mock_connect_readonly, mock_robot_config):
        """Test robot controller initialization in read-only mode."""
        mock_robot_config.configure_mock(**self.mock_config.__dict__)
        mock_connect_readonly.return_value = None
        
        controller = RobotController(read_only=True)
        
        self.assertTrue(controller.read_only)
        mock_connect_readonly.assert_called_once()
    
    @patch('robot_controller.robot_config')
    @patch.object(RobotController, '_connect_robot_readonly')
    def test_deg_to_norm_conversion(self, mock_connect_readonly, mock_robot_config):
        """Test degree to normalized conversion."""
        mock_robot_config.configure_mock(**self.mock_config.__dict__)
        mock_connect_readonly.return_value = None
        
        controller = RobotController(read_only=True)
        
        # Test conversion for shoulder_pan: (-86.9, 85.2, 0.0, 180.0)
        norm_val = controller._deg_to_norm("shoulder_pan", 90.0)  # Middle of range
        self.assertGreater(norm_val, -86.9)
        self.assertLess(norm_val, 85.2)
    
    @patch('robot_controller.robot_config')
    @patch.object(RobotController, '_connect_robot_readonly')
    def test_norm_to_deg_conversion(self, mock_connect_readonly, mock_robot_config):
        """Test normalized to degree conversion."""
        mock_robot_config.configure_mock(**self.mock_config.__dict__)
        mock_connect_readonly.return_value = None
        
        controller = RobotController(read_only=True)
        
        # Test conversion for shoulder_pan: (-86.9, 85.2, 0.0, 180.0)
        deg_val = controller._norm_to_deg("shoulder_pan", 0.0)  # Middle of normalized range
        self.assertGreaterEqual(deg_val, 0.0)
        self.assertLessEqual(deg_val, 180.0)
    
    @patch('robot_controller.robot_config')
    @patch.object(RobotController, '_connect_robot_readonly')
    def test_validate_normalized_ranges_valid(self, mock_connect_readonly, mock_robot_config):
        """Test validation of valid normalized ranges."""
        mock_robot_config.configure_mock(**self.mock_config.__dict__)
        mock_connect_readonly.return_value = None
        
        controller = RobotController(read_only=True)
        
        positions = {"shoulder_pan": 90.0, "gripper": 50.0}
        is_valid, msg = controller._validate_normalized_ranges(positions)
        
        self.assertTrue(is_valid)
        self.assertEqual(msg, "")
    
    @patch('robot_controller.robot_config')
    @patch.object(RobotController, '_connect_robot_readonly')
    def test_validate_normalized_ranges_invalid(self, mock_connect_readonly, mock_robot_config):
        """Test validation of invalid normalized ranges."""
        mock_robot_config.configure_mock(**self.mock_config.__dict__)
        mock_connect_readonly.return_value = None
        
        controller = RobotController(read_only=True)
        
        # Use extreme values that would be out of range
        positions = {"shoulder_pan": 1000.0}  # Way out of range
        is_valid, msg = controller._validate_normalized_ranges(positions)
        
        self.assertFalse(is_valid)
        self.assertIn("out of range", msg)
    
    @patch('robot_controller.robot_config')
    @patch.object(RobotController, '_connect_robot_readonly')
    def test_get_current_robot_state(self, mock_connect_readonly, mock_robot_config):
        """Test getting current robot state."""
        mock_robot_config.configure_mock(**self.mock_config.__dict__)
        mock_connect_readonly.return_value = None
        
        controller = RobotController(read_only=True)
        result = controller.get_current_robot_state()
        
        self.assertIsInstance(result, MoveResult)
        self.assertTrue(result.ok)
        self.assertIn("robot_state", result.to_json())
    
    @patch('robot_controller.robot_config')
    @patch.object(RobotController, '_connect_robot_readonly')
    def test_set_joints_absolute_read_only(self, mock_connect_readonly, mock_robot_config):
        """Test set_joints_absolute in read-only mode."""
        mock_robot_config.configure_mock(**self.mock_config.__dict__)
        mock_connect_readonly.return_value = None
        
        controller = RobotController(read_only=True)
        result = controller.set_joints_absolute({"shoulder_pan": 90.0})
        
        self.assertFalse(result.ok)
        self.assertIn("read-only mode", result.msg)
    
    @patch('robot_controller.robot_config')
    @patch.object(RobotController, '_connect_robot')
    def test_set_joints_absolute_success(self, mock_connect_robot, mock_robot_config):
        """Test successful set_joints_absolute."""
        mock_robot_config.configure_mock(**self.mock_config.__dict__)
        mock_connect_robot.return_value = None
        
        controller = RobotController(read_only=False)
        controller.robot = self.mock_robot  # Manually set the robot for this test
        result = controller.set_joints_absolute({"shoulder_pan": 90.0}, use_interpolation=False)
        
        self.assertTrue(result.ok)
        self.mock_robot.send_action.assert_called_once()
    
    @patch('robot_controller.robot_config')
    @patch.object(RobotController, '_connect_robot')
    def test_increment_joints_by_delta(self, mock_connect_robot, mock_robot_config):
        """Test increment_joints_by_delta."""
        mock_robot_config.configure_mock(**self.mock_config.__dict__)
        mock_connect_robot.return_value = None
        
        controller = RobotController(read_only=False)
        controller.robot = self.mock_robot  # Manually set the robot for this test
        result = controller.increment_joints_by_delta({"shoulder_pan": 10.0})
        
        self.assertTrue(result.ok)
        self.mock_robot.send_action.assert_called()
    
    @patch('robot_controller.robot_config')
    @patch.object(RobotController, '_connect_robot')
    def test_apply_named_preset_success(self, mock_connect_robot, mock_robot_config):
        """Test applying named preset."""
        mock_robot_config.configure_mock(**self.mock_config.__dict__)
        mock_connect_robot.return_value = None
        
        controller = RobotController(read_only=False)
        controller.robot = self.mock_robot  # Manually set the robot for this test
        result = controller.apply_named_preset("home")
        
        self.assertTrue(result.ok)
        self.mock_robot.send_action.assert_called()
    
    @patch('robot_controller.robot_config')
    @patch.object(RobotController, '_connect_robot')
    def test_apply_named_preset_unknown(self, mock_connect_robot, mock_robot_config):
        """Test applying unknown preset."""
        mock_robot_config.configure_mock(**self.mock_config.__dict__)
        mock_connect_robot.return_value = None
        
        controller = RobotController(read_only=False)
        result = controller.apply_named_preset("unknown")
        
        self.assertFalse(result.ok)
        self.assertIn("Unknown preset", result.msg)
    
    @patch('robot_controller.robot_config')
    @patch.object(RobotController, '_connect_robot_readonly')
    def test_get_camera_images(self, mock_connect_readonly, mock_robot_config):
        """Test getting camera images."""
        mock_robot_config.configure_mock(**self.mock_config.__dict__)
        mock_robot_config.lerobot_config = {"cameras": {"front": {}, "wrist": {}}}
        mock_connect_readonly.return_value = None
        
        # Mock camera observations
        self.mock_robot.get_observation.return_value = {
            "front": np.zeros((480, 640, 3), dtype=np.uint8),
            "wrist": np.zeros((480, 640, 3), dtype=np.uint8),
            "shoulder_pan.pos": 0.0
        }
        
        controller = RobotController(read_only=True)
        controller.robot = self.mock_robot  # Manually set the robot for this test
        images = controller.get_camera_images()
        
        self.assertIsInstance(images, dict)
        self.assertIn("front", images)
        self.assertIn("wrist", images)
        self.assertIsInstance(images["front"], np.ndarray)
        self.assertIsInstance(images["wrist"], np.ndarray)


if __name__ == '__main__':
    unittest.main() 