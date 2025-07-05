"""
Unit tests for configuration module.
"""

import unittest
from unittest.mock import Mock, patch
import os
import sys
from dataclasses import fields

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import RobotConfig, robot_config


class TestRobotConfig(unittest.TestCase):
    """Test RobotConfig dataclass."""
    
    def test_default_initialization(self):
        """Test RobotConfig with default values."""
        config = RobotConfig()
        
        # Test that all required fields exist
        field_names = {field.name for field in fields(RobotConfig)}
        expected_fields = {
            'lerobot_config', 'MOTOR_NORMALIZED_TO_DEGREE_MAPPING',
            'PRESET_POSITIONS', 'MOVEMENT_CONSTANTS', 'KINEMATIC_PARAMS',
            'robot_description'
        }
        
        self.assertTrue(expected_fields.issubset(field_names))
        
        # Test that mappings are properly structured
        self.assertIsInstance(config.MOTOR_NORMALIZED_TO_DEGREE_MAPPING, dict)
        self.assertIsInstance(config.PRESET_POSITIONS, dict)
        self.assertIsInstance(config.MOVEMENT_CONSTANTS, dict)
        self.assertIsInstance(config.KINEMATIC_PARAMS, dict)
        self.assertIsInstance(config.robot_description, str)
    
    def test_motor_mapping_structure(self):
        """Test motor mapping has correct structure."""
        config = RobotConfig()
        
        required_joints = {
            'shoulder_pan', 'shoulder_lift', 'elbow_flex',
            'wrist_flex', 'wrist_roll', 'gripper'
        }
        
        # Check all required joints are present
        self.assertTrue(required_joints.issubset(set(config.MOTOR_NORMALIZED_TO_DEGREE_MAPPING.keys())))
        
        # Check each mapping has 4 values (norm_min, norm_max, deg_min, deg_max)
        for joint_name, mapping in config.MOTOR_NORMALIZED_TO_DEGREE_MAPPING.items():
            self.assertEqual(len(mapping), 4, f"Joint {joint_name} should have 4 mapping values")
            self.assertTrue(all(isinstance(val, (int, float)) for val in mapping))
    
    def test_preset_positions_structure(self):
        """Test preset positions have correct structure."""
        config = RobotConfig()
        
        # Should have at least some preset positions
        self.assertGreater(len(config.PRESET_POSITIONS), 0)
        
        # Each preset should be a dict of joint positions
        for preset_name, positions in config.PRESET_POSITIONS.items():
            self.assertIsInstance(positions, dict, f"Preset {preset_name} should be a dict")
            
            # All values should be numeric
            for joint, value in positions.items():
                self.assertIsInstance(value, (int, float), 
                                    f"Joint {joint} in preset {preset_name} should be numeric")
    
    def test_movement_constants_structure(self):
        """Test movement constants have correct structure."""
        config = RobotConfig()
        
        required_constants = {
            'DEGREES_PER_STEP', 'MAX_INTERPOLATION_STEPS', 'STEP_DELAY_SECONDS'
        }
        
        # Check all required constants are present
        self.assertTrue(required_constants.issubset(set(config.MOVEMENT_CONSTANTS.keys())))
        
        # Check all values are numeric and positive
        for constant, value in config.MOVEMENT_CONSTANTS.items():
            self.assertIsInstance(value, (int, float), f"Constant {constant} should be numeric")
            self.assertGreater(value, 0, f"Constant {constant} should be positive")
    
    def test_kinematic_params_structure(self):
        """Test kinematic parameters have correct structure."""
        config = RobotConfig()
        
        required_robot_types = {'default', 'lekiwi'}
        
        # Check required robot types are present
        self.assertTrue(required_robot_types.issubset(set(config.KINEMATIC_PARAMS.keys())))
        
        required_params = {
            'L1', 'L2', 'BASE_HEIGHT_MM', 'SHOULDER_MOUNT_OFFSET_MM', 
            'ELBOW_MOUNT_OFFSET_MM', 'SPATIAL_LIMITS'
        }
        
        # Check each robot type has all required parameters
        for robot_type, params in config.KINEMATIC_PARAMS.items():
            self.assertTrue(required_params.issubset(set(params.keys())),
                          f"Robot type {robot_type} missing required parameters")
            
            # All values should be appropriate types
            for param, value in params.items():
                if param == 'SPATIAL_LIMITS':
                    self.assertIsInstance(value, dict, f"SPATIAL_LIMITS for {robot_type} should be a dict")
                    self.assertIn('x', value, f"SPATIAL_LIMITS for {robot_type} should have 'x' key")
                    self.assertIn('z', value, f"SPATIAL_LIMITS for {robot_type} should have 'z' key")
                else:
                    self.assertIsInstance(value, (int, float), 
                                        f"Parameter {param} for {robot_type} should be numeric")
                    self.assertGreater(value, 0, 
                                     f"Parameter {param} for {robot_type} should be positive")
    
    def test_robot_description_content(self):
        """Test robot description has meaningful content."""
        config = RobotConfig()
        
        self.assertIsInstance(config.robot_description, str)
        self.assertGreater(len(config.robot_description.strip()), 50, 
                          "Robot description should be substantial")
        
        # Should contain key robot-related terms
        description_lower = config.robot_description.lower()
        expected_terms = ['robot', 'gripper', 'move']
        
        for term in expected_terms:
            self.assertIn(term, description_lower, 
                         f"Robot description should mention '{term}'")


class TestRobotConfigInstance(unittest.TestCase):
    """Test the global robot_config instance."""
    
    def test_global_config_exists(self):
        """Test that global robot_config instance exists."""
        self.assertIsInstance(robot_config, RobotConfig)
    
    def test_global_config_lerobot_structure(self):
        """Test global config lerobot_config structure."""
        # Should have basic lerobot configuration
        self.assertIsInstance(robot_config.lerobot_config, dict)
        
        # Should have robot type
        if 'type' in robot_config.lerobot_config:
            self.assertIsInstance(robot_config.lerobot_config['type'], str)
    
    def test_global_config_consistency(self):
        """Test that global config values are consistent."""
        # Motor mappings should be consistent with joint names
        motor_joints = set(robot_config.MOTOR_NORMALIZED_TO_DEGREE_MAPPING.keys())
        
        # Preset positions should only reference valid joints
        for preset_name, positions in robot_config.PRESET_POSITIONS.items():
            preset_joints = set(positions.keys())
            self.assertTrue(preset_joints.issubset(motor_joints),
                          f"Preset {preset_name} references unknown joints: "
                          f"{preset_joints - motor_joints}")
    
    def test_global_config_value_ranges(self):
        """Test that global config values are in reasonable ranges."""
        # Movement constants should be reasonable
        constants = robot_config.MOVEMENT_CONSTANTS
        
        # Degrees per step should be small but not too small
        self.assertGreater(constants['DEGREES_PER_STEP'], 0.1)
        self.assertLess(constants['DEGREES_PER_STEP'], 10.0)
        
        # Max interpolation steps should be reasonable
        self.assertGreater(constants['MAX_INTERPOLATION_STEPS'], 10)
        self.assertLess(constants['MAX_INTERPOLATION_STEPS'], 1000)
        
        # Step delay should be small
        self.assertGreater(constants['STEP_DELAY_SECONDS'], 0.001)
        self.assertLess(constants['STEP_DELAY_SECONDS'], 1.0)


class TestConfigurationValidation(unittest.TestCase):
    """Test configuration validation and edge cases."""
    
    def test_motor_mapping_ranges(self):
        """Test motor mapping ranges are valid."""
        config = RobotConfig()
        
        for joint_name, (norm_min, norm_max, deg_min, deg_max) in config.MOTOR_NORMALIZED_TO_DEGREE_MAPPING.items():
            # Degree ranges should be valid
            if deg_min != deg_max:  # Allow for fixed joints
                self.assertNotEqual(deg_min, deg_max, f"Joint {joint_name} has invalid degree range")
            
            # Normalized ranges should be reasonable
            if norm_min != norm_max:  # Allow for fixed joints
                self.assertNotEqual(norm_min, norm_max, f"Joint {joint_name} has invalid normalized range")
            
            # Values should be finite
            for value in [norm_min, norm_max, deg_min, deg_max]:
                self.assertTrue(abs(value) < 1e6, f"Joint {joint_name} has extreme value: {value}")
    
    def test_preset_positions_validity(self):
        """Test that preset positions are within motor ranges."""
        config = RobotConfig()
        
        for preset_name, positions in config.PRESET_POSITIONS.items():
            for joint_name, position in positions.items():
                if joint_name in config.MOTOR_NORMALIZED_TO_DEGREE_MAPPING:
                    norm_min, norm_max, deg_min, deg_max = config.MOTOR_NORMALIZED_TO_DEGREE_MAPPING[joint_name]
                    
                    # Position should be within degree range
                    self.assertGreaterEqual(position, min(deg_min, deg_max),
                                          f"Preset {preset_name} joint {joint_name} below range")
                    self.assertLessEqual(position, max(deg_min, deg_max),
                                       f"Preset {preset_name} joint {joint_name} above range")
    
    def test_kinematic_params_consistency(self):
        """Test kinematic parameters are consistent across robot types."""
        config = RobotConfig()
        
        # All robot types should have same parameter structure
        param_sets = [set(params.keys()) for params in config.KINEMATIC_PARAMS.values()]
        
        if len(param_sets) > 1:
            first_set = param_sets[0]
            for i, param_set in enumerate(param_sets[1:], 1):
                self.assertEqual(first_set, param_set, 
                               f"Robot type {i} has different kinematic parameters")


if __name__ == '__main__':
    unittest.main() 