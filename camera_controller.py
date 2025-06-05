"""
Managing cameras attached to the robot.
"""

import logging
from typing import Dict, Optional
import numpy as np

from lerobot.common.robot_devices.cameras.opencv import OpenCVCamera, find_cameras
from lerobot.common.robot_devices.cameras.configs import OpenCVCameraConfig, CameraConfig
from config import robot_config

class CameraController:
    """Controller for managing multiple cameras attached to the robot."""
    
    def __init__(self, camera_configs: Optional[Dict[str, CameraConfig]] = None):
        """
        Initialize the camera controller.
        
        Args:
            camera_configs: Dictionary mapping camera names to their configurations.
                           If None, uses the configs from robot_config.
        """
        self.camera_configs = camera_configs if camera_configs is not None else robot_config.cameras
        self.cameras = {}
        self.is_connected = False
        
        if not self.camera_configs:
            logging.warning("No camera configurations specified")
    
    def connect(self):
        """Connect to all cameras specified in camera_configs."""
        if self.is_connected:
            logging.warning("Cameras are already connected")
            return
            
        if not self.camera_configs:
            logging.warning("No camera configurations specified.")
            return
            
        for name, config in self.camera_configs.items():
            try:
                camera = OpenCVCamera(config)
                camera.connect()
                self.cameras[name] = camera
                logging.info(f"Connected to camera '{name}' with resolution {camera.width}x{camera.height} @ {camera.fps}fps")
            except Exception as e:
                logging.error(f"Failed to connect to camera '{name}': {e}")
        
        self.is_connected = len(self.cameras) > 0
        
        if not self.is_connected:
            logging.warning("Failed to connect to any cameras")
    
    def get_images(self) -> Dict[str, np.ndarray]:
        """
        Get images from all connected cameras.
        
        Returns:
            Dictionary mapping camera names to numpy arrays containing the images
        """
        if not self.is_connected:
            raise RuntimeError("Cameras are not connected. Call connect() first.")
            
        images = {}
        for name, camera in self.cameras.items():
            try:
                images[name] = camera.read()
            except Exception as e:
                logging.error(f"Failed to read from camera '{name}': {e}")
        
        return images
    
    def disconnect(self):
        """Disconnect from all cameras."""
        if not self.is_connected:
            return
            
        for name, camera in self.cameras.items():
            try:
                camera.disconnect()
                logging.info(f"Disconnected from camera '{name}'")
            except Exception as e:
                logging.error(f"Error disconnecting from camera '{name}': {e}")
        
        self.cameras = {}
        self.is_connected = False
    
    def __del__(self):
        """Ensure cameras are disconnected when object is garbage collected."""
        self.disconnect()
