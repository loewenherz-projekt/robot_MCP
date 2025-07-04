import sys
from enum import IntEnum
from typing import Any

from ..teleoperator import Teleoperator
from ..gamepad.gamepad_utils import GamepadController, GamepadControllerHID
from .configuration_lekiwi_gamepad import LeKiwiGamepadConfig


class GripperAction(IntEnum):
    CLOSE = 0
    STAY = 1
    OPEN = 2


gripper_action_map = {
    "close": GripperAction.CLOSE.value,
    "open": GripperAction.OPEN.value,
    "stay": GripperAction.STAY.value,
}


class LeKiwiGamepadTeleop(Teleoperator):
    """Gamepad teleoperation for the LeKiwi robot."""

    config_class = LeKiwiGamepadConfig
    name = "lekiwi_gamepad"

    def __init__(self, config: LeKiwiGamepadConfig):
        super().__init__(config)
        self.config = config
        self.gamepad = None

    @property
    def action_features(self) -> dict:
        names = {
            "delta_x": 0,
            "delta_y": 1,
            "delta_z": 2,
            "x.vel": 3,
            "y.vel": 4,
            "theta.vel": 5,
        }
        if self.config.use_gripper:
            names["gripper"] = 6
        return {"dtype": "float32", "shape": (len(names),), "names": names}

    @property
    def feedback_features(self) -> dict:
        return {}

    def connect(self) -> None:
        if sys.platform == "darwin":
            Gamepad = GamepadControllerHID
        else:
            Gamepad = GamepadController
        self.gamepad = Gamepad()
        self.gamepad.start()

    def get_action(self) -> dict[str, Any]:
        self.gamepad.update()
        lx, ly, rx, ry, lt, rt = self.gamepad.get_axis_values()

        base_x = -ly * self.config.base_speed_scale
        base_y = -lx * self.config.base_speed_scale
        delta_x = -ry * self.config.arm_step_size
        delta_y = -rx * self.config.arm_step_size
        delta_z = (rt - lt) * self.config.arm_step_size

        action = {
            "delta_x": delta_x,
            "delta_y": delta_y,
            "delta_z": delta_z,
            "x.vel": base_x,
            "y.vel": base_y,
            "theta.vel": 0.0,
        }

        if self.config.use_gripper:
            gripper_command = self.gamepad.gripper_command()
            action["gripper"] = gripper_action_map[gripper_command]
        return action

    def disconnect(self) -> None:
        if self.gamepad is not None:
            self.gamepad.stop()
            self.gamepad = None

    def is_connected(self) -> bool:
        return self.gamepad is not None

    def calibrate(self) -> None:
        pass

    def is_calibrated(self) -> bool:
        return True

    def configure(self) -> None:
        pass

    def send_feedback(self, feedback: dict) -> None:
        pass
