from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("lekiwi_gamepad")
@dataclass
class LeKiwiGamepadConfig(TeleoperatorConfig):
    use_gripper: bool = True
    base_speed_scale: float = 0.3
    arm_step_size: float = 1.0

