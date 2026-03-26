import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model



def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class KinovaInputs(transforms.DataTransformFn):
    """
    Inputs for Kinova datasets in LeRobot format.


    Expected inputs:
    - fixed_camera/wrist_camera/goal_image: [channel, height, width]
    - state: [13]: [x_ee(9), q_grip(1), f_ext(3)]
    - actions: [action_horizon, 10]: [x_ee(9), q_grip(1)]
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        fixed_camera = _parse_image(data["observation/fixed_camera"])
        wrist_camera = _parse_image(data["observation/wrist_camera"])
        goal_image = _parse_image(data["observation/goal_image"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": fixed_camera,
                "left_wrist_0_rgb": wrist_camera,
                "right_wrist_0_rgb": goal_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class KinovaOutputs(transforms.DataTransformFn):
    """Outputs for Kinova datasets in LeRobot format."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :10])}
