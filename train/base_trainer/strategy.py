from abc import ABC
from typing import Any, Dict

from torch.optim import Optimizer

from scene.gaussians import BaseGaussians

from .configs import BaseConfigs


class BaseRefinementStrategy(ABC):
    def __init__(
        self,
        config: BaseConfigs,
        step: int,
        optimizer: Optimizer,
        gaussians: BaseGaussians,
    ):
        self._optimizer = optimizer
        self._gaussians = gaussians
        self._step = step

    def step(self, end_points: Dict[str, Any]) -> None:
        self._step += 1

    def step_after_backward(self, end_points: Dict[str, Any]) -> None:
        pass
