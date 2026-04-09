from typing import Dict, List, Literal

import torch
from torch.nn import Parameter
from torch.optim import Optimizer

from .base_gaussians import BaseGaussians, RenderGaussiansParams, STDGaussiansParams
from .gaussians_3d import Gaussians3D
from .gaussians_ray_v2 import GaussiansRayV2


class GaussiansMixed(BaseGaussians):
    def __init__(self, sh_degree: int = 3, device: str | torch.device = "cuda") -> None:
        self.gaussians_3d = Gaussians3D(sh_degree, device, "_3d")
        self.gaussians_ray = GaussiansRayV2(sh_degree, device, "_ray")
        self._device = device
        self._sh_degree = sh_degree
        self._sh_rest_dim = (sh_degree + 1) ** 2 - 1

    def __len__(self) -> int:
        return len(self.gaussians_3d) + len(self.gaussians_ray)

    @property
    def params_dict(self) -> torch.nn.ParameterDict:
        _prama_dict = torch.nn.ParameterDict()
        _prama_dict.update(self.gaussians_3d.params_dict)
        _prama_dict.update(self.gaussians_ray.params_dict)
        return _prama_dict

    def parameters(
        self,
    ) -> List[Dict[Literal["name", "params"], str | torch.nn.Parameter]]:
        return self.gaussians_3d.parameters() + self.gaussians_ray.parameters()

    def get_std_parameters(self) -> STDGaussiansParams:
        return (
            self.gaussians_3d.get_std_parameters()
            + self.gaussians_ray.get_std_parameters()
        )

    def get_render_parameters(
        self, degeree_to_use: int | None = None
    ) -> RenderGaussiansParams:
        return self.gaussians_3d.get_render_parameters(
            degeree_to_use
        ) + self.gaussians_ray.get_render_parameters(degeree_to_use)

    def add_parameters(
        self,
        optimizer: Optimizer,
        additional_params: Dict[str, torch.Tensor | Parameter],
    ) -> None:
        raise NotImplementedError("Call add_parameters on each gaussians separately.")

    def prune_parameters(self, optimizer: Optimizer, mask: torch.Tensor) -> None:
        raise NotImplementedError("Call prune_parameters on each gaussians separately.")

    def reset_parameters(
        self, optimizer: Optimizer, new_params: Dict[str, torch.Tensor]
    ) -> None:
        raise NotImplementedError("Call reset_parameters on each gaussians separately.")
