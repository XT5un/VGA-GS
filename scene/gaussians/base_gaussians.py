from abc import ABC, abstractmethod
from typing import Dict, List, Literal, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Parameter, ParameterDict
from torch.optim import Optimizer

from utils import write_ply_file


class STDGaussiansParams:
    """all kind of gaussians should have to export this data"""

    def __init__(
        self,
        means: Parameter | Tensor,
        quats: Parameter | Tensor,
        scales: Parameter | Tensor,
        opacities: Parameter | Tensor,
        features_dc: Parameter | Tensor,
        features_rest: Parameter | Tensor,
    ) -> None:
        self.means = means
        self.quats = quats
        self.scales = scales
        self.opacities = opacities
        self.features_dc = features_dc
        self.features_rest = features_rest

    def __add__(self, other: "STDGaussiansParams") -> "STDGaussiansParams":
        return STDGaussiansParams(
            means=torch.cat([self.means, other.means], dim=0),
            quats=torch.cat([self.quats, other.quats], dim=0),
            scales=torch.cat([self.scales, other.scales], dim=0),
            opacities=torch.cat([self.opacities, other.opacities], dim=0),
            features_dc=torch.cat([self.features_dc, other.features_dc], dim=0),
            features_rest=torch.cat([self.features_rest, other.features_rest], dim=0),
        )

    @staticmethod
    def cat(
        params_list: List["STDGaussiansParams"],
    ) -> "STDGaussiansParams":
        return STDGaussiansParams(
            means=torch.cat([p.means for p in params_list], dim=0),
            quats=torch.cat([p.quats for p in params_list], dim=0),
            scales=torch.cat([p.scales for p in params_list], dim=0),
            opacities=torch.cat([p.opacities for p in params_list], dim=0),
            features_dc=torch.cat([p.features_dc for p in params_list], dim=0),
            features_rest=torch.cat([p.features_rest for p in params_list], dim=0),
        )


class RenderGaussiansParams:
    def __init__(
        self,
        means: Parameter | Tensor,
        quats: Parameter | Tensor,
        scales: Parameter | Tensor,
        opacities: Parameter | Tensor,
        sh_coeffs: Parameter | Tensor | None = None,
        precomputed_colors: Parameter | Tensor | None = None,
        degree_to_use: int = 3,
    ) -> None:
        if precomputed_colors is not None:
            sh_coeffs = None

        self.means = means
        self.quats = quats
        self.scales = scales
        self.opacities = opacities
        self.sh_coeffs = sh_coeffs
        self.precomputed_colors = precomputed_colors
        self.degree_to_use = degree_to_use

    def __add__(self, other: "RenderGaussiansParams") -> "RenderGaussiansParams":
        if self.degree_to_use != other.degree_to_use:
            raise ValueError("degree_to_use must be the same")
        if type(self.precomputed_colors) != type(other.precomputed_colors):
            raise ValueError("precomputed_colors must be the same type")
        if type(self.sh_coeffs) != type(other.sh_coeffs):
            raise ValueError("sh_coeffs must be the same type")

        return RenderGaussiansParams(
            means=torch.cat([self.means, other.means], dim=0),
            quats=torch.cat([self.quats, other.quats], dim=0),
            scales=torch.cat([self.scales, other.scales], dim=0),
            opacities=torch.cat([self.opacities, other.opacities], dim=0),
            sh_coeffs=(
                torch.cat([self.sh_coeffs, other.sh_coeffs], dim=0)
                if self.sh_coeffs is not None
                else None
            ),
            precomputed_colors=(
                torch.cat([self.precomputed_colors, other.precomputed_colors], dim=0)
                if self.precomputed_colors is not None
                else None
            ),
            degree_to_use=self.degree_to_use,
        )

    @staticmethod
    def create_from_std_params(
        std_params: STDGaussiansParams,
        degree_to_use: int = 3,
    ) -> "RenderGaussiansParams":
        means = std_params.means
        quats = std_params.quats / torch.norm(std_params.quats, dim=-1, keepdim=True)
        scales = torch.exp(std_params.scales)
        opacities = torch.sigmoid(std_params.opacities)
        sh_coeffs = torch.cat(
            [std_params.features_dc, std_params.features_rest], dim=-2
        )
        return RenderGaussiansParams(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            sh_coeffs=sh_coeffs,
            degree_to_use=degree_to_use,
        )


class BaseGaussians(ABC):
    """
    Base class for all kinds of gaussians.

    It provides the basic interface for the gaussian rasterizer and io.
    """

    def __init__(
        self,
        sh_degree: int = 3,
        device: str | torch.device = "cuda",
        suffix: str | None = None,
    ) -> None:
        self._device = torch.device(device) if isinstance(device, str) else device
        self._sh_degree = sh_degree
        self._suffix = suffix if suffix is not None else ""

        self._sh_rest_dim = (sh_degree + 1) ** 2 - 1
        self._params_dict = ParameterDict(self._build_parameters())

        # register getter for each parameter
        self._build_parameters_getter()

    @property
    def params_dict(self) -> ParameterDict:
        return self._params_dict

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def sh_rest_dim(self) -> int:
        return self._sh_rest_dim

    @property
    def sh_degree(self) -> int:
        return self._sh_degree

    def _create_getter(self, name):
        def getter(self):
            return self._params_dict[name]

        return getter

    def parameters(self) -> List[Dict[Literal["name", "params"], str | Parameter]]:
        params_group = []
        for k, v in self._params_dict.items():
            params_group.append({"name": k, "params": v})
        return params_group

    def _build_parameters_getter(self) -> None:
        for name in self._params_dict.keys():
            getter_name = f"get_{name.replace(f'{self._suffix}', '')}"
            setattr(self.__class__, getter_name, property(self._create_getter(name)))

    def _build_parameters(self) -> Dict[str, Parameter]:
        raise NotImplementedError("Subclass must implement this method")

    def __len__(self) -> int:
        raise NotImplementedError("Subclass must implement this method")

    def get_std_parameters(self) -> STDGaussiansParams:
        raise NotImplementedError("Subclass must implement this method")

    def get_render_parameters(
        self,
        degeree_to_use: int | None = None,
    ) -> RenderGaussiansParams:
        if degeree_to_use is None:
            degeree_to_use = self._sh_degree
        else:
            degeree_to_use = min(degeree_to_use, self._sh_degree)

        return RenderGaussiansParams.create_from_std_params(
            self.get_std_parameters(), degeree_to_use
        )

    # def switch_requries_grad(self, enable_dict: Dict[str, bool]) -> None:
    #     for k, v in enable_dict.items():
    #         if k in self._params_dict:
    #             self._params_dict[k].requires_grad_(v)

    # def acitive_grad(self, enable: bool) -> None:
    #     for v in self._params_dict.values():
    #         v.requires_grad_(enable)

    def save_std_paramters(self, path: str) -> None:
        params = self.get_std_parameters()
        save_dict = {}

        # raw shape [*, 3]
        xyz = params.means.detach().reshape(-1, 3).cpu().numpy().astype("f4")
        save_dict["x"] = xyz[:, 0]
        save_dict["y"] = xyz[:, 1]
        save_dict["z"] = xyz[:, 2]

        # raw shape [*, 3]
        scales = params.scales.detach().reshape(-1, 3).cpu().numpy().astype("f4")
        save_dict[f"scale_0"] = scales[:, 0]
        save_dict[f"scale_1"] = scales[:, 1]
        save_dict[f"scale_2"] = scales[:, 2]

        # raw shape [*, 4]
        quats = params.quats.detach().reshape(-1, 4).cpu().numpy().astype("f4")
        save_dict[f"rot_{0}"] = quats[:, 0]
        save_dict[f"rot_{1}"] = quats[:, 1]
        save_dict[f"rot_{2}"] = quats[:, 2]
        save_dict[f"rot_{3}"] = quats[:, 3]

        # raw shape [*, 1, 3]
        f_dc = params.features_dc.detach().reshape(-1, 3).cpu().numpy().astype("f4")
        save_dict[f"f_dc_{0}"] = f_dc[:, 0]
        save_dict[f"f_dc_{1}"] = f_dc[:, 1]
        save_dict[f"f_dc_{2}"] = f_dc[:, 2]

        # raw shape [*, sh_rest_dim, 3]
        f_rest = (
            params.features_rest.detach()
            .reshape(-1, self._sh_rest_dim * 3)
            .cpu()
            .numpy()
            .astype("f4")
        )
        for i in range(self._sh_rest_dim * 3):
            save_dict[f"f_rest_{i}"] = f_rest[:, i]

        # raw shape [*, 1]
        opacity = params.opacities.detach().reshape(-1).cpu().numpy().astype("f4")
        save_dict["opacity"] = opacity

        write_ply_file(path, save_dict)

    def prune_parameters(self, optimizer: Optimizer, prune_mask: Tensor) -> None:
        valid_mask = ~prune_mask
        for group in optimizer.param_groups:
            assert len(group["params"]) == 1
            name = group["name"]
            if name not in self._params_dict:
                continue

            stored_state = optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][valid_mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][valid_mask]
                # step 1: delete the old key-value in the optimizer's state
                del optimizer.state[group["params"][0]]
                # step 2: update new gaussian parameter
                self._params_dict[name] = self._params_dict[name][valid_mask]
                # step 3: rewrite the parameter in the optimizer
                group["params"][0] = self._params_dict[name]
                # step 4: rewrite the optimizer's state
                optimizer.state[group["params"][0]] = stored_state
            else:
                self._params_dict[name] = self._params_dict[name][valid_mask]
                group["params"][0] = self._params_dict[name]

    def add_parameters(
        self,
        optimizer: Optimizer,
        additional_params: Dict[str, Union[Tensor, Parameter]],
    ) -> None:
        for group in optimizer.param_groups:
            assert len(group["params"]) == 1
            name = group["name"]
            if name not in self._params_dict:
                continue

            pure_name = name.replace(f"{self._suffix}", "")
            if name in additional_params:
                added_param = Parameter(
                    additional_params[name],
                    requires_grad=group["params"][0].requires_grad,
                )
            elif pure_name in additional_params:
                added_param = Parameter(
                    additional_params[pure_name],
                    requires_grad=group["params"][0].requires_grad,
                )
            else:
                continue

            stored_state = optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(added_param)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(added_param)), dim=0
                )
                # step 1: delete the old key-value in the optimizer's state
                del optimizer.state[group["params"][0]]
                # step 2: update new gaussian parameter
                self._params_dict[name] = torch.cat(
                    (self._params_dict[name], added_param), dim=0
                )
                # step 3: rewrite the parameter in the optimizer
                group["params"][0] = self._params_dict[name]
                # step 4: rewrite the optimizer's state
                optimizer.state[group["params"][0]] = stored_state
            else:
                self._params_dict[name] = torch.cat(
                    (self._params_dict[name], added_param), dim=0
                )
                group["params"][0] = self._params_dict[name]

    def reset_parameters(
        self, optimizer: Optimizer, new_params: Dict[str, Union[Tensor, Parameter]]
    ) -> None:
        for group in optimizer.param_groups:
            name = group["name"]
            if name not in self._params_dict:
                continue

            pure_name = name.replace(f"{self._suffix}", "")
            if name in new_params:
                new_param = Parameter(
                    new_params[name], requires_grad=group["params"][0].requires_grad
                )
            elif pure_name in new_params:
                new_param = Parameter(
                    new_params[pure_name],
                    requires_grad=group["params"][0].requires_grad,
                )
            else:
                continue

            assert len(group["params"]) == 1

            stored_state = optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.zeros_like(new_param)
                stored_state["exp_avg_sq"] = torch.zeros_like(new_param)
                # step 1: delete the old key-value in the optimizer's state
                del optimizer.state[group["params"][0]]
                # step 2: update new gaussian parameter
                self._params_dict[name] = new_param
                # step 3: rewrite the parameter in the optimizer
                group["params"][0] = self._params_dict[name]
                # step 4: rewrite the optimizer's state
                optimizer.state[group["params"][0]] = stored_state
            else:
                self._params_dict[name] = new_param
                group["params"][0] = self._params_dict[name]
