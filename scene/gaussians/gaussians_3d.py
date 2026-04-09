from typing import Dict

import torch
from simple_knn._C import distCUDA2
from torch import Tensor
from torch.nn import Parameter

from utils import read_ply_file

from .base_gaussians import BaseGaussians, STDGaussiansParams
from .utils import RGB2SH, generate_random_quats, quaternions_to_matrixes


class Gaussians3D(BaseGaussians):
    def __init__(
        self,
        sh_degree: int = 3,
        device: str | torch.device = "cuda",
        suffix: str | None = None,
    ) -> None:
        super().__init__(sh_degree, device, suffix)

    def _build_parameters(self) -> Dict[str, Parameter]:
        means = torch.empty(0, 3, dtype=torch.float32, device=self._device)
        quats = torch.empty(0, 4, dtype=torch.float32, device=self._device)
        scales = torch.empty(0, 3, dtype=torch.float32, device=self._device)
        opacities = torch.empty(0, 1, dtype=torch.float32, device=self._device)
        features_dc = torch.empty(0, 1, 3, dtype=torch.float32, device=self._device)
        features_rest = torch.empty(
            0, self._sh_rest_dim, 3, dtype=torch.float32, device=self._device
        )

        return {
            f"means{self._suffix}": Parameter(means, requires_grad=True),
            f"quats{self._suffix}": Parameter(quats, requires_grad=True),
            f"scales{self._suffix}": Parameter(scales, requires_grad=True),
            f"opacities{self._suffix}": Parameter(opacities, requires_grad=True),
            f"features_dc{self._suffix}": Parameter(features_dc, requires_grad=True),
            f"features_rest{self._suffix}": Parameter(
                features_rest, requires_grad=True
            ),
        }

    def get_std_parameters(self) -> STDGaussiansParams:
        return STDGaussiansParams(
            means=self._params_dict[f"means{self._suffix}"],
            quats=self._params_dict[f"quats{self._suffix}"],
            scales=self._params_dict[f"scales{self._suffix}"],
            opacities=self._params_dict[f"opacities{self._suffix}"],
            features_dc=self._params_dict[f"features_dc{self._suffix}"],
            features_rest=self._params_dict[f"features_rest{self._suffix}"],
        )

    def init_from_points(
        self,
        points: Tensor,
        colors: Tensor | None = None,
        init_opacity: float = 0.1,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        N = len(points)
        means = points.to(self._device)

        quats = generate_random_quats(N, self._device)

        dists = torch.clamp_min(torch.sqrt(distCUDA2(means)), 0.000001)
        q10_th = torch.quantile(dists, 0.1).item()
        dists = torch.clamp_max(dists, q10_th)
        scales = torch.log(dists).reshape(-1, 1).repeat(1, 3)

        opacities = torch.full(
            [N, 1], init_opacity, dtype=torch.float32, device=self._device
        )
        opacities = torch.logit(opacities)

        if colors is None:
            colors = torch.rand([N, 3], dtype=torch.float32, device=self._device)
        features_dc = RGB2SH(colors)[:, None, :]
        features_rest = torch.zeros(
            [N, self._sh_rest_dim, 3], dtype=torch.float32, device=self._device
        )

        params = {
            f"means{self._suffix}": means,
            f"quats{self._suffix}": quats,
            f"scales{self._suffix}": scales,
            f"opacities{self._suffix}": opacities,
            f"features_dc{self._suffix}": features_dc,
            f"features_rest{self._suffix}": features_rest,
        }

        if optimizer is None:
            for name, value in params.items():
                self._params_dict[name] = Parameter(
                    value,
                    requires_grad=self._params_dict[name].requires_grad,
                )
        else:
            self.reset_parameters(optimizer, params)

    def __len__(self) -> int:
        return len(self._params_dict[f"means{self._suffix}"])

    @property
    def is_empty(self) -> bool:
        return len(self) == 0

    @torch.no_grad()
    def clone(self, optimizer: torch.optim.Optimizer, mask: Tensor) -> None:
        if mask.sum() == 0:
            return

        new_params = {}
        for name, value in self._params_dict.items():
            new_params[name] = value[mask]
        self.add_parameters(optimizer, new_params)

    @torch.no_grad()
    def split(
        self, optimizer: torch.optim.Optimizer, mask: Tensor, n_split: int = 2
    ) -> None:
        if mask.sum() == 0:
            return

        stds = torch.exp(self._params_dict[f"scales{self._suffix}"])[mask].repeat(
            n_split, 1
        )
        means = torch.zeros(len(stds), 3, device=self._device)
        samples = torch.normal(means, stds)
        rots = quaternions_to_matrixes(
            self._params_dict[f"quats{self._suffix}"][mask]
        ).repeat(n_split, 1, 1)
        rotated_samples = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)

        new_means = rotated_samples + self._params_dict[f"means{self._suffix}"][
            mask
        ].repeat(n_split, 1)
        new_scales = torch.log(
            torch.exp(self._params_dict[f"scales{self._suffix}"][mask]).repeat(
                n_split, 1
            )
            / (0.8 * n_split)
        )
        new_quats = self._params_dict[f"quats{self._suffix}"][mask].repeat(n_split, 1)
        new_opacities = self._params_dict[f"opacities{self._suffix}"][mask].repeat(
            n_split, 1
        )
        new_features_dc = self._params_dict[f"features_dc{self._suffix}"][mask].repeat(
            n_split, 1, 1
        )
        new_features_rest = self._params_dict[f"features_rest{self._suffix}"][
            mask
        ].repeat(n_split, 1, 1)

        new_params = {
            f"means{self._suffix}": new_means,
            f"quats{self._suffix}": new_quats,
            f"scales{self._suffix}": new_scales,
            f"opacities{self._suffix}": new_opacities,
            f"features_dc{self._suffix}": new_features_dc,
            f"features_rest{self._suffix}": new_features_rest,
        }

        # 先删除参与分裂的高斯
        self.prune_parameters(optimizer, mask)
        # 再添加分裂得到的新高斯
        self.add_parameters(optimizer, new_params)

    def load_std_parameters(self, path: str) -> None:
        params_dict = read_ply_file(path)

        x = torch.from_numpy(params_dict["x"]).to(self._device)
        y = torch.from_numpy(params_dict["y"]).to(self._device)
        z = torch.from_numpy(params_dict["z"]).to(self._device)
        means = torch.stack([x, y, z], dim=-1)

        scales = []
        for i in range(3):
            scales.append(torch.from_numpy(params_dict[f"scale_{i}"]).to(self._device))
        scales = torch.stack(scales, dim=-1)

        quats = []
        for i in range(4):
            quats.append(torch.from_numpy(params_dict[f"rot_{i}"]).to(self._device))
        quats = torch.stack(quats, dim=-1)

        opacities = torch.from_numpy(params_dict["opacity"]).to(self._device)
        opacities = opacities.reshape(-1, 1)

        features_dc = []
        for i in range(3):
            features_dc.append(
                torch.from_numpy(params_dict[f"f_dc_{i}"]).to(self._device)
            )
        features_dc = torch.stack(features_dc, dim=-1).reshape(-1, 1, 3)

        features_rest = []
        for i in range(self._sh_rest_dim * 3):
            features_rest.append(
                torch.from_numpy(params_dict[f"f_rest_{i}"]).to(self._device)
            )
        features_rest = torch.stack(features_rest, dim=-1).reshape(
            -1, self._sh_rest_dim, 3
        )

        self._params_dict[f"means{self._suffix}"] = Parameter(means, requires_grad=True)
        self._params_dict[f"quats{self._suffix}"] = Parameter(quats, requires_grad=True)
        self._params_dict[f"scales{self._suffix}"] = Parameter(
            scales, requires_grad=True
        )
        self._params_dict[f"opacities{self._suffix}"] = Parameter(
            opacities, requires_grad=True
        )
        self._params_dict[f"features_dc{self._suffix}"] = Parameter(
            features_dc, requires_grad=True
        )
        self._params_dict[f"features_rest{self._suffix}"] = Parameter(
            features_rest, requires_grad=True
        )
