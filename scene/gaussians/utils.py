import pypose as pp
import torch
from torch import Tensor


def RGB2SH(rgb: Tensor) -> Tensor:
    """
    Converts from RGB values [0,1] to the 0th spherical harmonic coefficient
    """
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def SH2RGB(sh: Tensor) -> Tensor:
    """
    Converts from the 0th spherical harmonic coefficient to RGB values [0,1]
    """
    C0 = 0.28209479177387814
    return sh * C0 + 0.5


def generate_random_quats(N: int, device: torch.device) -> Tensor:
    """
    Defines a random quaternion tensor of shape (N, 4)
    """
    u = torch.rand(N, device=device)
    v = torch.rand(N, device=device)
    w = torch.rand(N, device=device)

    quats = torch.stack(
        [
            torch.sqrt(1 - u) * torch.sin(2 * torch.pi * v),
            torch.sqrt(1 - u) * torch.cos(2 * torch.pi * v),
            torch.sqrt(u) * torch.sin(2 * torch.pi * w),
            torch.sqrt(u) * torch.cos(2 * torch.pi * w),
        ],
        dim=-1,
    )

    return quats


def generate_quats_from_normals(normals: Tensor) -> Tensor:
    normals = normals / torch.norm(normals, dim=-1, keepdim=True)
    # plane: Ax + By + Cz = 0
    A, B, C = normals[..., 0], normals[..., 1], normals[..., 2]
    # if A==B==0, it is invalid
    ref = torch.where(
        torch.abs(C[:, None].expand(-1, 3)) >= 0.95,
        torch.stack([-C, torch.zeros_like(A), A], dim=-1),
        torch.stack([-B, A, torch.zeros_like(A)], dim=-1),
    )

    z_dirs = normals
    x_dirs = torch.cross(ref, z_dirs, dim=-1)
    y_dirs = torch.cross(z_dirs, x_dirs, dim=-1)
    x_dirs = x_dirs / torch.norm(x_dirs, dim=-1, keepdim=True)
    y_dirs = y_dirs / torch.norm(y_dirs, dim=-1, keepdim=True)

    matrixes = torch.stack([x_dirs, y_dirs, z_dirs], dim=-1)
    quats = matrixes_to_quaternions(matrixes)
    return quats


def matrixes_to_quaternions(rotation_matrix: Tensor) -> Tensor:
    """
    Converts rotation matrices to quaternions
    Args:
        rotation_matrix (Tensor): Rotation matrix tensor of shape (N, 3, 3)

    Returns:
        Tensor: Quaternion tensor of shape (N, 4)
    """
    quats = pp.mat2SO3(rotation_matrix, check=False).tensor()
    x, y, z, w = quats[..., 0], quats[..., 1], quats[..., 2], quats[..., 3]
    quats = torch.stack([w, x, y, z], dim=-1)
    return quats


def quaternions_to_matrixes(qvec: Tensor) -> Tensor:
    """
    Converts quaternions to rotation matrices
    Args:
        qvec (Tensor): Quaternion tensor of shape (N, 4)

    Returns:
        Tensor: Rotation matrix tensor of shape (N, 3, 3)
    """
    qvec = qvec / torch.norm(qvec, dim=-1, keepdim=True)  # Normalize quaternions
    w, x, y, z = qvec[..., 0], qvec[..., 1], qvec[..., 2], qvec[..., 3]
    qvec = torch.stack([x, y, z, w], dim=-1)
    rotation_matrix = pp.SO3(qvec).matrix()
    return rotation_matrix
