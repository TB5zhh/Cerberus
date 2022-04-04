# %%
import numpy as np
from PIL import Image
import torch

def convert_depth_to_coords(depth):
    unit_mat = torch.zeros((depth.shape[0], depth.shape[1], 3), device=depth.device)
    unit_mat[:, :, 0] = depth.shape[1] // 2
    unit_mat[:, :, 1] = -(torch.as_tensor(range(0, depth.shape[0])).reshape(
        (-1, 1)).repeat([1, depth.shape[1]]) - depth.shape[0] // 2).to(depth.device)
    unit_mat[:, :, 2] = (torch.as_tensor(range(0, depth.shape[1])).reshape(
        (1, -1)).repeat([depth.shape[0], 1]) - depth.shape[1] // 2).to(depth.device)
    # unit_mat /= np.linalg.norm(unit_mat, axis=2, keepdims=True)
    unit_mat /= depth.shape[1] // 2
    return unit_mat * torch.stack([depth for _ in range(3)], dim=2)


def convert_euler_angles_to_matrix(angles):
    z_axis, y_axis, x_axis = torch.as_tensor(angles[0] / 180 * np.pi,
                                             device=angles.device), torch.as_tensor(angles[1] / 180 * np.pi,
                                                                                    device=angles.device), torch.as_tensor(angles[2] / 180 * np.pi,
                                                                                                                           device=angles.device)
    R_Y = torch.as_tensor([
        [torch.cos(y_axis), 0, torch.sin(y_axis)],
        [0, 1, 0],
        [-torch.sin(y_axis), 0, torch.cos(y_axis)],
    ],
                          dtype=torch.float64,
                          device=angles.device)
    R_Z = torch.as_tensor([
        [torch.cos(z_axis), torch.sin(z_axis), 0],
        [-torch.sin(z_axis), torch.cos(z_axis), 0],
        [0, 0, 1],
    ],
                          dtype=torch.float64,
                          device=angles.device)
    R_X = torch.as_tensor([
        [1, 0, 0],
        [0, torch.cos(x_axis), torch.sin(x_axis)],
        [0, -torch.sin(x_axis), torch.cos(x_axis)],
    ],
                          dtype=torch.float64,
                          device=angles.device)
    return R_Y.T, R_Z.T, R_X.T


def project(depth, features, src_pose, dst_pose):

    src_coords = convert_depth_to_coords(depth)
    sry, srz, srx = convert_euler_angles_to_matrix(src_pose[1])
    dry, drz, drx = convert_euler_angles_to_matrix(dst_pose[1])
    src_trans = torch.as_tensor([src_pose[0][0], src_pose[0][2], src_pose[0][1]], device=src_pose.device)
    dst_trans = torch.as_tensor([dst_pose[0][0], dst_pose[0][2], dst_pose[0][1]], device=src_pose.device)
    coords = (srx @ sry @ srz @ src_coords.reshape((-1, 3)).T).T.reshape(src_coords.shape) + src_trans
    dst_coords = (drz.T @ dry.T @ drx.T @ (coords - dst_trans).reshape((-1, 3)).T).T.reshape(coords.shape)
    dst_coords *= (dst_coords.shape[1] // 2) / torch.abs(dst_coords[:, :, 0:1])
    dst_coords[:, :, 1] = dst_coords.shape[0] // 2 - dst_coords[:, :, 1]
    dst_coords[:, :, 2] += dst_coords.shape[1] // 2
    available = torch.logical_and(
        torch.logical_and(torch.logical_and(
            dst_coords[:, :, 1] < dst_coords.shape[0],
            dst_coords[:, :, 1] >= 0,
        ), torch.logical_and(
            dst_coords[:, :, 2] < dst_coords.shape[1],
            dst_coords[:, :, 2] >= 0,
        )),
        dst_coords[:, :, 0] >= 0,
    )

    unique_dst_coords, indices = np.unique(
        np.asarray(dst_coords[available].reshape((-1, 3))[:, 1:].T.cpu()).astype(int),
        axis=1,
        return_index=True,
    )

    unique_dst_img = torch.as_tensor(features[available].reshape((-1, features.shape[-1])))[indices]

    return torch.sparse_coo_tensor(unique_dst_coords, unique_dst_img, features.shape).to_dense()

