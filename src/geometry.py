import numpy as np
import transforms3d as t3d # see(https://matthew-brett.github.io/transforms3d/reference/transforms3d.affines.html)

from .sensors import Lidar
from .sensors import Camera


def _heading_position_to_mat(heading, position):
    quat = np.array([heading["w"], heading["x"], heading["y"], heading["z"]])
    pos = np.array([position["x"], position["y"], position["z"]])
    # Compose translations(移動), rotations, zooms, [shears] to affine
    transform_matrix = t3d.affines.compose(np.array(pos),
                                           t3d.quaternions.quat2mat(quat), # Calculate rotation matrix corresponding to quaternion
                                           [1.0, 1.0, 1.0])
    return transform_matrix  # return Affine transformation matrix


def projection(lidar_points, camera_data, camera_pose, camera_intrinsics, filter_outliers=True):
    """
    Args:
        - lidar_points:
          - np.array([N, 3]): lidar points in the world coordinates.
        - camera_data:
          - PIL.Image: image for one camera in one frame.
        - camera_pose:
          - pose in the world coordinates for one camera in one frame.
        - camera_intrinsics:
          - intrinsics for one camera in one frame.
        - filter_outliers:
          - (bool): filtering projected 2d-points out of image.
    Returns:
        - projection_points2d:
          - np.array([K, 2]): projected 2d-points in pixels.
        - camera_points_3d:
          - np.array([K, 3]): 3d-points in pixels in the camera frame.
        - inliner_idx:
          - np.array([K, 2]): the indices for lidar_points whose projected 2d-points are inside image.
    """
    camera_heading = camera_pose['heading']
    camera_position = camera_pose['position']
    camera_pose_mat = _heading_position_to_mat(camera_heading, camera_position)

    trans_lidar_to_camera = np.linalg.inv(camera_pose_mat) # 外部パラメータ行列
    points3d_lidar = lidar_points # ワールド座標
    points3d_camera = trans_lidar_to_camera[:3, :3] @ (points3d_lidar.T) + trans_lidar_to_camera[:3, 3].reshape(3, 1)

    # 内部パラメータ行列
    K = np.eye(3, dtype=np.float64)
    K[0, 0] = camera_intrinsics.fx
    K[1, 1] = camera_intrinsics.fy
    K[0, 2] = camera_intrinsics.cx
    K[1, 2] = camera_intrinsics.cy

    inliner_indices_arr = np.arange(points3d_camera.shape[1])
    if filter_outliers:
        condition = points3d_camera[2, :] > 0.0
        points3d_camera = points3d_camera[:, condition]
        inliner_indices_arr = inliner_indices_arr[condition]

    points2d_camera = K @ points3d_camera # カメラ座標 -> 画像座標
    points2d_camera = (points2d_camera[:2, :] / points2d_camera[2, :]).T # 同次座標?

    if filter_outliers:
        image_w, image_h = camera_data.size
        condition = np.logical_and(
            (points2d_camera[:, 1] < image_h) & (points2d_camera[:, 1] > 0),
            (points2d_camera[:, 0] < image_w) & (points2d_camera[:, 0] > 0))
        points2d_camera = points2d_camera[condition]
        points3d_camera = (points3d_camera.T)[condition]
        inliner_indices_arr = inliner_indices_arr[condition]
    return points2d_camera, points3d_camera, inliner_indices_arr


def lidar_points_to_ego(points, lidar_pose):
    lidar_pose_mat = _heading_position_to_mat(
        lidar_pose['heading'], lidar_pose['position'])
    transform_matrix = np.linalg.inv(lidar_pose_mat)
    return (transform_matrix[:3, :3] @ points.T + transform_matrix[:3, [3]]).T


def center_box_to_corners(box):
    pos_x, pos_y, pos_z, dim_x, dim_y, dim_z, yaw = box
    half_dim_x, half_dim_y, half_dim_z = dim_x/2.0, dim_y/2.0, dim_z/2.0
    corners = np.array([[half_dim_x, half_dim_y, -half_dim_z],
                        [half_dim_x, -half_dim_y, -half_dim_z],
                        [-half_dim_x, -half_dim_y, -half_dim_z],
                        [-half_dim_x, half_dim_y, -half_dim_z],
                        [half_dim_x, half_dim_y, half_dim_z],
                        [half_dim_x, -half_dim_y, half_dim_z],
                        [-half_dim_x, -half_dim_y, half_dim_z],
                        [-half_dim_x, half_dim_y, half_dim_z]])
    transform_matrix = np.array([
        [np.cos(yaw), -np.sin(yaw), 0, pos_x],
        [np.sin(yaw), np.cos(yaw), 0, pos_y],
        [0, 0, 1.0, pos_z],
        [0, 0, 0, 1.0],
    ])
    corners = (transform_matrix[:3, :3] @
               corners.T + transform_matrix[:3, [3]]).T
    return corners


if __name__ == '__main__':
    pass
