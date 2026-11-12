"""Utility functions for pose estimation."""

import numpy as np
from geometry_msgs.msg import Point, Pose, PoseWithCovariance, Quaternion, Transform, Vector3
from transforms3d.quaternions import mat2quat, quat2mat


def HT(
    x: float | list[float] | tuple[float] | np.ndarray = 0.0, y: float = 0.0, z: float = 0.0
) -> np.ndarray:
    """
    Homogeneous translation matrix.

    Args:
        x, y, z (float): the translation components.
        OR
        x (np.ndarray): the translation vector.

    Returns:
        np.ndarray: the (4, 4) transformation matrix.
    """
    if isinstance(x, (list, tuple, np.ndarray)):
        x, y, z = x

    return np.asarray(
        [[1.0, 0.0, 0.0, x], [0.0, 1.0, 0.0, y], [0.0, 0.0, 1.0, z], [0.0, 0.0, 0.0, 1.0]]
    )


def H(R: np.ndarray | None = None, T: np.ndarray | list[float] | None = None) -> np.ndarray:
    """
    Homogenous transformation matrix.

    Args:
        R (np.ndarray): the (3, 3) rotation matrix.
        T (np.ndarray): the (3,) translation vector or (4, 4) translation matrix.

    Returns:
        np.ndarray: the (4, 4) transformation matrix.
    """
    if R is None:
        R = np.eye(3)
    if T is None:
        T = np.eye(4)

    T = np.asarray(T)
    if T.shape != (4, 4):
        T = T.flatten()
        T = HT(T[0], T[1], T[2])

    M = np.copy(T)
    M[0:3, 0:3] = R
    return M


def _to_matrix(translation: Point | Vector3, rotation: Quaternion) -> np.ndarray:
    """
    Convert a translation/rotation pair to a transformation matrix.

    Args:
        translation: a point or vector in 3D space.
        rotation: a quaternion.

    Returns:
        np.ndarray: the transformation matrix.
    """
    return H(
        quat2mat([rotation.w, rotation.x, rotation.y, rotation.z]),
        [translation.x, translation.y, translation.z],
    )


def tf_to_matrix(transform: Transform) -> np.ndarray:
    """
    Convert a transform to a transformation matrix.

    Args:
        transform (Transform): the transform to convert.

    Returns:
        np.ndarray: the transformation matrix.
    """
    return _to_matrix(transform.translation, transform.rotation)


def matrix_to_tf(matrix: np.ndarray) -> Transform:
    """
    Convert a transformation matrix to a transform .

    Args:
        matrix (np.ndarray): the transformation matrix to convert.

    Returns:
        Transform: the equivalent transform.
    """
    qw, qx, qy, qz = mat2quat(matrix[0:3, 0:3])
    tx, ty, tz = matrix[0:3, 3]
    return Transform(
        translation=Vector3(x=tx, y=ty, z=tz), rotation=Quaternion(x=qx, y=qy, z=qz, w=qw)
    )


def pose_to_matrix(pose: Pose) -> np.ndarray:
    """
    Convert a pose to a transformation matrix.

    Args:
        pose (Pose): the pose to convert.

    Returns:
        np.ndarray: the transformation matrix.
    """
    return _to_matrix(pose.position, pose.orientation)


def matrix_to_pose(matrix: np.ndarray) -> Pose:
    """
    Convert a transformation matrix to a pose.

    Args:
        matrix (np.ndarray): the transformation matrix to convert.

    Returns:
        Pose: the equivalent pose.
    """
    qw, qx, qy, qz = mat2quat(matrix[0:3, 0:3])
    tx, ty, tz = matrix[0:3, 3]
    return Pose(position=Point(x=tx, y=ty, z=tz), orientation=Quaternion(x=qx, y=qy, z=qz, w=qw))


def skew(v) -> np.ndarray:
    """Skew-symmetric matrix [v]x such that [v]x @ u = v x u."""
    x, y, z = v
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ]
    )


def invert_pose(pose: Pose) -> Pose:
    """
    Return the inverse of a pose.

    Args:
        pose (geometry_msgs.msg.Pose): the pose to invert.

    Returns:
        geometry_msgs.msg.Pose: the inverted pose.
    """
    # object->world transform matrix
    How = pose_to_matrix(pose)

    # Invert transform
    Hwo = np.linalg.inv(How)

    # Convert back to Pose
    return matrix_to_pose(Hwo)


def invert_pose_with_covariance(pose_cov: PoseWithCovariance) -> PoseWithCovariance:
    """
    Return the inverse of a pose with covariance.

    Covariance matrices are expressed in parent frame (left side).
    Uses a first order approximation for the covariance propagation.

    Args:
        pose (geometry_msgs.msg.PoseWithCovariance): the pose to invert.

    Returns:
        geometry_msgs.msg.PoseWithCovariance: the inverted pose.
    """
    # object->world transform matrix
    How = pose_to_matrix(pose_cov.pose)

    # Invert transform
    Hwo = np.linalg.inv(How)
    Rwo = Hwo[0:3, 0:3]
    two = Hwo[0:3, 3]

    # Convert back to PoseWithCovariance
    result = PoseWithCovariance()
    result.pose = matrix_to_pose(Hwo)

    # Adjoint of the inverse mapping
    # https://arxiv.org/pdf/1812.01537, p. 7
    Awo = np.zeros((6, 6))
    Awo[:3, :3] = Rwo  # d(p')/dp  = R'
    Awo[:3, 3:] = skew(two) @ Rwo  # d(p')/dr  = [p']x @ R'
    Awo[3:, 3:] = Rwo  # d(r')/dr  = R'

    # Invert covariance, first order approximation
    # https://gtsam.org/2021/02/23/uncertainties-part3.html, eq (11)
    cov = np.array(pose_cov.covariance, dtype=float).reshape(6, 6)
    cov_inv = Awo @ cov @ Awo.T

    result.covariance = cov_inv.flatten()

    return result
