"""Utility functions for pose estimation."""

from math import cos, sin

import numpy as np
from geometry_msgs.msg import Point, Pose, PoseWithCovariance, Quaternion, Transform, Vector3
from transforms3d.quaternions import mat2quat, quat2mat


def RX(theta: float) -> np.ndarray:
    """
    3D rotation matrix about the X axis.

    Args:
        theta (float): the rotation angle.

    Returns:
        np.ndarray: the (3, 3) rotation matrix.
    """
    return np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos(theta), -sin(theta)],
            [0.0, sin(theta), cos(theta)],
        ],
        dtype=np.float64,
    )


def RY(theta: float) -> np.ndarray:
    """
    3D rotation matrix about the Y axis.

    Args:
        theta (float): the rotation angle.

    Returns:
        np.ndarray: the (3, 3) rotation matrix.
    """
    return np.asarray(
        [
            [cos(theta), 0.0, sin(theta)],
            [0.0, 1.0, 0.0],
            [-sin(theta), 0.0, cos(theta)],
        ],
        dtype=np.float64,
    )


def RZ(theta: float) -> np.ndarray:
    """
    3D rotation matrix about the Z axis.

    Args:
        theta (float): the rotation angle.

    Returns:
        np.ndarray: the (3, 3) rotation matrix.
    """
    return np.asarray(
        [
            [cos(theta), -sin(theta), 0.0],
            [sin(theta), cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


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


def HX(theta: float) -> np.ndarray:
    """
    Homogenous transformation matrix of rotation around X axis.

    Args:
        theta (float): the rotation angle.

    Returns:
        np.ndarray: the (4, 4) transformation matrix.
    """
    return H(RX(theta))


def HY(theta: float) -> np.ndarray:
    """
    Homogenous transformation matrix of rotation around Y axis.

    Args:
        theta (float): the rotation angle.

    Returns:
        np.ndarray: the (4, 4) transformation matrix.
    """
    return H(RY(theta))


def HZ(theta: float) -> np.ndarray:
    """
    Homogenous transformation matrix of rotation around Z axis.

    Args:
        theta (float): the rotation angle.

    Returns:
        np.ndarray: the (4, 4) transformation matrix.
    """
    return H(RZ(theta))


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


def find_rotation(R: np.ndarray, axis: str) -> float:
    """
    Find rotation that aligns the specified axis.

    Find theta such that R_axis(theta) @ R puts the two coordinate axes other
    than `axis` into the planes containing `axis`:
      axis='x': Y axis in XY plane (Z comp = 0), Z axis in XZ plane (Y comp = 0)
      axis='y': Z axis in YZ plane (X comp = 0), X axis in XY plane (Z comp = 0)
      axis='z': X axis in XZ plane (Y comp = 0), Y axis in YZ plane (X comp = 0)

    Returns the least-squares theta when the two conditions conflict.
    Result is unique only mod pi. We return the smallest angle.
    """
    R = np.asarray(R, dtype=float)
    a = {"x": 0, "y": 1, "z": 2}[axis]
    p = (a + 1) % 3  # first remaining axis (cyclic order x -> y -> z -> x)
    q = (a + 2) % 3  # second remaining axis

    # Multiplying R' = R_a @ R and extracting the elements we want to be zero, we find
    #   R'[q, p] = cos*R[q, p] + sin*R[p, p] = 0
    #   R'[p, q] = cos*R[p, q] - sin*R[q, q] = 0

    # M @ [cos, sin] = 0, with ||[cos, sin]|| = 1.
    M = np.array([[R[q, p], R[p, p]], [R[p, q], -R[q, q]]])

    # Least-squares: minimize [c, s] @ S @ [c, s] over the unit circle ->
    # the eigenvector of S belonging to its smallest eigenvalue.
    S = M.T @ M
    _, eigvecs = np.linalg.eigh(S)
    c, s = eigvecs[:, 0]
    theta = np.arctan2(s, c)

    # Find minimal angle
    if theta < -0.5 * np.pi:
        theta += np.pi
    if theta > 0.5 * np.pi:
        theta -= np.pi

    return theta


def align(frame1: np.ndarray, frame2: np.ndarray, commands: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Aligns two frames using a sequence of commands.

    Commands in UPPERCASE align frame2 with frame1
    Commands in lowercase align frame1 with frame2.

    Assumes both frames are given in a common base frame.

    Commands:
        x: Project to YZ plane
        y: Project to XZ plane
        z: Project to XY plane
        r: Align roll
        p: Align pitch
        w: Align yaw

    Returns the aligned frames.
    """
    for command in commands:
        if command.isupper():
            # Work on frame2 in terms of frame1
            frame = np.linalg.inv(frame1) @ frame2
            frame_id = 2
        else:
            # Work on frame1 in terms of frame2
            frame = np.linalg.inv(frame2) @ frame1
            frame_id = 1

        command = command.lower()

        if command == "x":
            # Project to YZ plane
            frame[0][3] = 0.0
        if command == "y":
            # Project to XZ plane
            frame[1][3] = 0.0
        if command == "z":
            # Project to XY plane
            frame[2][3] = 0.0
        if command == "r":
            # Align roll
            theta = find_rotation(frame[0:3, 0:3], "x")
            frame[0:3, 0:3] = RX(theta) @ frame[0:3, 0:3]
        if command == "p":
            # Align pitch
            theta = find_rotation(frame[0:3, 0:3], "y")
            frame[0:3, 0:3] = RY(theta) @ frame[0:3, 0:3]
        if command == "w":
            # Align yaw
            theta = find_rotation(frame[0:3, 0:3], "z")
            frame[0:3, 0:3] = RZ(theta) @ frame[0:3, 0:3]

        if frame_id == 1:
            frame1 = frame2 @ frame
        else:
            frame2 = frame1 @ frame

    return frame1, frame2
