"""Functions to deal with ArUco board markers."""

import re
from math import sqrt

import cv2
import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseWithCovariance

from multi_marker_robot_pose_estimation.utils import H, matrix_to_pose, tf_to_matrix


def create_marker_board(tf_buffer, world_frame_id, aruco_dict, marker_size) -> cv2.aruco.Board:
    """
    Create Aruco board from published TF references.

    All TF frames matching aruco_ref_<id> are used. The marker coordinates
    will be the transform between the marker frame and world_frame_id.
    The corners are calculated using marker_size, in the XY plane of the
    marker frame.
    """
    obj_points = []
    ids = []

    half_size = marker_size / 2.0

    frames_yaml = yaml.safe_load(tf_buffer.all_frames_as_yaml())
    if len(frames_yaml) == 0:
        return None

    frames = frames_yaml.keys()

    for frame in frames:
        match = re.search(r".*aruco_ref_(\d+)", frame)
        if not match:
            continue

        marker_id = int(match.group(1))

        try:
            tf = tf_buffer.lookup_transform(
                world_frame_id,
                frame,
                rclpy.time.Time(),
            )
        except Exception:
            continue

        # Transform from marker frame to world frame
        Hmw = tf_to_matrix(tf.transform)

        # Marker corners in marker-local coordinates
        local_corners = np.array(
            [
                [-half_size, half_size, 0.0, 1.0],
                [half_size, half_size, 0.0, 1.0],
                [half_size, -half_size, 0.0, 1.0],
                [-half_size, -half_size, 0.0, 1.0],
            ]
        )

        # Transform corners into board/world coordinates
        world_corners = Hmw @ local_corners.T

        obj_points.append(world_corners[0:3, :].T.astype(np.float32))
        ids.append(marker_id)

    if len(ids) == 0:
        return None

    board = cv2.aruco.Board(
        objPoints=obj_points,
        dictionary=aruco_dict,
        ids=np.array(ids, dtype=np.int32),
    )

    return board


def detect_marker_board(
    corners, ids, board, matrix_coefficients, distortion_coefficients, frame
) -> "tuple[np.ndarray, np.ndarray]|None":
    """Detect marker board."""
    if ids is None or len(ids) == 0:
        return None

    # Remove small and acute markers
    newcorners = []
    newids = []
    for corner, _id in zip(corners, ids):
        x = corner[0, :, 0]
        y = corner[0, :, 1]
        area = (
            x[0] * y[1]
            + x[1] * y[2]
            + x[2] * y[3]
            + x[3] * y[0]
            - x[1] * y[0]
            - x[2] * y[1]
            - x[3] * y[2]
            - x[0] * y[3]
        ) / 2

        # Remove small markers
        if area < 400:
            continue
        perimeter = (
            sqrt((x[1] - x[0]) ** 2 + (y[1] - y[0]) ** 2)
            + sqrt((x[2] - x[1]) ** 2 + (y[2] - y[1]) ** 2)
            + sqrt((x[3] - x[2]) ** 2 + (y[3] - y[2]) ** 2)
            + sqrt((x[0] - x[3]) ** 2 + (y[0] - y[3]) ** 2)
        )
        squareness = 4 * sqrt(area) / perimeter

        # Remove acute markers
        if squareness < 0.8:
            continue

        newcorners.append(corner)
        newids.append(_id)
    newids = np.array(newids)

    if newids is None or newids.size == 0:
        return None

    objPoints, imgPoints = board.matchImagePoints(newcorners, newids)

    if objPoints is None:
        return None

    detections = len(objPoints) / 4
    if detections == 0:
        return None

    if frame is not None:
        # Show debug information superimposed on camera image
        cv2.aruco.drawDetectedMarkers(frame, newcorners, newids)

    return objPoints, imgPoints


def estimate_board_pose(
    objPoints, imgPoints, matrix_coefficients, distortion_coefficients, frame=None
) -> PoseWithCovariance:
    """Estimate board pose from detected markers."""
    pose = PoseWithCovariance()

    retval, rvec, tvec = cv2.solvePnP(
        objPoints,
        imgPoints,
        matrix_coefficients,
        distortion_coefficients,
        flags=cv2.SOLVEPNP_SQPNP,
    )

    if not retval:
        return None

    R, _ = cv2.Rodrigues(rvec)
    Hwc = H(R, tvec)
    pose.pose = matrix_to_pose(Hwc)

    # Project object points back onto the image, both for the reprojection error
    # and to get the image Jacobian d(uv)/d(object_pose)
    projPoints, jacobian = cv2.projectPoints(
        objPoints, rvec, tvec, matrix_coefficients, distortion_coefficients
    )

    # Calculate reprojection error
    diff = (imgPoints - projPoints).reshape(-1, 2)
    errors = np.linalg.norm(diff, axis=1)
    rms_image = np.sqrt(np.mean(errors**2))

    # Calculate pose covariance from image jacobian.
    # https://stackoverflow.com/questions/36618269/uncertainty-on-pose-estimate-when-minimizing-measurement-errors
    J_pose = jacobian[:, [3, 4, 5, 0, 1, 2]]
    J_T_J = np.dot(J_pose.T, J_pose)
    cov_pose = np.linalg.pinv(J_T_J) * (rms_image**2)

    pose.covariance = cov_pose.flatten()

    if frame is not None:
        # Show debug information superimposed on camera image
        cv2.drawFrameAxes(
            image=frame,
            cameraMatrix=matrix_coefficients,
            distCoeffs=distortion_coefficients,
            rvec=rvec,
            tvec=tvec,
            length=0.1,
            thickness=3,
        )

        for p in projPoints:
            cv2.circle(
                frame, (int(p[0][0]), int(p[0][1])), radius=5, color=(255, 255, 255), thickness=2
            )

    return pose


def estimate_camera_pose(
    objPoints, imgPoints, matrix_coefficients, distortion_coefficients, frame=None
) -> PoseWithCovariance:
    """Estimate camera pose from detected markers."""
    pose = PoseWithCovariance()

    retval, rvec, tvec = cv2.solvePnP(
        objPoints,
        imgPoints,
        matrix_coefficients,
        distortion_coefficients,
        flags=cv2.SOLVEPNP_SQPNP,
    )

    if not retval:
        return None

    R, _ = cv2.Rodrigues(rvec)
    Hwc = H(R, tvec)
    Hcw = np.linalg.inv(Hwc)
    pose.pose = matrix_to_pose(Hcw)

    # Convert object points from world to camera frame. Then, the object frame is the camera frame.
    objPoints1 = np.hstack((objPoints.reshape(-1, 3), np.ones((objPoints.shape[0], 1))))
    objPointsCamera1 = (Hwc @ objPoints1.T).T
    objPointsCamera = objPointsCamera1[:, np.newaxis, 0:3]

    # Project transformed object points back onto the image, both for the reprojection error
    # and to get the image Jacobian d(uv)/d(camera_pose)
    projPoints, jacobian = cv2.projectPoints(
        objPointsCamera,
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0]),
        matrix_coefficients,
        distortion_coefficients,
    )

    # Calculate reprojection error
    diff = (imgPoints - projPoints).reshape(-1, 2)
    errors = np.linalg.norm(diff, axis=1)
    rms_image = np.sqrt(np.mean(errors**2))

    # Calculate pose covariance from image jacobian.
    # https://stackoverflow.com/questions/36618269/uncertainty-on-pose-estimate-when-minimizing-measurement-errors
    J_pose = jacobian[:, [3, 4, 5, 0, 1, 2]]
    J_T_J = np.dot(J_pose.T, J_pose)
    cov_pose = np.linalg.pinv(J_T_J) * (rms_image**2)

    # Convert pose covariance to world frame.
    # We only need to rotate it, as there is no lever arm effect
    # (where the world is does not impact how uncertain you are about your position,
    # just how you express it)
    R6 = np.zeros((6, 6))
    R6[:3, :3] = R.T
    R6[3:, 3:] = R.T
    cov_world = R6 @ cov_pose @ R6.T

    pose.covariance = cov_world.flatten()

    if frame is not None:
        # Show debug information superimposed on camera image
        cv2.drawFrameAxes(
            image=frame,
            cameraMatrix=matrix_coefficients,
            distCoeffs=distortion_coefficients,
            rvec=rvec,
            tvec=tvec,
            length=0.1,
            thickness=3,
        )

        for p in projPoints:
            cv2.circle(
                frame, (int(p[0][0]), int(p[0][1])), radius=5, color=(255, 255, 255), thickness=2
            )

    return pose
