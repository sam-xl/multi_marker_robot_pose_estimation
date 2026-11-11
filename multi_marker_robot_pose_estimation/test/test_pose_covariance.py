"""Check the camera pose covariance of estimate_camera_pose against Monte Carlo pose errors."""

import cv2
import numpy as np
import pytest
from synthetic_camera import MARKER_SIZE, K, rotation_x

from multi_marker_robot_pose_estimation.aruco_board import estimate_camera_pose
from multi_marker_robot_pose_estimation.utils import pose_to_matrix

PIXEL_NOISE = 0.3
TRIALS = 2000
CAMERA_DISTANCE = 1.0
TILT = np.radians(30.0)


def marker_corners(cx, cy):
    """Return the corners of the marker centred at (cx, cy) on z = 0, as detected by OpenCV."""
    half = MARKER_SIZE / 2
    return [
        [cx - half, cy + half, 0.0],
        [cx + half, cy + half, 0.0],
        [cx + half, cy - half, 0.0],
        [cx - half, cy - half, 0.0],
    ]


def pose_errors_and_covariances(marker_centers):
    """
    Estimate the camera pose from noisy projections of the markers TRIALS times.

    The camera is CAMERA_DISTANCE from the world origin and looks at it, tilted by TILT about x
    to avoid the flip ambiguity of a fronto-parallel square. Return the errors
    [position, rotation vector], both in the world frame, and the covariances.
    """
    camera_in_world = rotation_x(np.pi - TILT)
    camera_position = camera_in_world @ np.array([0.0, 0.0, -CAMERA_DISTANCE])
    rvec = cv2.Rodrigues(camera_in_world.T)[0]
    tvec = -camera_in_world.T @ camera_position

    obj_points = np.array(
        [corner for center in marker_centers for corner in marker_corners(*center)],
        dtype=np.float32,
    )[:, np.newaxis, :]
    true_img_points = cv2.projectPoints(obj_points, rvec, tvec, K, None)[0]

    rng = np.random.default_rng(0)
    errors, covariances = [], []
    for _ in range(TRIALS):
        img_points = true_img_points + rng.normal(0.0, PIXEL_NOISE, true_img_points.shape)
        pose = estimate_camera_pose(
            obj_points, img_points.astype(np.float32), K, np.zeros(5), sigma_min=0.0
        )
        H = pose_to_matrix(pose.pose)
        rotation_error = cv2.Rodrigues(H[:3, :3] @ camera_in_world.T)[0].ravel()
        errors.append(np.concatenate((H[:3, 3] - camera_position, rotation_error)))
        covariances.append(np.reshape(pose.covariance, (6, 6)))
    return np.array(errors), np.array(covariances)


@pytest.mark.parametrize(
    "marker_centers",
    [[(0.0, 0.0)], [(-0.15, 0.15), (0.15, 0.15), (0.15, -0.15)]],
    ids=["one_marker", "three_markers"],
)
def test_covariance_matches_pose_errors(marker_centers):
    """
    The mean covariance matches the covariance of the pose errors.

    Per-corner σ² over N corners would give 0.5 times the error variance for one marker and
    1.5 times for three.
    """
    errors, covariances = pose_errors_and_covariances(marker_centers)
    error_covariance = np.cov(errors, rowvar=False)
    ratio = np.diag(covariances.mean(axis=0)) / np.diag(error_covariance)
    np.testing.assert_allclose(ratio, 1.0, atol=0.15)
