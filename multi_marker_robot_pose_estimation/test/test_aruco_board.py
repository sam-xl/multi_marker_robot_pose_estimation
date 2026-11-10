import math

import cv2
import numpy as np
import pytest

from multi_marker_robot_pose_estimation.aruco_board import (
    create_marker_board,
    detect_marker_board,
    estimate_camera_pose,
)
from multi_marker_robot_pose_estimation.utils import pose_to_matrix

WORLD = "world"
MARKER_SIZE = 0.2
S45 = math.sin(math.pi / 4)
QUAT_IDENTITY = (0.0, 0.0, 0.0, 1.0)
QUAT_Z90 = (0.0, 0.0, S45, S45)

# Marker corners in OpenCV order TL -> TR -> BR -> BL, for a marker of size 0.2
IDENTITY_CORNERS = np.array(
    [
        [-0.1, 0.1, 0.0],
        [0.1, 0.1, 0.0],
        [0.1, -0.1, 0.0],
        [-0.1, -0.1, 0.0],
    ],
    dtype=np.float32,
)

# Identity corners rotated 90 degrees about Z ((x, y) -> (-y, x)), then shifted by (1, 0, 0)
ROTATED_SHIFTED_CORNERS = np.array(
    [
        [0.9, -0.1, 0.0],
        [0.9, 0.1, 0.0],
        [1.1, 0.1, 0.0],
        [1.1, -0.1, 0.0],
    ],
    dtype=np.float32,
)


def board_ids(board) -> list[int]:
    return [int(i) for i in np.asarray(board.getIds()).ravel()]


def board_corners(board, marker_id: int) -> np.ndarray:
    return np.asarray(board.getObjPoints()[board_ids(board).index(marker_id)])


# T5: create_marker_board ids and corner geometry
def test_create_marker_board_ids_and_corners(make_tf_buffer, make_transform, aruco_dict):
    tf_buffer = make_tf_buffer(
        transforms={
            WORLD: make_transform(),
            "aruco_ref_3": make_transform((0.0, 0.0, 0.0), QUAT_IDENTITY),
            "aruco_ref_7": make_transform((1.0, 0.0, 0.0), QUAT_Z90),
        }
    )

    board = create_marker_board(tf_buffer, WORLD, aruco_dict, MARKER_SIZE)

    assert board is not None
    assert set(board_ids(board)) == {3, 7}

    corners_3 = board_corners(board, 3)
    assert corners_3.dtype == np.float32
    assert corners_3.shape == (4, 3)
    np.testing.assert_allclose(corners_3, IDENTITY_CORNERS, atol=1e-6)

    corners_7 = board_corners(board, 7)
    assert corners_7.dtype == np.float32
    assert corners_7.shape == (4, 3)
    np.testing.assert_allclose(corners_7, ROTATED_SHIFTED_CORNERS, atol=1e-6)


# T6: create_marker_board filtering
def test_create_marker_board_ignores_non_matching_frames(
    make_tf_buffer, make_transform, aruco_dict
):
    # Non-matching frames have valid transforms, so only the name filter can reject them
    tf_buffer = make_tf_buffer(
        transforms={
            "camera_link": make_transform(),
            "aruco_marker_3": make_transform(),
            WORLD: make_transform(),
            "aruco_ref_5": make_transform(),
        }
    )

    board = create_marker_board(tf_buffer, WORLD, aruco_dict, MARKER_SIZE)

    assert board is not None
    assert set(board_ids(board)) == {5}


def test_create_marker_board_skips_failed_lookup(make_tf_buffer, make_transform, aruco_dict):
    tf_buffer = make_tf_buffer(
        transforms={"aruco_ref_7": make_transform()},
        failing={"aruco_ref_3"},
    )

    board = create_marker_board(tf_buffer, WORLD, aruco_dict, MARKER_SIZE)

    assert board is not None
    assert set(board_ids(board)) == {7}


@pytest.mark.parametrize(
    ("transform_frames", "failing"),
    [
        ([], set()),
        (["camera_link", "aruco_marker_3", WORLD], set()),
        ([], {"aruco_ref_3", "aruco_ref_7"}),
    ],
    ids=["empty_yaml", "no_matching_frames", "all_lookups_fail"],
)
def test_create_marker_board_returns_none(
    transform_frames, failing, make_tf_buffer, make_transform, aruco_dict
):
    tf_buffer = make_tf_buffer(
        transforms={name: make_transform() for name in transform_frames},
        failing=failing,
    )
    if not transform_frames and not failing:
        assert tf_buffer.all_frames_as_yaml().strip() == "{}"

    assert create_marker_board(tf_buffer, WORLD, aruco_dict, MARKER_SIZE) is None


# T8: detect_marker_board filters
def square(x0: float, y0: float, w: float, h: float) -> np.ndarray:
    """Marker corners in image order TL -> TR -> BR -> BL (clockwise, y down)."""
    return np.array(
        [[[x0, y0], [x0 + w, y0], [x0 + w, y0 + h], [x0, y0 + h]]],
        dtype=np.float32,
    )


@pytest.fixture
def board_with_marker_3(aruco_dict):
    return cv2.aruco.Board(
        objPoints=[IDENTITY_CORNERS],
        dictionary=aruco_dict,
        ids=np.array([3], dtype=np.int32),
    )


def ids_array(*ids: int) -> np.ndarray:
    return np.array([[i] for i in ids], dtype=np.int32)


@pytest.mark.parametrize(
    ("corners", "ids", "expected_points"),
    [
        ([square(100, 100, 21, 21)], ids_array(3), 4),
        ([square(100, 100, 19, 19)], ids_array(3), None),
        ([square(100, 100, 100, 10)], ids_array(3), None),
        ([square(100, 100, 21, 21)], ids_array(42), None),
        ([square(100, 100, 21, 21), square(300, 300, 19, 19)], ids_array(3, 3), 4),
        ([], None, None),
        ([], np.empty((0, 1), dtype=np.int32), None),
    ],
    ids=[
        "square_21px_kept",
        "square_19px_too_small",
        "rectangle_not_square",
        "id_not_on_board",
        "valid_plus_too_small",
        "ids_none",
        "ids_empty",
    ],
)
def test_detect_marker_board_filters(
    corners, ids, expected_points, board_with_marker_3, camera_intrinsics
):
    K, dist = camera_intrinsics

    result = detect_marker_board(corners, ids, board_with_marker_3, K, dist, None)

    if expected_points is None:
        assert result is None
    else:
        assert result is not None
        obj_points, img_points = result
        assert len(obj_points) == expected_points
        assert len(img_points) == expected_points


# T9: estimate_camera_pose on synthetic data
def rot_x(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_z(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_estimate_camera_pose_synthetic(camera_intrinsics):
    K, dist = camera_intrinsics

    # Three markers on the world XY plane
    centers = [(0.0, 0.0), (0.5, 0.0), (0.0, 0.5)]
    world_points = np.concatenate(
        [IDENTITY_CORNERS.astype(np.float64) + [cx, cy, 0.0] for cx, cy in centers]
    )

    # Camera ~2 m above the plane, looking down (camera Z towards world -Z), slightly tilted
    T_world_camera = np.eye(4)
    T_world_camera[:3, :3] = rot_z(0.3) @ rot_x(math.pi - 0.1)
    T_world_camera[:3, 3] = [0.2, 0.1, 2.0]

    T_camera_world = np.linalg.inv(T_world_camera)
    rvec, _ = cv2.Rodrigues(T_camera_world[:3, :3])
    tvec = T_camera_world[:3, 3]
    img_points, _ = cv2.projectPoints(world_points, rvec, tvec, K, dist)

    obj_points = world_points.reshape(-1, 1, 3).astype(np.float32)
    img_points = img_points.reshape(-1, 1, 2).astype(np.float32)

    result = estimate_camera_pose(obj_points, img_points, K, dist)

    assert result is not None
    M = pose_to_matrix(result.pose)
    np.testing.assert_allclose(M[:3, 3], T_world_camera[:3, 3], atol=1e-4)
    np.testing.assert_allclose(M[:3, :3], T_world_camera[:3, :3], atol=1e-4)
    assert len(result.covariance) == 36
