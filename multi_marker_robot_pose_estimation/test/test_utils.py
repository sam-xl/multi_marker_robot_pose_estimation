import math

import numpy as np
import pytest
from geometry_msgs.msg import Point, Pose, Quaternion

from multi_marker_robot_pose_estimation.utils import (
    H,
    matrix_to_pose,
    pose_to_matrix,
    tf_to_matrix,
)

S45 = math.sin(math.pi / 4)
C45 = math.cos(math.pi / 4)

# 90 degrees about Z, (x, y, z, w)
QUAT_Z90 = (0.0, 0.0, S45, C45)
TRANSLATION = (1.0, 2.0, 3.0)
R_Z90 = np.array(
    [
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
)


def make_pose(position, quat_xyzw) -> Pose:
    x, y, z = position
    qx, qy, qz, qw = quat_xyzw
    return Pose(
        position=Point(x=x, y=y, z=z),
        orientation=Quaternion(x=qx, y=qy, z=qz, w=qw),
    )


def quat_tuple(q: Quaternion) -> tuple[float, float, float, float]:
    return (q.x, q.y, q.z, q.w)


# T1: known-value conversion (catches wxyz vs xyzw mix-ups)
@pytest.mark.parametrize("source", ["pose", "transform"])
def test_known_value_conversion(source, make_transform):
    if source == "pose":
        M = pose_to_matrix(make_pose(TRANSLATION, QUAT_Z90))
    else:
        M = tf_to_matrix(make_transform(TRANSLATION, QUAT_Z90))

    assert M.shape == (4, 4)
    np.testing.assert_allclose(M[:3, :3], R_Z90, atol=1e-12)
    np.testing.assert_allclose(M[:3, 3], TRANSLATION, atol=1e-12)
    np.testing.assert_array_equal(M[3, :], [0.0, 0.0, 0.0, 1.0])


# T2: round trip pose -> matrix -> pose
@pytest.mark.parametrize(
    ("position", "quat_xyzw"),
    [
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        (TRANSLATION, QUAT_Z90),
        ((0.5, -1.0, 2.0), (1.0, 0.0, 0.0, 0.0)),
    ],
    ids=["identity", "rot_z_90", "rot_x_180_offset"],
)
def test_pose_matrix_round_trip(position, quat_xyzw, assert_quat_equal):
    pose = make_pose(position, quat_xyzw)

    result = matrix_to_pose(pose_to_matrix(pose))

    p = result.position
    assert (p.x, p.y, p.z) == pytest.approx(position, abs=1e-12)
    assert_quat_equal(quat_tuple(result.orientation), quat_xyzw)


# T3: H with a solvePnP-shaped tvec
R_FIXED = np.array(
    [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
)
TVEC = np.array([[0.1], [-0.2], [1.5]])


@pytest.mark.parametrize(
    "tvec",
    [TVEC, TVEC.ravel(), TVEC.ravel().tolist()],
    ids=["array_3x1", "array_3", "list"],
)
def test_H_translation_shapes(tvec):
    M = H(R_FIXED, tvec)

    assert M.shape == (4, 4)
    np.testing.assert_array_equal(M[:3, :3], R_FIXED)
    np.testing.assert_array_equal(M[:3, 3], TVEC.ravel())
    np.testing.assert_array_equal(M[3, :], [0.0, 0.0, 0.0, 1.0])
    np.testing.assert_array_equal(M, H(R_FIXED, TVEC))
