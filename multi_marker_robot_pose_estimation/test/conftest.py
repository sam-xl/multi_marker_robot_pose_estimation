from collections.abc import Sequence
from dataclasses import dataclass, field

import cv2
import numpy as np
import pytest

# aruco_board.create_marker_board calls rclpy.time.Time() after only `import rclpy`, which does
# not load the rclpy.time submodule. The node gets it as a side effect of importing rclpy.node;
# import it here so the tests run under the same conditions as the node.
import rclpy.time  # noqa: F401
import yaml
from geometry_msgs.msg import Quaternion, Transform, TransformStamped, Vector3


@dataclass
class FakeTfBuffer:
    """
    Minimal stand-in for tf2_ros.Buffer.

    `transforms` maps a source frame name to the Transform returned by `lookup_transform`.
    Frames in `failing` are listed by `all_frames_as_yaml` but raise on lookup.
    Frames in `extra_frames` are listed but have no transform (lookup raises).
    """

    transforms: dict[str, Transform] = field(default_factory=dict)
    failing: set[str] = field(default_factory=set)
    extra_frames: Sequence[str] = ()

    def _frames(self) -> list[str]:
        return [*self.transforms, *self.failing, *self.extra_frames]

    def all_frames_as_yaml(self) -> str:
        return yaml.safe_dump({name: {} for name in self._frames()})

    def lookup_transform(self, target, source, time) -> TransformStamped:
        if source in self.failing or source not in self.transforms:
            raise RuntimeError(f"Could not transform {source} to {target}")
        return TransformStamped(transform=self.transforms[source])


def make_transform_msg(
    translation: Sequence[float] = (0.0, 0.0, 0.0),
    quat_xyzw: Sequence[float] = (0.0, 0.0, 0.0, 1.0),
) -> Transform:
    x, y, z = translation
    qx, qy, qz, qw = quat_xyzw
    return Transform(
        translation=Vector3(x=float(x), y=float(y), z=float(z)),
        rotation=Quaternion(x=float(qx), y=float(qy), z=float(qz), w=float(qw)),
    )


def _assert_quat_equal(q1: Sequence[float], q2: Sequence[float], atol: float = 1e-9) -> None:
    """Assert two (x, y, z, w) quaternions represent the same rotation (q == -q)."""
    a = np.asarray(q1, dtype=float)
    b = np.asarray(q2, dtype=float)
    err = min(np.linalg.norm(a - b), np.linalg.norm(a + b))
    assert err <= atol, f"quaternions differ: {a} vs {b} (err={err})"


@pytest.fixture
def make_tf_buffer():
    return FakeTfBuffer


@pytest.fixture
def make_transform():
    return make_transform_msg


@pytest.fixture
def assert_quat_equal():
    return _assert_quat_equal


@pytest.fixture
def camera_intrinsics():
    K = np.array(
        [
            [600.0, 0.0, 320.0],
            [0.0, 600.0, 240.0],
            [0.0, 0.0, 1.0],
        ]
    )
    dist = np.zeros(5)
    return K, dist


@pytest.fixture
def aruco_dict():
    return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
