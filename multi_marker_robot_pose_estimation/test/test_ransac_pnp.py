"""Check that mm_pose_estimator fits the pose to the marker corners that agree on it."""

import unittest

import launch_testing
import numpy as np
import pytest
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from launch import LaunchDescription
from synthetic_camera import (
    CAMERA_HEIGHT,
    SyntheticCamera,
    broadcast_marker_frames,
    camera_tf_publisher,
    estimator_node,
    render_markers,
)

BOARD_MARKERS = [(0, (-0.3, 0.25)), (1, (0.0, 0.25)), (2, (0.3, 0.25)), (3, (-0.3, -0.2))]
FALSE_ID = (4, (0.3, -0.2))
MARKER_FRAMES = [(f"aruco_ref_{i}", center) for i, center in BOARD_MARKERS] + [
    ("aruco_ref_4", (0.0, -0.2)),
]


@pytest.mark.launch_test
def generate_test_description():
    """Launch the estimator with the default min_markers of 3."""
    return LaunchDescription(
        [
            estimator_node(),
            camera_tf_publisher(),
            launch_testing.actions.ReadyToTest(),
        ]
    )


class TestRansacPnp(unittest.TestCase):
    """The base is at (0, 0, CAMERA_HEIGHT) with yaw 0."""

    @classmethod
    def setUpClass(cls):
        """Create the test node, its pose subscription and the synthetic camera."""
        rclpy.init()
        cls.node = rclpy.create_node("test_ransac_pnp")
        cls.poses = []
        cls.node.create_subscription(
            PoseWithCovarianceStamped, "/mm_pose_estimator/pose", cls.poses.append, 10
        )
        cls.camera = SyntheticCamera(cls.node)
        broadcast_marker_frames(cls.node, MARKER_FRAMES)

    @classmethod
    def tearDownClass(cls):
        """Destroy the test node."""
        cls.node.destroy_node()
        rclpy.shutdown()

    def test_marker_with_false_id_is_rejected(self):
        """Marker 4 is rendered 0.3 m from its aruco_ref_4 frame, as if its ID were misread."""
        image = render_markers(0.0, BOARD_MARKERS + [FALSE_ID])
        pose = self.camera.wait_for(image, self.poses).pose.pose
        p = pose.position
        position_error = np.linalg.norm([p.x, p.y, p.z - CAMERA_HEIGHT])
        angle_error_deg = np.degrees(2 * np.arccos(min(1.0, abs(pose.orientation.w))))
        self.assertLess(position_error, 0.01)
        self.assertLess(angle_error_deg, 1.0)

    def test_fewer_than_min_markers_give_no_pose(self):
        """Two markers give no pose; four markers first show that the estimator is running."""
        self.camera.wait_for(render_markers(0.0, BOARD_MARKERS), self.poses)
        with self.assertRaises(AssertionError):
            self.camera.wait_for(render_markers(0.0, BOARD_MARKERS[:2]), self.poses, 3.0)
