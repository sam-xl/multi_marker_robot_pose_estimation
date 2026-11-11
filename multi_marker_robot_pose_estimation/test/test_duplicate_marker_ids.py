"""Check that mm_pose_estimator ignores marker IDs seen twice in the image or the TF."""

import unittest

import launch_testing
import numpy as np
import pytest
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from launch import LaunchDescription
from synthetic_camera import CAMERA_HEIGHT, SyntheticCamera, estimator_node, render_markers

BOARD_MARKERS = [(0, (-0.15, 0.15)), (1, (0.15, 0.15)), (2, (-0.15, -0.15))]
FREE_SPOT = (0.15, -0.15)
MARKER_FRAMES = [(f"aruco_ref_{i}", center) for i, center in BOARD_MARKERS] + [
    ("aruco_ref_3", FREE_SPOT),
    ("cell/aruco_ref_3", (0.6, 0.6)),
]


@pytest.mark.launch_test
def generate_test_description():
    """Launch the estimator."""
    return LaunchDescription([estimator_node(), launch_testing.actions.ReadyToTest()])


class TestDuplicateMarkerIds(unittest.TestCase):
    """The base is at (0, 0, CAMERA_HEIGHT) with yaw 0; the duplicated marker must not move it."""

    @classmethod
    def setUpClass(cls):
        """Create the test node, its pose subscription and the synthetic camera."""
        rclpy.init()
        cls.node = rclpy.create_node("test_duplicate_marker_ids")
        cls.poses = []
        cls.node.create_subscription(
            PoseWithCovarianceStamped, "/mm_pose_estimator/pose", cls.poses.append, 10
        )
        cls.camera = SyntheticCamera(cls.node, MARKER_FRAMES)

    @classmethod
    def tearDownClass(cls):
        """Destroy the test node."""
        cls.node.destroy_node()
        rclpy.shutdown()

    def assert_base_at_truth(self, markers):
        """Render `markers` and check the estimated base pose against the true one."""
        pose = self.camera.wait_for(render_markers(0.0, markers), self.poses).pose.pose
        p = pose.position
        position_error = np.linalg.norm([p.x, p.y, p.z - CAMERA_HEIGHT])
        angle_error_deg = np.degrees(2 * np.arccos(min(1.0, abs(pose.orientation.w))))
        self.assertLess(position_error, 0.01)
        self.assertLess(angle_error_deg, 1.0)

    def test_marker_seen_twice_is_ignored(self):
        """A second copy of marker 1 in the image would pull the fit towards FREE_SPOT."""
        self.assert_base_at_truth(BOARD_MARKERS + [(1, FREE_SPOT)])

    def test_marker_given_by_two_frames_is_ignored(self):
        """Marker 3 is given by aruco_ref_3 and cell/aruco_ref_3, out of view."""
        self.assert_base_at_truth(BOARD_MARKERS + [(3, FREE_SPOT)])
