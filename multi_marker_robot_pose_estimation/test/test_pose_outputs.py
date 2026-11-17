"""Check the debug image and the smoothed TF of mm_pose_estimator."""

import unittest

import numpy as np
import pytest
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from launch import LaunchDescription
from launch_testing.actions import ReadyToTest
from sensor_msgs.msg import Image
from synthetic_camera import SyntheticCamera, estimator_node, render_markers, scene_tf_publishers
from tf2_ros import Buffer, TransformListener

MARKERS = [(0, (-0.3, 0.25)), (1, (0.0, 0.25)), (2, (0.3, 0.25)), (3, (-0.3, -0.2))]
MARKER_FRAMES = [(f"aruco_ref_{i}", center) for i, center in MARKERS]
TF_ALPHA = 0.25


@pytest.mark.launch_test
def generate_test_description():
    """Launch the estimator with the debug image and the smoothed TF."""
    return LaunchDescription(
        [
            estimator_node(debug=True, tf_frame_id="base_tf", tf_alpha=TF_ALPHA),
            *scene_tf_publishers(MARKER_FRAMES),
            ReadyToTest(),
        ]
    )


class TestPoseOutputs(unittest.TestCase):
    """The base is at (0, 0, CAMERA_HEIGHT) with the yaw each test renders."""

    @classmethod
    def setUpClass(cls):
        """Create the test node, its subscriptions and the synthetic camera."""
        rclpy.init()
        cls.node = rclpy.create_node("test_pose_outputs")
        cls.poses = []
        cls.debug_images = []
        cls.node.create_subscription(
            PoseWithCovarianceStamped, "/mm_pose_estimator/pose", cls.poses.append, 10
        )
        cls.node.create_subscription(
            Image, "/mm_pose_estimator/debug_image", cls.debug_images.append, 10
        )
        cls.tf_buffer = Buffer()
        cls.tf_listener = TransformListener(cls.tf_buffer, cls.node)
        cls.camera = SyntheticCamera(cls.node)

    @classmethod
    def tearDownClass(cls):
        """Destroy the test node."""
        cls.node.destroy_node()
        rclpy.shutdown()

    def tf_yaw(self, stamp):
        """Return the yaw of world → base_tf at `stamp`, once the estimator broadcast it."""
        while not self.tf_buffer.can_transform("world", "base_tf", stamp):
            rclpy.spin_once(self.node, timeout_sec=0.05)
        q = self.tf_buffer.lookup_transform("world", "base_tf", stamp).transform.rotation
        return 2 * np.arctan2(q.z, q.w)

    def test_frame_without_markers_has_a_debug_image(self):
        """A blank image gives no pose but its debug image."""
        self.camera.wait_for(render_markers(0.0, []), self.debug_images)

    def test_tf_averages_yaws_across_the_quaternion_sign_flip(self):
        """
        The TF settles at a yaw just below π, then gets a yaw just above π.

        Their quaternions have opposite signs, so the average is only right when the TF keeps
        them on one sign. Weighting the new yaw by tf_alpha moves the TF a quarter of the way.
        """
        for _ in range(60):
            self.camera.wait_for(render_markers(np.pi - 0.1, MARKERS), self.poses)
        pose = self.camera.wait_for(render_markers(-np.pi + 0.1, MARKERS), self.poses)
        yaw = self.tf_yaw(pose.header.stamp) % (2 * np.pi)
        self.assertAlmostEqual(yaw, np.pi - 0.1 + TF_ALPHA * 0.2, delta=np.radians(1.0))
