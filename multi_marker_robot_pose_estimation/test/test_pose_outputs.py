"""Check the debug image, the smoothed TF and the pose covariance of mm_pose_estimator."""

import unittest

import numpy as np
import pytest
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from launch import LaunchDescription
from launch_testing.actions import ReadyToTest
from rclpy.parameter import Parameter
from rclpy.parameter_client import AsyncParameterClient
from sensor_msgs.msg import Image
from synthetic_camera import (
    SyntheticCamera,
    broadcast_marker_frames,
    camera_tf_publisher,
    estimator_node,
    render_markers,
)
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
            camera_tf_publisher(),
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
        broadcast_marker_frames(cls.node, MARKER_FRAMES)

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

    def test_covariance_scales_with_sigma_min_squared(self):
        """The synthetic corners leave sub-pixel residuals, so sigma_min sets σ²."""
        image = render_markers(0.0, MARKERS)
        parameters = AsyncParameterClient(self.node, "mm_pose_estimator")
        self.assertTrue(parameters.wait_for_services(timeout_sec=30.0))
        covariances = []
        for sigma_min in (1.0, 2.0):
            future = parameters.set_parameters([Parameter("sigma_min", value=sigma_min)])
            rclpy.spin_until_future_complete(self.node, future)
            pose = self.camera.wait_for(image, self.poses).pose
            covariances.append(np.reshape(pose.covariance, (6, 6)))
        np.testing.assert_allclose(covariances[1], 4 * covariances[0], rtol=1e-6)
        np.testing.assert_allclose(covariances[0], covariances[0].T, rtol=1e-6)
        self.assertGreater(np.linalg.eigvalsh(covariances[0]).min(), 0.0)
