"""Check that mm_pose_estimator floors the pixel noise of the pose covariance at sigma_min."""

import unittest

import launch_testing
import numpy as np
import pytest
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from launch import LaunchDescription
from rcl_interfaces.srv import SetParameters
from rclpy.parameter import Parameter
from synthetic_camera import SyntheticCamera, estimator_node, render_markers

MARKER = (0, (0.0, 0.0))


@pytest.mark.launch_test
def generate_test_description():
    """Launch the estimator with a floor far above the residuals of a rendered marker."""
    return LaunchDescription([estimator_node(sigma_min=2.0), launch_testing.actions.ReadyToTest()])


class TestSigmaMin(unittest.TestCase):
    """A noiseless single-marker image has near-zero residuals, so σ is sigma_min."""

    @classmethod
    def setUpClass(cls):
        """Create the test node, its pose subscription, parameter client and synthetic camera."""
        rclpy.init()
        cls.node = rclpy.create_node("test_sigma_min")
        cls.poses = []
        cls.node.create_subscription(
            PoseWithCovarianceStamped, "/mm_pose_estimator/pose", cls.poses.append, 10
        )
        cls.set_parameters = cls.node.create_client(
            SetParameters, "/mm_pose_estimator/set_parameters"
        )
        cls.camera = SyntheticCamera(cls.node, [("aruco_ref_0", MARKER[1])])

    @classmethod
    def tearDownClass(cls):
        """Destroy the test node."""
        cls.node.destroy_node()
        rclpy.shutdown()

    def set_sigma_min(self, value):
        """Set sigma_min on the running estimator."""
        self.assertTrue(self.set_parameters.wait_for_service(timeout_sec=10.0))
        request = SetParameters.Request(
            parameters=[Parameter("sigma_min", value=value).to_parameter_msg()]
        )
        future = self.set_parameters.call_async(request)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=10.0)
        self.assertTrue(future.result().results[0].successful)

    def test_covariance_scales_with_sigma_min_squared(self):
        """Doubling sigma_min at runtime quadruples the covariance of the same image."""
        image = render_markers(0.0, [MARKER])
        covariance = self.camera.wait_for(image, self.poses).pose.covariance
        self.set_sigma_min(4.0)
        quadrupled = self.camera.wait_for(image, self.poses).pose.covariance
        np.testing.assert_allclose(quadrupled, 4.0 * np.asarray(covariance), rtol=1e-6, atol=0)
