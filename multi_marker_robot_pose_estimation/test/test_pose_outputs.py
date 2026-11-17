"""Check the debug image of mm_pose_estimator."""

import unittest

import pytest
import rclpy
from launch import LaunchDescription
from launch_testing.actions import ReadyToTest
from sensor_msgs.msg import Image
from synthetic_camera import SyntheticCamera, estimator_node, render_markers, scene_tf_publishers

MARKERS = [(0, (-0.3, 0.25)), (1, (0.0, 0.25)), (2, (0.3, 0.25)), (3, (-0.3, -0.2))]
MARKER_FRAMES = [(f"aruco_ref_{i}", center) for i, center in MARKERS]


@pytest.mark.launch_test
def generate_test_description():
    """Launch the estimator with the debug image."""
    return LaunchDescription(
        [
            estimator_node(debug=True),
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
        cls.debug_images = []
        cls.node.create_subscription(
            Image, "/mm_pose_estimator/debug_image", cls.debug_images.append, 10
        )
        cls.camera = SyntheticCamera(cls.node)

    @classmethod
    def tearDownClass(cls):
        """Destroy the test node."""
        cls.node.destroy_node()
        rclpy.shutdown()

    def test_frame_without_markers_has_a_debug_image(self):
        """A blank image gives no pose but its debug image."""
        self.camera.wait_for(render_markers(0.0, []), self.debug_images)
