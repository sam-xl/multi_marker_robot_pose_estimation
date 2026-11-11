"""Check the TF EMA of mm_pose_estimator for a base yawed close to 180°."""

import unittest

import launch_testing
import numpy as np
import pytest
import rclpy
from launch import LaunchDescription
from synthetic_camera import SyntheticCamera, estimator_node, render_markers
from tf2_msgs.msg import TFMessage

ESTIMATE_FRAME_ID = "base_link_estimate"
MARKER_CENTERS = [(-0.15, 0.15), (0.15, 0.15), (0.15, -0.15), (-0.15, -0.15)]
BASE_YAWS_DEG = [179.5, -179.5] * 3


@pytest.mark.launch_test
def generate_test_description():
    """Launch the estimator broadcasting its TF."""
    return LaunchDescription(
        [estimator_node(tf_frame_id=ESTIMATE_FRAME_ID), launch_testing.actions.ReadyToTest()]
    )


class TestTfEma(unittest.TestCase):
    """Feed images of a base alternating between yaw ±179.5° and check the published TF."""

    @classmethod
    def setUpClass(cls):
        """Create the test node, its TF subscription and the synthetic camera."""
        rclpy.init()
        cls.node = rclpy.create_node("test_tf_ema")
        cls.estimates = []
        cls.node.create_subscription(
            TFMessage,
            "/tf",
            lambda msg: cls.estimates.extend(
                t for t in msg.transforms if t.child_frame_id == ESTIMATE_FRAME_ID
            ),
            10,
        )
        cls.camera = SyntheticCamera(
            cls.node, [(f"aruco_ref_{i}", center) for i, center in enumerate(MARKER_CENTERS)]
        )

    @classmethod
    def tearDownClass(cls):
        """Destroy the test node."""
        cls.node.destroy_node()
        rclpy.shutdown()

    def test_rotation_stays_unit_and_at_180_deg(self):
        """Without sign alignment, the second image would average q and ≈ -q to norm 0.70."""
        images = {
            yaw: render_markers(np.radians(yaw), enumerate(MARKER_CENTERS))
            for yaw in set(BASE_YAWS_DEG)
        }
        for yaw in BASE_YAWS_DEG:
            estimate = self.camera.wait_for(images[yaw], self.estimates)
            r = estimate.transform.rotation
            q = np.array([r.x, r.y, r.z, r.w])
            with self.subTest(base_yaw_deg=yaw, stamp_sec=estimate.header.stamp.sec):
                self.assertAlmostEqual(np.linalg.norm(q), 1.0, delta=1e-6)
                angle_from_yaw_180 = np.degrees(2 * np.arccos(min(1.0, abs(r.z))))
                self.assertLess(angle_from_yaw_180, 1.0)
