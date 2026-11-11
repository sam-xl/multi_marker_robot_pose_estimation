"""Check the TF EMA of mm_pose_estimator for a base yawed close to 180°."""

import time
import unittest

import cv2
import launch_testing
import numpy as np
import pytest
import rclpy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import TransformStamped
from launch import LaunchDescription
from launch_ros.actions import Node
from sensor_msgs.msg import CameraInfo, Image
from tf2_msgs.msg import TFMessage
from tf2_ros import StaticTransformBroadcaster

ESTIMATE_FRAME_ID = "base_link_estimate"
MARKER_SIZE = 0.2
MARKER_CENTERS = [(-0.15, 0.15), (0.15, 0.15), (0.15, -0.15), (-0.15, -0.15)]
WIDTH, HEIGHT = 640, 480
K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
CAMERA_HEIGHT = 1.0
BASE_YAWS_DEG = [179.5, -179.5] * 3


@pytest.mark.launch_test
def generate_test_description():
    """Launch the estimator with a dictionary that the test can draw with system OpenCV."""
    estimator = Node(
        package="multi_marker_robot_pose_estimation",
        executable="mm_pose_estimator",
        parameters=[
            {
                "tf_frame_id": ESTIMATE_FRAME_ID,
                "aruco_dictionary_id": "DICT_4X4_50",
                "marker_size": MARKER_SIZE,
                "fps": 0,
            }
        ],
    )
    return LaunchDescription([estimator, launch_testing.actions.ReadyToTest()])


def rotation_x(angle):
    """Return the rotation matrix of `angle` radians about x."""
    return cv2.Rodrigues(np.array([angle, 0.0, 0.0]))[0]


def rotation_z(angle):
    """Return the rotation matrix of `angle` radians about z."""
    return cv2.Rodrigues(np.array([0.0, 0.0, angle]))[0]


def static_transform(parent, child, translation, quaternion_xyzw):
    """Return a TransformStamped for the static broadcaster."""
    t = TransformStamped()
    t.header.frame_id = parent
    t.child_frame_id = child
    t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = translation
    r = t.transform.rotation
    r.x, r.y, r.z, r.w = quaternion_xyzw
    return t


def render_board(base_yaw):
    """
    Render the markers on z = 0 as seen by the camera of a base with `base_yaw` in world.

    The camera is CAMERA_HEIGHT above the board origin, looking down (base_link rotated by
    π about x), so camera-in-world is RZ(base_yaw) @ RX(π).
    """
    camera_in_world = rotation_z(base_yaw) @ rotation_x(np.pi)
    camera_position = np.array([0.0, 0.0, CAMERA_HEIGHT])
    rvec = cv2.Rodrigues(camera_in_world.T)[0]
    tvec = -camera_in_world.T @ camera_position

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    side = 120
    marker_corners = np.float32([[0, 0], [side, 0], [side, side], [0, side]])
    image = np.full((HEIGHT, WIDTH), 255, dtype=np.uint8)
    half = MARKER_SIZE / 2
    for marker_id, (cx, cy) in enumerate(MARKER_CENTERS):
        world_corners = np.array(
            [
                [cx - half, cy + half, 0.0],
                [cx + half, cy + half, 0.0],
                [cx + half, cy - half, 0.0],
                [cx - half, cy - half, 0.0],
            ]
        )
        image_corners = cv2.projectPoints(world_corners, rvec, tvec, K, None)[0]
        homography = cv2.getPerspectiveTransform(marker_corners, np.float32(image_corners))
        marker = cv2.aruco.drawMarker(dictionary, marker_id, side)
        warped = cv2.warpPerspective(marker, homography, (WIDTH, HEIGHT), borderValue=255)
        image = np.minimum(image, warped)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


class TestTfEma(unittest.TestCase):
    """Feed images of a base alternating between yaw ±179.5° and check the published TF."""

    @classmethod
    def setUpClass(cls):
        """Create the test node, its publishers and the static TF of the scene."""
        rclpy.init()
        cls.node = rclpy.create_node("test_tf_ema")
        cls.info_pub = cls.node.create_publisher(CameraInfo, "camera_info", 10)
        cls.image_pub = cls.node.create_publisher(Image, "image", 10)
        cls.estimates = []
        cls.node.create_subscription(
            TFMessage,
            "/tf",
            lambda msg: cls.estimates.extend(
                t for t in msg.transforms if t.child_frame_id == ESTIMATE_FRAME_ID
            ),
            10,
        )
        cls.tf_broadcaster = StaticTransformBroadcaster(cls.node)
        cls.tf_broadcaster.sendTransform(
            [
                static_transform("world", f"aruco_ref_{i}", (cx, cy, 0.0), (0.0, 0.0, 0.0, 1.0))
                for i, (cx, cy) in enumerate(MARKER_CENTERS)
            ]
            + [static_transform("base_link", "camera", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))]
        )

    @classmethod
    def tearDownClass(cls):
        """Destroy the test node."""
        cls.node.destroy_node()
        rclpy.shutdown()

    def publish(self, image, stamp):
        """Publish the camera info and the image with the same stamp."""
        info = CameraInfo()
        info.header.stamp = stamp
        info.header.frame_id = "camera"
        info.width, info.height = WIDTH, HEIGHT
        info.k = K.flatten().tolist()
        info.d = [0.0] * 5
        msg = Image()
        msg.header = info.header
        msg.width, msg.height = WIDTH, HEIGHT
        msg.encoding = "bgr8"
        msg.step = WIDTH * 3
        msg.data = image.tobytes()
        self.info_pub.publish(info)
        self.image_pub.publish(msg)

    def estimate_for(self, image, stamp_sec, timeout_sec=30.0):
        """
        Publish `image` until the estimator broadcasts its TF for it and return that TF.

        The first images are dropped while the estimator starts up and receives the static TF,
        so each retry is published with a new stamp.
        """
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            stamp_sec += 1
            stamp = Time(sec=stamp_sec)
            self.publish(image, stamp)
            retry = time.monotonic() + 1.0
            while time.monotonic() < retry:
                rclpy.spin_once(self.node, timeout_sec=0.05)
                for t in self.estimates:
                    if t.header.stamp == stamp:
                        return t, stamp_sec
        self.fail(f"no {ESTIMATE_FRAME_ID} TF within {timeout_sec} s")

    def test_rotation_stays_unit_and_at_180_deg(self):
        """Without sign alignment, the second image would average q and ≈ -q to norm 0.70."""
        images = {yaw: render_board(np.radians(yaw)) for yaw in set(BASE_YAWS_DEG)}
        stamp_sec = 0
        for yaw in BASE_YAWS_DEG:
            estimate, stamp_sec = self.estimate_for(images[yaw], stamp_sec)
            r = estimate.transform.rotation
            q = np.array([r.x, r.y, r.z, r.w])
            with self.subTest(base_yaw_deg=yaw, stamp_sec=stamp_sec):
                self.assertAlmostEqual(np.linalg.norm(q), 1.0, delta=1e-6)
                angle_from_yaw_180 = np.degrees(2 * np.arccos(min(1.0, abs(r.z))))
                self.assertLess(angle_from_yaw_180, 1.0)
