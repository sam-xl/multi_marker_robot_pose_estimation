"""
Synthetic camera above flat ArUco markers, for launch tests of mm_pose_estimator.

Inspired by:
  https://github.com/opencv/opencv/blob/b4c5ec4042f097e2a5b386b9d413ec7333d0a184/modules/objdetect/test/test_arucodetection.cpp
"""

import time

import cv2
import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from cv_bridge import CvBridge
from launch_ros.actions import Node
from sensor_msgs.msg import CameraInfo, Image
from transforms3d.euler import euler2mat

MARKER_SIZE = 0.2
WIDTH, HEIGHT = 640, 480
K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
CAMERA_HEIGHT = 1.0


def estimator_node(**parameters):
    """Return the estimator with a dictionary that the test can draw with system OpenCV."""
    return Node(
        package="multi_marker_robot_pose_estimation",
        executable="mm_pose_estimator",
        parameters=[
            {"aruco_dictionary_id": "DICT_4X4_50", "marker_size": MARKER_SIZE, "fps": 0},
            parameters,
        ],
    )


def static_transform_publisher(frame_id, child_frame_id, *pose_arguments):
    """Return a tf2_ros static_transform_publisher of child_frame_id in frame_id."""
    return Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=[*pose_arguments, "--frame-id", frame_id, "--child-frame-id", child_frame_id],
    )


def scene_tf_publishers(marker_frames):
    """
    Return the static TF publishers of the scene that render_markers draws.

    `marker_frames`, [(frame, (x, y)), ...], are in world; the camera looks down from base_link.
    """
    return [
        static_transform_publisher("world", frame, "--x", str(x), "--y", str(y))
        for frame, (x, y) in marker_frames
    ] + [static_transform_publisher("base_link", "camera", "--roll", str(np.pi))]


def render_markers(base_yaw, markers):
    """
    Render `markers`, [(marker_id, (x, y)), ...] flat on z = 0, for a base with `base_yaw`.

    The base, and its camera, are CAMERA_HEIGHT above the world origin. The camera looks down
    (base_link rotated by π about x), so camera-in-world is RZ(base_yaw) @ RX(π).
    """
    camera_in_world = euler2mat(np.pi, 0.0, base_yaw)
    camera_position = np.array([0.0, 0.0, CAMERA_HEIGHT])
    rvec = cv2.Rodrigues(camera_in_world.T)[0]
    tvec = -camera_in_world.T @ camera_position

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    side = 120
    marker_corners = np.float32([[0, 0], [side, 0], [side, side], [0, side]])
    image = np.full((HEIGHT, WIDTH), 255, dtype=np.uint8)
    half = MARKER_SIZE / 2
    for marker_id, (cx, cy) in markers:
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


class SyntheticCamera:
    """Publishes the rendered images on the estimator's inputs."""

    def __init__(self, node):
        """Create the camera info and image publishers on `node`."""
        self.node = node
        self.info_pub = node.create_publisher(CameraInfo, "camera_info", 10)
        self.image_pub = node.create_publisher(Image, "image", 10)
        self.bridge = CvBridge()
        self.stamp_sec = 0

    def publish(self, image, stamp):
        """Publish the camera info and the image with the same stamp."""
        info = CameraInfo()
        info.header.stamp = stamp
        info.header.frame_id = "camera"
        info.width, info.height = WIDTH, HEIGHT
        info.k = K.flatten().tolist()
        info.d = [0.0] * 5
        self.info_pub.publish(info)
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(image, "bgr8", info.header))

    def wait_for(self, image, received, timeout_sec=30.0):
        """
        Publish `image` until a message with its stamp is appended to `received`; return it.

        The first images are dropped while the estimator starts up and receives the static TF,
        so each retry is published with a new stamp.

        https://github.com/ros2/launch_ros/blob/jazzy/launch_testing_ros/test/examples/talker_listener_launch_test.py#L108-L112
        """
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            self.stamp_sec += 1
            stamp = Time(sec=self.stamp_sec)
            self.publish(image, stamp)
            # Wait for the estimator's reply to the frame just published
            retry = time.monotonic() + 1.0
            while time.monotonic() < retry:
                rclpy.spin_once(self.node, timeout_sec=0.05)
                for msg in received:
                    if msg.header.stamp == stamp:
                        return msg
        raise AssertionError(f"no estimate within {timeout_sec} s")
