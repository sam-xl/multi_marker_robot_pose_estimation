#!/usr/bin/env python3
"""
ArUco multi-marker robot pose estimator.

ROS2 wrapper code taken from:
https://github.com/JMU-ROBOTICS-VIVA/ros2_aruco/tree/main
"""

import cv2
import message_filters
import numpy as np
import rclpy
import rclpy.node
import tf2_geometry_msgs  # noqa: F401
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.executors import ExternalShutdownException
from rclpy.qos import QoSHistoryPolicy, QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformBroadcaster, TransformListener

from multi_marker_robot_pose_estimation.aruco_board import (
    create_marker_board,
    detect_marker_board,
    estimate_camera_pose,
)
from multi_marker_robot_pose_estimation.utils import (
    matrix_to_pose,
    pose_to_matrix,
    skew,
    tf_to_matrix,
)


class ArucoLocalizationNode(rclpy.node.Node):
    """Node that performs localization using aruco marker references given in TF."""

    def __init__(self):
        """Create node."""
        super().__init__("mm_pose_estimator")

        self.initialize_parameters()

        dictionary_id = cv2.aruco.__getattribute__(self.dictionary_id_name)

        self.tf_buffer = Buffer(node=self)
        self.destroy_service(self.tf_buffer.srv)
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.bridge = CvBridge()

        info_sub = message_filters.Subscriber(
            self, CameraInfo, "camera_info", qos_profile=qos_profile_sensor_data
        )
        image_sub = message_filters.Subscriber(
            self,
            Image,
            "image",
            qos_profile=QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1),
        )

        self.ts = message_filters.TimeSynchronizer([info_sub, image_sub], 1)
        self.ts.registerCallback(self.sync_callback)

        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, "~/pose", 10)

        if self.debug:
            self.image_pub = self.create_publisher(Image, "~/debug_image", 10)

        self.transform = None
        self.last_img_stamp = rclpy.time.Time(clock_type=self.get_clock().clock_type)

        # Set up aruco detection. Requires OpenCV >= 4.8.0
        self.aruco_dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self.aruco_parameters = cv2.aruco.DetectorParameters()
        self.aruco_parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dictionary, self.aruco_parameters)

    def sync_callback(self, info_msg, img_msg):
        """
        Synchronized camera info and image callback.

        Performs the robot pose estimation.
        """
        intrinsic_mat = np.reshape(np.array(info_msg.k), (3, 3))
        distortion = np.array(info_msg.d)

        # Throttle localization rate
        if self.fps > 0:
            img_stamp = rclpy.time.Time.from_msg(img_msg.header.stamp)
            if (img_stamp - self.last_img_stamp).nanoseconds < (1.0 / self.fps) * 1e9:
                return
            self.last_img_stamp = img_stamp

        # convert the image messages to cv2 format
        cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")

        # Create Aruco board from published TF references
        board = create_marker_board(
            self.tf_buffer,
            self.world_frame_id,
            aruco_dict=self.aruco_dictionary,
            marker_size=self.marker_size,
        )

        if board is None:
            self.get_logger().warn("No aruco reference frames found", throttle_duration_sec=1.0)
            return

        # Detect markers
        corners, ids, _ = self.aruco_detector.detectMarkers(cv_image)

        if ids is None or len(ids) == 0:
            self.get_logger().info("No aruco markers detected", throttle_duration_sec=10.0)
            return

        result = detect_marker_board(
            corners, ids, board, intrinsic_mat, distortion, frame=cv_image if self.debug else None
        )

        if result is None:
            self.get_logger().info("No valid board markers detected", throttle_duration_sec=10.0)
            return

        objPoints, imgPoints = result

        # Estimate camera pose from detected markers
        # Pose is camera pose in world frame
        pose = estimate_camera_pose(
            objPoints,
            imgPoints,
            matrix_coefficients=intrinsic_mat,
            distortion_coefficients=distortion,
            frame=cv_image if self.debug else None,
        )

        if self.debug:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8"))

        if pose is None:
            self.get_logger().info(
                "Could not estimate camera pose from detected aruco board",
                throttle_duration_sec=10.0,
            )
            return

        try:
            # Get transform from base to camera
            Tbc = self.tf_buffer.lookup_transform(
                img_msg.header.frame_id, self.base_frame_id, rclpy.time.Time()
            )
        except Exception as e:
            self.get_logger().warn(
                f"Could not get base to camera transform: {e}",
                throttle_duration_sec=5.0,
            )
            return

        # Construct base frame pose in world frame
        Hbc = tf_to_matrix(Tbc.transform)
        Hcw = pose_to_matrix(pose.pose)
        Hbw = Hcw @ Hbc

        pose_base = PoseWithCovarianceStamped()
        pose_base.header.stamp = img_msg.header.stamp
        pose_base.header.frame_id = self.world_frame_id
        pose_base.pose.pose = matrix_to_pose(Hbw)

        # Transform covariance
        # Rotation covariance doesn't change because both the camera and
        # body pose are in the world frame. Only translation covariance
        # needs to take into account the lever arm effect.
        J = np.eye(6)
        lever = Hcw[:3, :3] @ Hbc[:3, 3]
        J[:3, 3:] = -skew(lever)
        cov_base = J @ pose.covariance.reshape((6, 6)) @ J.T

        pose_base.pose.covariance = cov_base.flatten()

        # Publish
        self.pose_pub.publish(pose_base)

        if self.tf_frame_id:
            t = TransformStamped()
            t.header = pose_base.header
            t.child_frame_id = self.tf_frame_id

            t.transform.translation.x = pose_base.pose.pose.position.x
            t.transform.translation.y = pose_base.pose.pose.position.y
            t.transform.translation.z = pose_base.pose.pose.position.z
            t.transform.rotation = pose_base.pose.pose.orientation

            # EMA filter
            if self.transform is not None:
                t.transform.translation.x = (
                    self.tf_alpha * self.transform.translation.x
                    + (1 - self.tf_alpha) * t.transform.translation.x
                )
                t.transform.translation.y = (
                    self.tf_alpha * self.transform.translation.y
                    + (1 - self.tf_alpha) * t.transform.translation.y
                )
                t.transform.translation.z = (
                    self.tf_alpha * self.transform.translation.z
                    + (1 - self.tf_alpha) * t.transform.translation.z
                )
                t.transform.rotation.x = (
                    self.tf_alpha * self.transform.rotation.x
                    + (1 - self.tf_alpha) * t.transform.rotation.x
                )
                t.transform.rotation.y = (
                    self.tf_alpha * self.transform.rotation.y
                    + (1 - self.tf_alpha) * t.transform.rotation.y
                )
                t.transform.rotation.z = (
                    self.tf_alpha * self.transform.rotation.z
                    + (1 - self.tf_alpha) * t.transform.rotation.z
                )
                t.transform.rotation.w = (
                    self.tf_alpha * self.transform.rotation.w
                    + (1 - self.tf_alpha) * t.transform.rotation.w
                )
            self.transform = t.transform

            self.tf_broadcaster.sendTransform(t)

    def initialize_parameters(self):
        """Declare and set ROS parameters."""
        self.declare_parameter(
            name="marker_size",
            value=0.2,
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description="Size of the markers in meters.",
            ),
        )

        self.declare_parameter(
            name="aruco_dictionary_id",
            value="DICT_ARUCO_MIP_36h12",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Dictionary that was used to generate markers.",
            ),
        )

        self.declare_parameter(
            name="world_frame_id",
            value="world",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Frame in which to publish pose",
            ),
        )

        self.declare_parameter(
            name="base_frame_id",
            value="base_link",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Frame of which to publish pose",
            ),
        )

        self.declare_parameter(
            name="tf_frame_id",
            value="",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Frame as which to publish TF. If empty, no TF is published",
            ),
        )

        self.declare_parameter(
            name="tf_alpha",
            value=0.85,
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description="Parameter of EMA filter for TF",
            ),
        )

        self.declare_parameter(
            name="fps",
            value=5,
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description="Maximum frames per second to process",
            ),
        )

        self.declare_parameter(
            name="debug",
            value=False,
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_BOOL,
                description="Show camera image to debug pose estimation",
            ),
        )

        self.marker_size = self.get_parameter("marker_size").get_parameter_value().double_value
        self.get_logger().info(f"Marker size: {self.marker_size}")

        self.dictionary_id_name = (
            self.get_parameter("aruco_dictionary_id").get_parameter_value().string_value
        )
        self.get_logger().info(f"Marker type: {self.dictionary_id_name}")

        self.world_frame_id = (
            self.get_parameter("world_frame_id").get_parameter_value().string_value
        )
        self.get_logger().info(f"World frame: {self.world_frame_id}")

        self.base_frame_id = self.get_parameter("base_frame_id").get_parameter_value().string_value
        self.get_logger().info(f"Base frame: {self.base_frame_id}")

        self.tf_frame_id = self.get_parameter("tf_frame_id").get_parameter_value().string_value
        self.get_logger().info(f"TF frame: {self.tf_frame_id}")

        self.tf_alpha = self.get_parameter("tf_alpha").get_parameter_value().double_value
        self.get_logger().info(f"Alpha: {self.tf_alpha}")

        self.fps = self.get_parameter("fps").get_parameter_value().integer_value
        self.get_logger().info(f"Max FPS: {self.fps}")

        self.debug = self.get_parameter("debug").get_parameter_value().bool_value
        self.get_logger().info(f"Show debug information: {self.debug}")


def main():
    """Entry point."""
    rclpy.init()
    node = None
    try:
        node = ArucoLocalizationNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
