"""PoseEstimator Node, estimates pose based on a single ArUco marker."""

from typing import Optional

import numpy as np
import rclpy
import rclpy.node
import tf2_geometry_msgs  # noqa: F401
import tf2_ros
from aruco_interfaces.msg import ArucoMarkers
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.time import Duration, Time

from multi_marker_robot_pose_estimation.utils import matrix_to_tf, tf_to_matrix
from multi_marker_robot_pose_estimation_interfaces.srv import EstimatePose


class PoseEstimator(rclpy.node.Node):
    """PoseEstimator Node estimates pose based on a single ArUco marker."""

    def __init__(self) -> None:
        """Initializer of the PoseEstimator node."""
        super().__init__("pose_estimator")

        self._latest_markers: Optional[ArucoMarkers] = None
        self._marker_msg_sub = self.create_subscription(
            ArucoMarkers,
            "/aruco/markers",
            self._marker_callback,
            5,
        )

        self._get_estimated_pose_srv = self.create_service(
            EstimatePose,
            "/get_estimated_pose",
            self._get_estimated_pose_callback,
        )

        self._tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0), node=self)
        # We don't need this service, so let's clean it up
        self.destroy_service(self._tf_buffer.srv)
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

    def _marker_callback(self, msg: ArucoMarkers):
        self._latest_markers = msg

    def _get_estimated_pose_callback(
        self, req: EstimatePose.Request, res: EstimatePose.Response
    ) -> EstimatePose.Response:
        latest_msg: Optional[ArucoMarkers] = self._latest_markers

        if latest_msg is None:
            res.success = False
            self.get_logger().error("Message not yet received")
            return res

        target_marker_id = req.marker_id
        if target_marker_id not in latest_msg.marker_ids:
            res.success = False
            self.get_logger().error(f"Marker {target_marker_id} not found")
            return res

        detection_pose_stamped = PoseStamped(
            pose=latest_msg.poses[latest_msg.marker_ids.index(target_marker_id)],
            header=latest_msg.header,
        )

        try:
            detect_to_base_pose: PoseStamped = self._tf_buffer.transform(
                detection_pose_stamped,
                target_frame=req.base_frame_id,
                timeout=Duration(seconds=0.5),
            )
        except Exception as err:
            res.success = False
            self.get_logger().error(
                f"Unable to lookup transform between {req.base_frame_id!r}"
                f" and {detection_pose_stamped.header.frame_id!r}: {err}"
            )
            return res

        detect_to_base_tf = TransformStamped(header=detect_to_base_pose.header)
        detect_to_base_tf.transform.rotation = detect_to_base_pose.pose.orientation
        detect_to_base_tf.transform.translation.x = detect_to_base_pose.pose.position.x
        detect_to_base_tf.transform.translation.y = detect_to_base_pose.pose.position.y
        detect_to_base_tf.transform.translation.z = detect_to_base_pose.pose.position.z

        self.get_logger().info("Waiting for transform")
        try:
            floor_to_marker_tf: TransformStamped = self._tf_buffer.lookup_transform(
                target_frame=req.marker_frame_id,
                source_frame=req.parent_frame_id,
                time=Time(),
                timeout=Duration(seconds=0.5),
            )
        except Exception as err:
            res.success = False
            self.get_logger().error(
                f"Unable to lookup transform between {req.parent_frame_id!r}"
                f" and {req.marker_frame_id!r}: {err}"
            )
            return res

        parent_to_base_tf = matrix_to_tf(
            np.linalg.inv(
                tf_to_matrix(detect_to_base_tf.transform)
                @ tf_to_matrix(floor_to_marker_tf.transform)
            )
        )

        self.get_logger().info("Successfully positioned base_frame")

        res.success = True
        res.pose.header.stamp = floor_to_marker_tf.header.stamp
        res.pose.header.frame_id = req.parent_frame_id
        res.pose.pose.position.x = parent_to_base_tf.translation.x
        res.pose.pose.position.y = parent_to_base_tf.translation.y
        res.pose.pose.position.z = parent_to_base_tf.translation.z
        res.pose.pose.orientation = parent_to_base_tf.rotation

        return res


def main(args=None):
    """Entry point for 'pose_estimator'."""
    rclpy.init(args=args)

    node = PoseEstimator()

    rclpy.spin(node)

    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
