# multi_marker_robot_pose_estimation

## Overview
This repository provides the `multi_marker_robot_pose_estimation` package, which can be used to estimate the robot pose based on a multiple ArUco markers.

**Authors: Jasper van Brakel, Wouter Caarls ([SAM XL](https://samxl.tudelftcampus.nl/))**

**Maintainers: Jasper van Brakel, J.C.vanBrakel@tudelft.nl**

This package hase been developed and tested under ROS2 Jazzy and Ubuntu 24.04 (Noble).

## Installation

### Building
To build from source, clone the latest version from this repository into your workspace:

```bash
cd ros_ws/src
git clone https://github.com/sam-xl/multi_marker_robot_pose_estimation.git
```

Use [`vcstool`](https://github.com/dirk-thomas/vcstool) to clone other project packages and complete the project workspace:

```bash
cd ros_ws/src
vcs import < multi_marker_robot_pose_estimation/.repos
```

Install the dependencies of the cloned packages using `rosdep`:

```bash
cd ros_ws/src
rosdep install --from-paths . --ignore-src -y
```

Finally, build all packages in the workspace:

```bash
colcon build
```

## Usage
To run the `pose_estimator` node run the following command

```bash
ros2 run multi_marker_robot_pose_estimation pose_estimator
```

To run the `mm_pose_estimator` node run the following command

```bash
ros2 run multi_marker_robot_pose_estimation mm_pose_estimator --ros-args -r image:=/camera/camera/color/image_raw -r camera_info:=/camera/camera/color/camera_info -p tf_frame_id:=base_link_estimated
```

## Nodes

### `pose_estimator`
Provides a service to estimate the pose based on a single specified ArUco marker.

#### Subscribed Topics
* `/tf{,_static}` ([tf2_msgs/msg/TFMessage](https://docs.ros.org/en/jazzy/p/tf2_msgs/msg/TFMessage.html))\
This nodes listens to the TF topics to keep it's own TF Buffer.

* `/aruco/markers` ([aruco_interfaces/msg/ArucoMarkers](https://github.com/sam-xl/ros2-aruco-pose-estimation/blob/main/aruco_interfaces/msg/ArucoMarkers.msg))\
The ArUco marker detections.

#### Services

* `get_estimated_pose` ([multi_marker_robot_pose_estimation_interfaces/srv/EstimatePose](./multi_marker_robot_pose_estimation_interfaces/srv/EstimatePose.srv))\
 Gets the estimated pose based on the specified `marker_id` and frames.

### `mm_pose_estimator`
Provides robot pose estimation based on all available aruco marker reference TFs. All TF frames ending in `aruco_ref_XXX` are used, with `XXX` being the ArUco marker ID. The node can publish the pose of any frame connected to the camera (default `base_link`) in any frame connected to the aruco markers (default `world`). It can optionally publish this pose as a TF under a chosen name.

#### Subscribed Topics
* `/tf{,_static}` ([tf2_msgs/msg/TFMessage](https://docs.ros.org/en/jazzy/p/tf2_msgs/msg/TFMessage.html))\
This node listens to the TF topics to keep its own TF Buffer.

* `image` ([sensor_msgs/Image](https://docs.ros.org/en/jazzy/p/sensor_msgs/msg/Image.html))\
The RGB camera image to detect markers on.

* `camera_info` ([sensor_msgs/CameraInfo](https://docs.ros.org/en/jazzy/p/sensor_msgs/msg/CameraInfo.html))\
The camera parameters associated with `image`.

#### Published Topics
* `/tf` ([tf2_msgs/msg/TFMessage](https://docs.ros.org/en/jazzy/p/tf2_msgs/msg/TFMessage.html))\
If desired, the estimated robot pose can be published on TF.

* `~/pose` ([geometry_msgs/PoseWithCovarianceStamped](https://docs.ros.org/en/jazzy/p/geometry_msgs/msg/PoseWithCovarianceStamped.html))
Robot pose estimate. The covariance is [calculated](docs/pose_estimation.pdf) based on the reprojection error of the marker detection.

* `~/debug_image` ([sensor_msgs/Image](https://docs.ros.org/en/jazzy/p/sensor_msgs/msg/Image.html))\
Camera image with superimposed detection information. Only published if `debug` parameter is set.

#### Parameters

* `marker_size` (double)\
Size of the markers in meters. Defaults to `0.2`.

* `aruco_dictionary_id` (string)\
Dictionary that was used to generate markers. Defaults to `DICT_ARUCO_MIP_36h12`

* `world_frame_id` (string)\
Frame in which to publish pose. Defaults to `world`. All aruco marker reference positions must be convertible to this frame.

* `base_frame_id` (string)\
Frame of which to publish pose. Defaults to `base_link`. The detected camera pose is converted to this frame.

* `tf_frame_id` (string)\
Frame as which to publish TF. If empty, no TF is published. This is useful if another node is already publishing the actual `world`->`base_frame_id` transformation.

* `tf_alpha` (double)\
Parameter of EMA filter for TF broadcaster. Defaults to `0.85`. This increases accuracy by averaging over multiple frames, provided that the robot is stationary.

* `fps` (double)\
Maximum frames per second to process. Defaults to `5`.

* `debug` (bool)\
Publish `~/debug_image` to debug pose estimation.

## Bugs & Feature Requests
Please report bugs and request features using the [Issue Tracker](https://github.com/sam-xl/multi_marker_robot_pose_estimation/issues).
