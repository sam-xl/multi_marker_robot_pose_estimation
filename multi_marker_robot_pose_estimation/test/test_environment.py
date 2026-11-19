import os

import cv2
from ament_index_python.packages import get_package_share_directory


def test_opencv_is_the_pinned_version():
    """The tests run with the OpenCV that requirements.txt pins, as the node does."""
    share = get_package_share_directory("multi_marker_robot_pose_estimation")
    with open(os.path.join(share, "requirements.txt")) as requirements:
        [pin] = [
            line.split("==")[1].strip()
            for line in requirements
            if line.startswith("opencv-python==")
        ]
    # opencv-python appends a build number: 4.8.0.76 is OpenCV 4.8.0.
    assert cv2.__version__ == pin.rsplit(".", 1)[0]
