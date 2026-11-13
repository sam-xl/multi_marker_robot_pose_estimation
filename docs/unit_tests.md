# Unit tests

The unit tests for `multi_marker_robot_pose_estimation` cover this package's own logic in `utils.py` and `aruco_board.py`. They don't re-test numpy, OpenCV or transforms3d. Each test targets a mistake this package could make: a quaternion ordering mix-up, a wrong corner order, a filter threshold, or a frame-convention error.

## Running the tests

```bash
source /opt/ros/jazzy/setup.bash
cd ros_ws
colcon build --packages-select multi_marker_robot_pose_estimation
source install/setup.bash
colcon test --packages-select multi_marker_robot_pose_estimation --event-handlers console_direct+
colcon test-result --verbose
```

For a quicker loop after sourcing:

```bash
python3 -m pytest src/multi_marker_robot_pose_estimation/multi_marker_robot_pose_estimation/test -v
```

The tests use pytest only and need no extra dependencies. They must pass `ruff check` and `ruff format --check`, using the repository's `pyproject.toml`, the same as the rest of the code.

## Layout

```
multi_marker_robot_pose_estimation/test/
├── conftest.py            shared fixtures
├── test_utils.py          T1, T2, T3
└── test_aruco_board.py    T5, T6, T8, T9
```

## Shared fixtures (`conftest.py`)

| Fixture             | Provides                                                                                                                                                                                                                                  |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `make_tf_buffer`    | The `FakeTfBuffer` class, a stand-in for `tf2_ros.Buffer`. `all_frames_as_yaml()` lists every configured frame. `lookup_transform()` returns the configured `Transform`, or raises for frames listed in `failing` and for unknown frames. |
| `make_transform`    | Builds a `geometry_msgs/Transform` from a translation and an `(x, y, z, w)` quaternion.                                                                                                                                                   |
| `assert_quat_equal` | Compares two `(x, y, z, w)` quaternions and treats `q` and `-q` as equal.                                                                                                                                                                 |
| `camera_intrinsics` | `(K, dist)` with `fx = fy = 600`, `cx = 320`, `cy = 240` and zero distortion.                                                                                                                                                             |
| `aruco_dict`        | `cv2.aruco.DICT_4X4_50`.                                                                                                                                                                                                                  |

All values are fixed and nothing is random, so every run is deterministic.

## Tests

### `test_utils.py`

| ID  | Test                                           | What it checks                                                                                                                                                                                                                                       |
| --- | ---------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| T1  | `test_known_value_conversion[pose\|transform]` | `pose_to_matrix` and `tf_to_matrix` turn a 90° rotation about Z, `(x=0, y=0, z=sin45°, w=cos45°)`, with translation `(1, 2, 3)` into the exact expected 4×4 matrix. A known value catches a `w,x,y,z` vs `x,y,z,w` mix-up, which a round trip can't. |
| T2  | `test_pose_matrix_round_trip[...]`             | `matrix_to_pose(pose_to_matrix(p))` gives back `p` for the identity, the T1 pose, and a 180° rotation about X with an offset. Orientations are compared with `q ≡ -q`.                                                                               |
| T3  | `test_H_translation_shapes[...]`               | `H(R, t)` gives the same matrix whether `t` has shape `(3, 1)` (as `cv2.solvePnP` returns it) or `(3,)`, or is a plain list.                                                                                                                         |

### `test_aruco_board.py`

| ID  | Test                                                   | What it checks                                                                                                                                                                                                                                                                                                                                                                           |
| --- | ------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| T5  | `test_create_marker_board_ids_and_corners`             | From TF frames `aruco_ref_3` and `aruco_ref_7`, the board has ids `{3, 7}`. Each marker's corners are `float32` with shape `(4, 3)`, in OpenCV's TL→TR→BR→BL order. For the marker with a rotated and translated TF, the corners are the identity corners moved by that TF. The expected values are worked out by hand, not with the code under test.                                    |
| T6  | `test_create_marker_board_ignores_non_matching_frames` | Frames that don't match `aruco_ref_<n>` (`camera_link`, `aruco_marker_3`, `world`) are ignored. They have valid transforms, so only the name filter can reject them.                                                                                                                                                                                                                     |
| T6  | `test_create_marker_board_skips_failed_lookup`         | A marker whose TF lookup fails is skipped, and the other markers are kept.                                                                                                                                                                                                                                                                                                               |
| T6  | `test_create_marker_board_returns_none[...]`           | Returns `None` for an empty frame list (`"{}"`), when no frames match, and when every lookup fails.                                                                                                                                                                                                                                                                                      |
| T8  | `test_detect_marker_board_filters[...]`                | The size and squareness filters, using image corners in TL→TR→BR→BL order: a 21×21 px square is kept (4 object points). A 19×19 px square is rejected (area 361 < 400). A 100×10 px rectangle is rejected (squareness ≈ 0.57 < 0.8). An id that isn't on the board is rejected. A valid marker plus a too-small one gives exactly 4 points. `ids=None` and empty `ids` return `None`.    |
| T9  | `test_estimate_camera_pose_synthetic`                  | 12 corners from 3 markers on the world XY plane are projected with a known camera pose, about 2 m above the plane looking down with a slight tilt. `estimate_camera_pose` must return that pose (`T_world_camera`) within 1e-4 m and 1e-4 per rotation entry, plus a 36-element covariance. This checks the package's `Hwc → Hcw` inversion and frame convention, not `solvePnP` itself. |

## Things to know

- **`rclpy.time` is imported in `conftest.py`.** `create_marker_board` calls `rclpy.time.Time()` but `aruco_board.py` only does `import rclpy`, which doesn't load the `rclpy.time` submodule. In the node, `rclpy.node` loads it as a side effect. In isolation the call raises `AttributeError`. The broad `except Exception` around the TF lookup swallows that error, so every marker is skipped and the function quietly returns `None`. `conftest.py` imports `rclpy.time` to match the node's runtime. The real fix is `import rclpy.time` in `aruco_board.py`.
- **OpenCV ≥ 4.7 is required.** `detect_marker_board` uses `cv2.aruco.Board.matchImagePoints`, which was added in OpenCV 4.7. rosdep installs Ubuntu's `python3-opencv` 4.6. CI therefore installs the version pinned in the package's `requirements.txt` (`opencv-python==4.8.0.76`) inside the industrial_ci container (see `.github/workflows/ci.yml`). Locally, check with `python3 -c "import cv2; print(cv2.__version__)"`.
