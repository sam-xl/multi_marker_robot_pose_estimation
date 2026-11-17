^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for multi_marker_robot_pose_estimation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Forthcoming
-----------
- Publish ``~/debug_image`` also for frames without a pose, without
  overlays, and with the camera image's header. It kept showing the
  last frame with a pose and had an empty header.
- Fix the TF EMA dropping the TF for rotations near 180°: average the
  quaternion on the previous one's sign and renormalise it.
- Make ``tf_alpha`` the weight of the newest estimate, as in the usual
  EMA definition; it was the weight of the previous one. Replace a value
  ``a`` set before with ``1 - a``. The default changes from 0.85 to 0.15,
  which smooths the same.
- Estimate the pixel noise of the pose covariance per coordinate over the
  2N - 6 residual degrees of freedom, floored by the new ``sigma_min``
  parameter. It was half the noise variance for a single marker.
- Fit the pose only to the marker corners that agree on it, found with
  ``cv2.solvePnPRansac``, and publish no pose when they come from fewer
  than ``min_markers`` markers. The new ``reprojection_error`` parameter
  sets the agreement threshold. Rejected corners are circled red on
  ``~/debug_image``.
- Remove the marker area (400 px²) and squareness filters. The outlier
  rejection catches the misdetections they caught, and the area filter
  mostly dropped usable markers.

0.1.2 (2026-11-03)
------------------
- Estimate pose covariance from reprojection error
- Remove ``pose_estimator`` node.
- Add ci pipeline for github actions.
- Remove unused functions in utils.py

0.1.1 (2026-09-09)
------------------
- Fix crashes caused by all markers being filtered out

0.1.0 (2026-08-28)
------------------
- Add ``pose_estimator`` node for single-marker pose estimation
- Add ``mm_pose_estimator`` node for multi-marker pose estimation
