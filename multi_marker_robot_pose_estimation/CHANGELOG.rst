^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for multi_marker_robot_pose_estimation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Forthcoming
-----------
- Fix the TF EMA dropping the TF for rotations near 180°: average the
  quaternion on the previous one's sign and renormalise it.
- Ignore marker IDs detected more than once in an image or given by more
  than one ``aruco_ref_<id>`` frame, instead of fitting the pose to both.
- Estimate the pixel noise of the pose covariance per coordinate over the
  2N - 6 residual degrees of freedom, floored by the new ``sigma_min``
  parameter. It was half the noise variance for a single marker.

0.1.2 (2026-11-03)
------------------
- Estimate pose covariance from reprojection error
- Remove ``pose_estimator`` node.
- Add ci pipeline for github actions.

0.1.1 (2026-09-09)
------------------
- Fix crashes caused by all markers being filtered out

0.1.0 (2026-08-28)
------------------
- Add ``pose_estimator`` node for single-marker pose estimation
- Add ``mm_pose_estimator`` node for multi-marker pose estimation
