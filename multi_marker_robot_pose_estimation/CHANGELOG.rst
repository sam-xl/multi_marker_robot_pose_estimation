^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for multi_marker_robot_pose_estimation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Forthcoming
-----------
- Publish ``~/debug_image`` also for frames without a pose, without
  overlays, and with the camera image's header. It kept showing the
  last frame with a pose and had an empty header.

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
