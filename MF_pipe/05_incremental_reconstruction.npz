"""
Step 05: Add the camera at frame 60 to our growing 3-D reconstruction.

Beginner's overview
-------------------
We have already reconstructed some scene points using frames 0 and 30.
Now we do two jobs:

1. Find where the camera was when frame 60 was taken.
   We do this by matching known 3-D points to their 2-D image locations.
2. Find additional 3-D points.
   We look for tracks seen in both frames 30 and 60 that do not already
   have a reconstructed 3-D point, then estimate their 3-D positions.

The result is still a sparse reconstruction: a collection of points, not
a complete solid model of the scene.
"""

import numpy as np
import cv2


# ------------------------------------------------------------
# 1. Load the reconstruction built by Step 04
# ------------------------------------------------------------
# Step 04 used frames 0 and 30 to estimate the first camera poses and
# reconstruct some scene points.
#
# The .npz file is like a small box containing named arrays. We load:
# - K and distortion: information about the camera lens (calibration).
# - points_3d: our current estimates of the scene points' 3-D positions.
# - point_track_ids: which feature track belongs to each 3-D point.
# - R0, t0 and R30, t30: the estimated positions/orientations of cameras 0 and 30.
initial = np.load("04_initial_reconstruction.npz")

K = initial["K"]
distortion = initial["distortion"]

# Keep the 3-D coordinates as float64 for OpenCV's geometry calculations.
# Each row is one point: [X, Y, Z].
points_3d = initial["points_3d"].astype(np.float64)

# Each entry identifies the feature track associated with the corresponding
# row in points_3d. The two arrays must stay in the same order.
point_track_ids = initial["point_track_ids"].astype(np.int32)

R0 = initial["R0"]
t0 = initial["t0"]

R30 = initial["R30"]
t30 = initial["t30"]

print("Initial reconstruction loaded")
print(f"Existing 3-D points: {len(points_3d)}")
print(f"Registered frames: {initial['registered_frame_ids']}")


# ------------------------------------------------------------
# 2. Load feature tracks connecting observations across frames
# ------------------------------------------------------------
# A feature is a recognizable image detail, such as a corner or texture patch.
# A track groups observations that the matching process believes are the same
# real-world detail across multiple frames.
#
# For example, one track might say:
#   frame 0 -> feature 17
#   frame 30 -> feature 42
#   frame 60 -> feature 9
#
# We need this information to find where existing 3-D points appear in frame 60,
# and to find new points that can be reconstructed from frames 30 and 60.
track_data = np.load("03_tracks.npz", allow_pickle=True)

# Sort "track_2", "track_10", etc. by their numeric suffix.
# Plain alphabetical sorting would put "track_10" before "track_2".
track_keys = sorted(
    track_data.files,
    key=lambda key: int(key.removeprefix("track_"))
)

# IMPORTANT: Step 04 assigned track IDs by enumerating this sorted list.
# We repeat the same rule so a saved point's track ID still refers to the
# same track here. Changing this convention would connect points to the wrong
# image observations.
track_lookup = {
    track_id: track_data[key].tolist()
    for track_id, key in enumerate(track_keys)
}

print(f"Total tracks available: {len(track_lookup)}")


# ------------------------------------------------------------
# 3. Load detected image features for frames 30 and 60
# ------------------------------------------------------------
# The feature file stores keypoints detected earlier.
# Each keypoint row contains image coordinates; [:2] selects its x and y pixels.
features = np.load("01_features.npz")

keypoints_30 = features["keypoints_30"]
keypoints_60 = features["keypoints_60"]


# ------------------------------------------------------------
# 4. Find known 3-D points that are also visible in frame 60
# ------------------------------------------------------------
# To locate camera 60, we need pairs of:
#   - a known 3-D point in our current reconstruction, and
#   - the pixel where that same point appears in frame 60.
#
# These are called 3-D-to-2-D correspondences.
object_points = []
image_points = []

for point_3d, track_id in zip(points_3d, point_track_ids):

    # Find the observations belonging to this point's feature track.
    # dict(track) lets us look up a feature index by frame number.
    track = track_lookup[int(track_id)]
    observations = dict(track)

    # If the track was not observed in frame 60, it cannot help locate camera 60.
    if 60 not in observations:
        continue

    # The track tells us which detected keypoint in frame 60 matches this 3-D point.
    feature_idx_60 = observations[60]

    object_points.append(point_3d)
    image_points.append(keypoints_60[feature_idx_60, :2])


# Convert lists to NumPy arrays in the numeric formats expected by OpenCV.
object_points = np.asarray(object_points, dtype=np.float64)
image_points = np.asarray(image_points, dtype=np.float32)

print()
print("Frame 60 camera registration")
print(f"3-D-to-2-D correspondences: {len(object_points)}")

# PnP needs several reliable correspondences to estimate a camera pose.
# This is a simple safety check to stop if we have too few.
if len(object_points) < 6:
    raise RuntimeError(
        "Not enough 3-D-to-2-D correspondences to register frame 60."
    )


# ------------------------------------------------------------
# 5. Estimate the pose (orientation and position) of camera 60 with PnP
# ------------------------------------------------------------
# Perspective-n-Point (PnP) answers:
#   "Given known 3-D points and where they appear in this image,
#    what camera orientation and position best explain those observations?"
#
# solvePnPRansac tries to estimate the pose while ignoring mismatched or
# inaccurate correspondences (called outliers). Its settings here ask it to:
# - try up to 1000 candidate iterations;
# - treat a point as an inlier if its reprojection error is within 4 pixels;
# - use a 99% confidence target.
#
# K and distortion describe the camera, so OpenCV can account for the lens.
success, rvec, t60, pnp_inliers = cv2.solvePnPRansac(
    object_points,
    image_points,
    K,
    distortion,
    iterationsCount=1000,
    reprojectionError=4.0,
    confidence=0.99,
    flags=cv2.SOLVEPNP_ITERATIVE,
)

if not success or pnp_inliers is None:
    raise RuntimeError("Could not estimate frame 60 camera pose.")

# OpenCV returns rotation as a compact rotation vector (rvec).
# Convert it into a 3x3 rotation matrix so it can be used with our other poses.
R60, _ = cv2.Rodrigues(rvec)

print(f"PnP inliers: {len(pnp_inliers)}")

print()
print("Rotation R60:")
print(R60)

print()
print("Translation t60:")
print(t60)


# ------------------------------------------------------------
# 6. Triangulate new 3-D points using frames 30 and 60
# ------------------------------------------------------------
# Now that we know the poses of cameras 30 and 60, we can estimate the 3-D
# position of a feature observed in both images.
#
# Imagine drawing a ray from each camera through the feature's image location.
# The rays ideally meet at the real-world point. Triangulation estimates that
# 3-D point from the two observations.
#
# These projection matrices use NORMALIZED image coordinates (not raw pixels).
# A projection matrix combines camera rotation and translation.
P30 = np.hstack((R30, t30))
P60 = np.hstack((R60, t60))

# Tracks that already have 3-D points should not be triangulated a second time.
existing_track_ids = set(point_track_ids.tolist())

# For each candidate new point, keep its track ID and the feature index in
# frames 30 and 60. The arrays must remain aligned row by row.
new_track_ids = []
new_features_30 = []
new_features_60 = []

for track_id, track in track_lookup.items():

    if track_id in existing_track_ids:
        continue

    observations = dict(track)

    # We need the same feature track to appear in both images to triangulate it.
    if 30 not in observations or 60 not in observations:
        continue

    new_track_ids.append(track_id)
    new_features_30.append(observations[30])
    new_features_60.append(observations[60])


new_track_ids = np.asarray(new_track_ids, dtype=np.int32)
new_features_30 = np.asarray(new_features_30, dtype=np.int32)
new_features_60 = np.asarray(new_features_60, dtype=np.int32)

print()
print("New point triangulation")
print(f"Candidate tracks: {len(new_track_ids)}")

# Start with an empty array in case no new tracks are available.
new_points_3d = np.empty((0, 3), dtype=np.float64)

if len(new_track_ids) > 0:

    # Look up the pixel coordinates of each candidate feature in both images.
    pixels_30 = keypoints_30[new_features_30, :2]
    pixels_60 = keypoints_60[new_features_60, :2]

    # Remove lens distortion and convert pixels to normalized camera coordinates.
    # Normalized coordinates express image position relative to the camera's
    # optical center and focal length, so the projection matrices above can be used.
    normalized_30 = cv2.undistortPoints(
        pixels_30.reshape(-1, 1, 2),
        K,
        distortion,
    ).reshape(-1, 2)

    normalized_60 = cv2.undistortPoints(
        pixels_60.reshape(-1, 1, 2),
        K,
        distortion,
    ).reshape(-1, 2)

    # OpenCV returns each triangulated point in homogeneous coordinates [X, Y, Z, W].
    # To get ordinary 3-D coordinates, divide X, Y and Z by W.
    points_4d = cv2.triangulatePoints(
        P30,
        P60,
        normalized_30.T,
        normalized_60.T,
    )

    # Division by W is unsafe if W is zero or the result contains invalid numbers.
    # Mark only columns with finite values and a non-negligible W as usable.
    w = points_4d[3]

    valid_homogeneous = (
        np.isfinite(points_4d).all(axis=0)
        & (np.abs(w) > 1e-10)
    )

    # Convert valid homogeneous results into ordinary [X, Y, Z] coordinates.
    new_points_3d = (
        points_4d[:3, valid_homogeneous]
        / w[valid_homogeneous]
    ).T

    # Keep the track IDs in exactly the same order as the points we retained.
    new_track_ids = new_track_ids[valid_homogeneous]

    # A real visible point should be in front of both cameras, not behind them.
    # Transform each point into each camera's coordinate system; its Z coordinate
    # is depth. Positive depth means it is in front of that camera.
    points_camera30 = (
        R30 @ new_points_3d.T + t30
    ).T

    points_camera60 = (
        R60 @ new_points_3d.T + t60
    ).T

    valid_depth = (
        (points_camera30[:, 2] > 0)
        & (points_camera60[:, 2] > 0)
    )

    # Discard points behind either camera, keeping IDs aligned with points.
    new_points_3d = new_points_3d[valid_depth]
    new_track_ids = new_track_ids[valid_depth]

    print(f"New 3-D points triangulated: {len(new_points_3d)}")

else:
    print("No new tracks available for triangulation.")


# ------------------------------------------------------------
# 7. Add the newly triangulated points to the existing 3-D point cloud
# ------------------------------------------------------------
# Stack the new rows under the old rows. This expands our sparse point cloud.
points_3d = np.vstack((
    points_3d,
    new_points_3d,
))

# Store the matching track ID for every point, in the same order.
point_track_ids = np.concatenate((
    point_track_ids,
    new_track_ids,
))

print()
print(f"Total reconstructed 3-D points: {len(points_3d)}")


# ------------------------------------------------------------
# 8. Record which image feature belongs to each 3-D point in frames 0, 30, 60
# ------------------------------------------------------------
# Later steps need to know which 2-D observations go with each 3-D point.
# For every reconstructed point, save the feature index in each registered frame.
# If the track was not observed in a particular frame, store -1 as "not present".
feature_indices_0 = []
feature_indices_30 = []
feature_indices_60 = []

for track_id in point_track_ids:

    observations = dict(track_lookup[int(track_id)])

    feature_indices_0.append(observations.get(0, -1))
    feature_indices_30.append(observations.get(30, -1))
    feature_indices_60.append(observations.get(60, -1))


feature_indices_0 = np.asarray(feature_indices_0, dtype=np.int32)
feature_indices_30 = np.asarray(feature_indices_30, dtype=np.int32)
feature_indices_60 = np.asarray(feature_indices_60, dtype=np.int32)


# ------------------------------------------------------------
# 9. Save the expanded reconstruction for the next pipeline step
# ------------------------------------------------------------
# Save the camera calibration, all three camera poses, the 3-D points, and the
# links between each 3-D point and its image observations. The next step can load
# this file instead of starting again from scratch.
output_file = "05_incremental_reconstruction.npz"

np.savez_compressed(
    output_file,

    # Camera calibration
    K=K,
    distortion=distortion,

    # Frames whose camera poses have been estimated so far
    registered_frame_ids=np.array([0, 30, 60], dtype=np.int32),

    # Each camera pose is represented by a rotation matrix and translation vector
    R0=R0,
    t0=t0,

    R30=R30,
    t30=t30,

    R60=R60,
    t60=t60,

    # Sparse 3-D reconstruction and the track ID associated with each point
    points_3d=points_3d,
    point_track_ids=point_track_ids,

    # Feature indices link each 3-D point to a detected feature in each frame
    feature_indices_0=feature_indices_0,
    feature_indices_30=feature_indices_30,
    feature_indices_60=feature_indices_60,
)

print()
print(f"Saved reconstruction: {output_file}")
print("Registered cameras: 3")
