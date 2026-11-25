import numpy as np
import cv2

# ------------------------------------------------------------
# Load tracks
# ------------------------------------------------------------

data = np.load("03_tracks.npz", allow_pickle=True)

track_keys = sorted(
    data.files,
    key=lambda key: int(key.removeprefix("track_"))
)

tracks = [
    data[key].tolist()
    for key in track_keys
]

print(f"Total tracks: {len(tracks)}")


# ------------------------------------------------------------
# Find tracks observed in both frame 0 and frame 30
# Preserve track IDs and feature indices.
# ------------------------------------------------------------

correspondences = []
point_track_ids = []

for track_id, track in enumerate(tracks):
    observations = dict(track)

    if 0 in observations and 30 in observations:
        feature_0 = observations[0]
        feature_30 = observations[30]

        correspondences.append((feature_0, feature_30))
        point_track_ids.append(track_id)

point_track_ids = np.asarray(point_track_ids, dtype=np.int32)

print()
print(
    f"Frame 0 ↔ Frame 30 correspondences: "
    f"{len(correspondences)}"
)


# ------------------------------------------------------------
# Load keypoints
# ------------------------------------------------------------

features = np.load("01_features.npz")

keypoints_0 = features["keypoints_0"]
keypoints_30 = features["keypoints_30"]


# ------------------------------------------------------------
# Convert feature indices to pixel coordinates
# ------------------------------------------------------------

feature_indices_0 = np.asarray(
    [feature_0 for feature_0, feature_30 in correspondences],
    dtype=np.int32,
)

feature_indices_30 = np.asarray(
    [feature_30 for feature_0, feature_30 in correspondences],
    dtype=np.int32,
)

points_0 = keypoints_0[feature_indices_0, :2].astype(np.float32)
points_30 = keypoints_30[feature_indices_30, :2].astype(np.float32)

print()
print(f"points_0 shape:  {points_0.shape}")
print(f"points_30 shape: {points_30.shape}")


# ------------------------------------------------------------
# Load camera calibration
# ------------------------------------------------------------

calibration = np.load("camera_calibration.npz")

K = calibration["K"]
distortion = calibration["distortion"]


# ------------------------------------------------------------
# Estimate Fundamental Matrix
# ------------------------------------------------------------

F, fundamental_mask = cv2.findFundamentalMat(
    points_0,
    points_30,
    cv2.FM_RANSAC,
    1.0,
    0.99,
)

if F is None or fundamental_mask is None:
    raise RuntimeError("Fundamental Matrix estimation failed.")

fundamental_mask = fundamental_mask.ravel().astype(bool)

print()
print("Fundamental Matrix:")
print(F)

print(f"Initial correspondences: {len(points_0)}")
print(f"Fundamental Matrix inliers: {np.sum(fundamental_mask)}")


# ------------------------------------------------------------
# Undistort points into normalized camera coordinates
# ------------------------------------------------------------

normalized_0 = cv2.undistortPoints(
    points_0.reshape(-1, 1, 2),
    K,
    distortion,
).reshape(-1, 2)

normalized_30 = cv2.undistortPoints(
    points_30.reshape(-1, 1, 2),
    K,
    distortion,
).reshape(-1, 2)


# ------------------------------------------------------------
# Estimate Essential Matrix
# ------------------------------------------------------------

E, essential_mask = cv2.findEssentialMat(
    normalized_0,
    normalized_30,
    np.eye(3),
    method=cv2.RANSAC,
    prob=0.999,
    threshold=1.0 / K[0, 0],
)

if E is None or essential_mask is None:
    raise RuntimeError("Essential Matrix estimation failed.")

essential_mask = essential_mask.ravel().astype(bool)

essential_points_0 = normalized_0[essential_mask]
essential_points_30 = normalized_30[essential_mask]

print()
print("Essential Matrix E:")
print(E)

print(f"Essential Matrix inliers: {len(essential_points_0)}")


# ------------------------------------------------------------
# Recover relative camera pose
# ------------------------------------------------------------

pose_inliers, R, t, pose_mask = cv2.recoverPose(
    E,
    essential_points_0,
    essential_points_30,
    np.eye(3),
)

print()
print("Relative rotation R:")
print(R)

print()
print("Relative translation direction t:")
print(t)

print(f"Pose inliers after cheirality check: {pose_inliers}")


# ------------------------------------------------------------
# Triangulate initial 3-D points
# ------------------------------------------------------------

# Camera 0 defines the world coordinate system.
P0 = np.hstack((
    np.eye(3),
    np.zeros((3, 1)),
))

# Camera 30 pose relative to Camera 0.
P30 = np.hstack((
    R,
    t,
))

pose_mask = pose_mask.ravel().astype(bool)

# Select the observations that survived both filtering stages.
triangulation_points_0 = essential_points_0[pose_mask]
triangulation_points_30 = essential_points_30[pose_mask]

points_4d = cv2.triangulatePoints(
    P0,
    P30,
    triangulation_points_0.T,
    triangulation_points_30.T,
)

# Convert homogeneous coordinates to ordinary 3-D coordinates.
points_3d = (
    points_4d[:3] / points_4d[3]
).T

print()
print("Projection matrix P0:")
print(P0)

print()
print("Projection matrix P30:")
print(P30)

print()
print(f"Triangulated points: {len(points_3d)}")
print(f"3-D point array shape: {points_3d.shape}")

print()
print("First 10 3-D points:")
print(points_3d[:10])


# ------------------------------------------------------------
# Preserve feature associations for triangulated points
# ------------------------------------------------------------

# essential_mask selects the Essential Matrix inliers from the
# original correspondences.
#
# pose_mask selects the pose inliers from those Essential Matrix
# inliers. Apply the same two masks to all association arrays.

initial_track_ids = point_track_ids[essential_mask][pose_mask]

initial_feature_indices_0 = (
    feature_indices_0[essential_mask][pose_mask]
)

initial_feature_indices_30 = (
    feature_indices_30[essential_mask][pose_mask]
)


# ------------------------------------------------------------
# Define the initial camera poses
# ------------------------------------------------------------

# Camera 0 is the reference camera.
R0 = np.eye(3, dtype=np.float64)
t0 = np.zeros((3, 1), dtype=np.float64)

# R and t describe Camera 30 relative to Camera 0.
R30 = R.copy()
t30 = t.copy()


# ------------------------------------------------------------
# Save initial reconstruction for incremental SfM
# ------------------------------------------------------------

output_file = "04_initial_reconstruction.npz"

np.savez_compressed(
    output_file,

    # Camera calibration
    K=K,
    distortion=distortion,

    # Initial registered cameras
    registered_frame_ids=np.array([0, 30], dtype=np.int32),

    R0=R0,
    t0=t0,

    R30=R30,
    t30=t30,

    # Initial sparse 3-D reconstruction
    points_3d=points_3d,

    # Track identity for each reconstructed 3-D point
    point_track_ids=initial_track_ids,

    # SIFT feature indices associated with each 3-D point
    feature_indices_0=initial_feature_indices_0,
    feature_indices_30=initial_feature_indices_30,
)

print()
print(f"Saved initial reconstruction: {output_file}")
print(f"Saved 3-D points: {len(points_3d)}")
print(f"Saved track associations: {len(initial_track_ids)}")
print(f"Registered cameras: {len([0, 30])}")