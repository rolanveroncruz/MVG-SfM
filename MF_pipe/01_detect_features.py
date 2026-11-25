import cv2
import numpy as np


# ------------------------------------------------------------
# Open video
# ------------------------------------------------------------

video = cv2.VideoCapture("desktop.mp4")

if not video.isOpened():
    raise RuntimeError("Could not open video")

# ------------------------------------------------------------
# Frames to process- create an array of frames taking every frame_interval_th
# frame from the video.
# ------------------------------------------------------------

frame_interval = 30

frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

frame_indices = list(range(0, frame_count, frame_interval))

print(f"Video has {frame_count} frames; processing {len(frame_indices)}")

# ------------------------------------------------------------
# Feature detector - Create an ORB feature detector
#
# ORB = oriented FAST keypoints + rotated BRIEF descriptors.
# Descriptors are 256-bit binary strings (stored as 32 uint8 bytes),
# so they must be compared with Hamming distance (cv2.NORM_HAMMING),
# not L2 as with SIFT.
#
# nfeatures:     maximum keypoints kept per frame (ORB default is 500).
# scaleFactor:   ratio between pyramid levels.
# nlevels:       number of pyramid levels (gives scale invariance).
# fastThreshold: FAST corner threshold; lower finds more (weaker) corners.
# ------------------------------------------------------------

orb = cv2.ORB.create(
    nfeatures=2000,
    scaleFactor=1.2,
    nlevels=8,
    fastThreshold=20,
)


# ------------------------------------------------------------
# Detect features in selected frames- create a map from frame_index
# to an object of { "descriptors":descriptors,  "keypoints": keypoints}
# ------------------------------------------------------------

features = {}

for frame_index in frame_indices:

    video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    success, frame = video.read()

    if not success:
        raise RuntimeError(
            f"Could not read frame {frame_index}"
        )

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY,
    )

    keypoints, descriptors = orb.detectAndCompute(
        gray,
        None,
    )

    # ORB returns descriptors=None when it finds no keypoints
    # (e.g. a blank or very blurry frame). Later stages assume an
    # (N, 32) uint8 array, so fail loudly here instead.
    if descriptors is None or len(keypoints) == 0:
        raise RuntimeError(
            f"No ORB features found in frame {frame_index}"
        )

    features[frame_index] = {
        "keypoints": keypoints,
        "descriptors": descriptors,
    }

    print(
        f"Frame {frame_index}: "
        f"{len(keypoints)} keypoints"
    )
# ------------------------------------------------------------
# Save features to file.
# ------------------------------------------------------------

feature_data = {}

for frame_index, frame_features in features.items():

    keypoints = frame_features["keypoints"]
    descriptors = frame_features["descriptors"]

    # Columns: x, y, size, angle, response, octave, class_id
    # (class_id is always -1 for ORB; kept so the column layout
    # matches the earlier SIFT file and downstream indexing [:, :2]
    # keeps working.)
    keypoint_data = np.array([
        [
            kp.pt[0],
            kp.pt[1],
            kp.size,
            kp.angle,
            kp.response,
            kp.octave,
            kp.class_id,
        ]
        for kp in keypoints
    ], dtype=np.float32)

    # ORB descriptors are binary: shape (N, 32), dtype uint8.
    assert descriptors.dtype == np.uint8 and descriptors.shape[1] == 32

    feature_data[f"keypoints_{frame_index}"] = keypoint_data
    feature_data[f"descriptors_{frame_index}"] = descriptors

np.savez(
    "01_features.npz",
    **feature_data,
)

print()
print("Saved:")
print("  01_features.npz")


video.release()
