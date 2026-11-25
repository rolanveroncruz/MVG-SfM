import cv2
import numpy as np


# ------------------------------------------------------------
# Load features
# ------------------------------------------------------------

data = np.load("01_features.npz")
print("Keys in 01_features.npz:")

for key in data.files:
    print(" ", key)

# ------------------------------------------------------------
# Frame indices
# ------------------------------------------------------------

frame_indices = sorted(
    int(key.removeprefix("descriptors_"))
    for key in data.files
    if key.startswith("descriptors_")
)
print()
print("Frames:")
print(frame_indices)
# ------------------------------------------------------------
# Load descriptors- Read the descriptors to create the map
# descriptors[frame_index] = descriptor
# A descriptor in ORB is  a 256-bit binary string stored as 32 unit8 bytes,
# It is a compact fingerprint of the small image patch around a keypoint.
# ------------------------------------------------------------

descriptors = {}

for frame_index in frame_indices:

    key = f"descriptors_{frame_index}"

    descriptors[frame_index] = data[key]

    print(
        f"Frame {frame_index}: "
        f"{descriptors[frame_index].shape[0]} descriptors"
    )
# ------------------------------------------------------------
# Match consecutive frames
# ------------------------------------------------------------

matcher = cv2.BFMatcher(cv2.NORM_L2)

match_data = {}

ratio_threshold = 0.7

for frame1, frame2 in zip(
    frame_indices[:-1],
    frame_indices[1:],
):

    desc1 = descriptors[frame1]
    desc2 = descriptors[frame2]

    # --------------------------------------------------------
    # Frame 1 → Frame 2
    # --------------------------------------------------------

    matches_12 = matcher.knnMatch(
        desc1,
        desc2,
        k=2,
    )

    good_12 = {}

    for m, n in matches_12:
        if m.distance < ratio_threshold * n.distance:
            good_12[m.queryIdx] = m

    # --------------------------------------------------------
    # Frame 2 → Frame 1
    # --------------------------------------------------------

    matches_21 = matcher.knnMatch(
        desc2,
        desc1,
        k=2,
    )

    good_21 = {}

    for m, n in matches_21:
        if m.distance < ratio_threshold * n.distance:
            good_21[m.queryIdx] = m

    # --------------------------------------------------------
    # Keep only mutual matches
    # --------------------------------------------------------

    mutual_matches = []

    for query_idx, match in good_12.items():

        train_idx = match.trainIdx

        reverse_match = good_21.get(train_idx)

        if reverse_match is not None:
            if reverse_match.trainIdx == query_idx:
                mutual_matches.append(match)

    # --------------------------------------------------------
    # Store matches
    # --------------------------------------------------------

    match_array = np.array([
        [
            m.queryIdx,
            m.trainIdx,
            m.distance,
        ]
        for m in mutual_matches
    ], dtype=np.float32)

    match_data[f"matches_{frame1}_{frame2}"] = match_array

    print(
        f"Frame {frame1} → Frame {frame2}: "
        f"{len(mutual_matches)} mutual good matches"
    )

# ------------------------------------------------------------
# Save matches
# ------------------------------------------------------------

np.savez(
    "02_matches.npz",
    **match_data,
)

print()
print("Saved:")
print("  02_matches.npz")
