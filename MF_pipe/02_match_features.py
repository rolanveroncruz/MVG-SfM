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

    # Step 01 saves ORB descriptors: (N, 32) uint8. Catch a stale
    # SIFT-era file (float32, 128 columns) before matching silently
    # produces garbage.
    if (
        descriptors[frame_index].dtype != np.uint8
        or descriptors[frame_index].shape[1] != 32
    ):
        raise RuntimeError(
            f"Frame {frame_index}: expected ORB descriptors "
            f"(N, 32) uint8, got "
            f"{descriptors[frame_index].shape} "
            f"{descriptors[frame_index].dtype}. "
            f"Re-run 01_detect_features.py."
        )

    print(
        f"Frame {frame_index}: "
        f"{descriptors[frame_index].shape[0]} descriptors"
    )
# ------------------------------------------------------------
# Match consecutive frames - Given descriptors map which has frame_indices as keys
# and values as descriptors, take adjacent frames and find
# matching features using the ratio test plus a mutual (two-way) check.
# ***Ratio test*** (is the match unambiguous)- For a feature in frame A, find its two closest descriptors
# in frame B. The best (m) and the second best (n) math only if distance(m) < 0.75 * distance (n).
# If the best is much closer to the feature in question than the runner-up, the match is probably right.
# If nearly tied, match is discarded.
# *** Mutual Check *** - The ratio test only runs in one direction, so it can produce a match whre A picks  up B but B would pick something else.
# The Mutual Check runs both ways. So we only keep only if each frame's feature chooses the other.
# 
# ORB descriptors are binary, so distance is Hamming distance: the
# number of differing bits (0 = identical, 256 = opposite).
# ------------------------------------------------------------

# BFMatcher is BruteForce Matching
matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

match_data = {}

# Lowe's ratio test: keep a match only if the best candidate is clearly
# better than the second best. Hamming distances are coarse integers, so
# ORB usually needs a looser ratio than SIFT's 0.7. Typical range 0.75-0.8;
# lower = fewer but cleaner matches.
ratio_threshold = 0.75


def ratio_test_matches(query_desc, train_desc):
    """
    Match each query descriptor to its nearest train descriptor, keeping
    only matches that pass the ratio test.

    Returns a dict: queryIdx -> DMatch.
    """
    knn_matches = matcher.knnMatch(query_desc, train_desc, k=2)

    good = {}

    for pair in knn_matches:

        # knnMatch can return fewer than k results (e.g. when the train
        # set has only one descriptor). The ratio test needs two.
        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < ratio_threshold * n.distance:
            good[m.queryIdx] = m

    return good

for frame1, frame2 in zip(
    frame_indices[:-1],
    frame_indices[1:],
):

    desc1 = descriptors[frame1]
    desc2 = descriptors[frame2]

    # --------------------------------------------------------
    # Frame 1 → Frame 2
    # --------------------------------------------------------

    good_12 = ratio_test_matches(desc1, desc2)

    # --------------------------------------------------------
    # Frame 2 → Frame 1
    # --------------------------------------------------------

    good_21 = ratio_test_matches(desc2, desc1)

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

    # Columns: queryIdx, trainIdx, Hamming distance.
    # reshape keeps the (0, 3) shape when there are no matches, so
    # step 03 can still unpack rows without special-casing.
    #
    # Note: float32 represents integers exactly up to 2**24, far above
    # any keypoint index here, so storing indices as float32 is safe.
    # Step 03 converts them back with int().
    match_array = np.array([
        [
            m.queryIdx,
            m.trainIdx,
            m.distance,
        ]
        for m in mutual_matches
    ], dtype=np.float32).reshape(-1, 3)

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
