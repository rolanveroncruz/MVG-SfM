"""
Step 03: Build feature tracks
=============================

WHAT IS A TRACK?
----------------
Step 02 gave us matches between *consecutive* frames only, e.g.

    frame 0  <-> frame 1   : keypoint 17 matches keypoint 52
    frame 1  <-> frame 2   : keypoint 52 matches keypoint 8
    frame 2  <-> frame 3   : keypoint 8  matches keypoint 91

Each of those is a separate pairwise statement. But clearly they all
describe the *same physical point* in the scene (say, the corner of a
window) seen in four different frames. A **track** chains these pairwise
matches together into a single record:

    track = [(frame 0, kp 17), (frame 1, kp 52), (frame 2, kp 8), (frame 3, kp 91)]

Each (frame, keypoint_index) entry is called an **observation**: "this
3-D point was seen in this frame, at this keypoint".

WHY DO WE NEED TRACKS?
----------------------
Structure-from-Motion needs to know which image features correspond to
the same 3-D point across MANY views, not just two:

  * Step 04 uses tracks that appear in both frame 0 and frame 30 to
    estimate the relative camera pose and triangulate the first 3-D points.
  * Step 05 uses tracks to (a) find where already-reconstructed 3-D points
    appear in a new frame (for camera pose estimation via PnP), and
    (b) find new tracks to triangulate.

Longer tracks are more valuable: a point seen in many frames constrains
the cameras and the 3-D position much better than one seen in two.

HOW WE BUILD THEM
-----------------
We treat every observation (frame, keypoint) as a node in a graph, and
every match from step 02 as an edge connecting two nodes. A track is then
a *connected component* of that graph. We find the components with a
union-find (disjoint-set) structure, which correctly handles chains
(A-B, B-C  =>  A, B, C in one track) and merging of tracks that meet.

CLEANING
--------
A track is only usable if it is *consistent*: it must not contain two
different keypoints from the same frame (that would claim one 3-D point
is in two places in one image). Such tracks usually come from a wrong
match that glued two real tracks together. We drop them.

INPUT
-----
  02_matches.npz   keys "matches_<frameA>_<frameB>", each an (M, 3)
                   float32 array with columns: queryIdx, trainIdx,
                   Hamming distance. queryIdx indexes the keypoints of
                   frameA, trainIdx the keypoints of frameB.

OUTPUT
------
  03_tracks.npz    keys "track_0", "track_1", ... Each is an (N, 2) int32
                   array whose rows are (frame_index, keypoint_index),
                   sorted by frame. The numeric suffix is the track ID
                   that steps 04 and 05 rely on.
"""

import numpy as np


# ------------------------------------------------------------
# Load matches
# ------------------------------------------------------------

data = np.load("02_matches.npz")

print("Keys in 02_matches.npz:")

for key in data.files:
    print(" ", key)


# ------------------------------------------------------------
# Frame pairs
# Keys look like "matches_<frame1>_<frame2>". Parse them once into
# integers and sort numerically (not alphabetically, which would put
# "matches_10_..." before "matches_2_...").
# ------------------------------------------------------------

def parse_pair_key(key):
    """'matches_30_31' -> (30, 31)"""
    frame1, frame2 = key.removeprefix("matches_").split("_")
    return int(frame1), int(frame2)


pair_keys = sorted(
    (key for key in data.files if key.startswith("matches_")),
    key=parse_pair_key,
)

if not pair_keys:
    raise RuntimeError(
        "No 'matches_*' arrays found in 02_matches.npz. "
        "Re-run 02_match_features.py."
    )

print()
print("Match pairs:")

for key in pair_keys:
    print(" ", key)


# ------------------------------------------------------------
# Union-find (disjoint set) over observations
#
# An observation is a (frame_index, keypoint_index) tuple.
# Initially every observation is in its own group. Each match says
# "these two observations are the same 3-D point", so we merge their
# groups. At the end, each remaining group is one track.
# ------------------------------------------------------------

# parent[obs] points toward the group's representative ("root").
# A root is its own parent.
parent = {}


def find(observation):
    """Return the root of the observation's group (with path compression)."""
    # Register unseen observations as their own singleton group.
    if observation not in parent:
        parent[observation] = observation
        return observation

    # Walk up to the root.
    root = observation
    while parent[root] != root:
        root = parent[root]

    # Path compression: point everything on the path straight at the
    # root so later lookups are fast.
    while parent[observation] != root:
        parent[observation], observation = root, parent[observation]

    return root


def union(obs_a, obs_b):
    """Merge the groups containing the two observations."""
    root_a = find(obs_a)
    root_b = find(obs_b)

    if root_a != root_b:
        parent[root_b] = root_a


# ------------------------------------------------------------
# Link observations using every pairwise match from step 02
# ------------------------------------------------------------

total_matches = 0

for pair_key in pair_keys:

    frame1, frame2 = parse_pair_key(pair_key)

    matches = data[pair_key]

    # Step 02 stores indices as float32 (exact for integers < 2**24).
    # Convert to plain Python ints so dictionary keys are consistent
    # everywhere. Column 2 (Hamming distance) is not needed here.
    for query_idx, train_idx, _distance in matches:

        union(
            (frame1, int(query_idx)),
            (frame2, int(train_idx)),
        )

    total_matches += len(matches)

print()
print(f"Total pairwise matches: {total_matches}")
print(f"Total observations:     {len(parent)}")


# ------------------------------------------------------------
# Collect groups into tracks
# ------------------------------------------------------------

groups = {}

for observation in parent:
    groups.setdefault(find(observation), []).append(observation)

# Sort observations inside each track by frame, and sort the tracks
# themselves so track IDs are deterministic from run to run.
all_tracks = sorted(
    sorted(group) for group in groups.values()
)

print(f"Tracks before cleaning: {len(all_tracks)}")


# ------------------------------------------------------------
# Remove inconsistent tracks
# A valid track has at most one keypoint per frame.
# ------------------------------------------------------------

tracks = []
inconsistent = 0

for track in all_tracks:

    frames_in_track = [frame for frame, _keypoint in track]

    if len(frames_in_track) != len(set(frames_in_track)):
        inconsistent += 1
        continue

    tracks.append(track)

print()
print(f"Inconsistent tracks removed (same frame twice): {inconsistent}")
print(f"Tracks kept: {len(tracks)}")

if not tracks:
    raise RuntimeError("No valid tracks were built.")


# ------------------------------------------------------------
# Inspect tracks
# ------------------------------------------------------------

print()
print(f"Total tracks: {len(tracks)}")

track_lengths = [
    len(track)
    for track in tracks
]

print(
    f"Longest track: "
    f"{max(track_lengths)} frames"
)

print(
    f"Average track length: "
    f"{np.mean(track_lengths):.2f} frames"
)

# How many tracks have each length. Longer tracks are more useful for
# reconstruction, so a healthy run has a decent tail of long tracks.
print()
print("Track length histogram (length: count):")

lengths, counts = np.unique(track_lengths, return_counts=True)

for length, count in zip(lengths, counts):
    print(f"  {length:3d}: {count}")

print()
print("First 10 tracks:")

for i, track in enumerate(tracks[:10]):
    print(f"Track {i}: {track}")


# ------------------------------------------------------------
# Save tracks
# Same format as before so steps 04 and 05 are unaffected:
# "track_<id>" -> (N, 2) int32 array of (frame_index, keypoint_index).
# ------------------------------------------------------------

track_data = {
    f"track_{i}": np.array(track, dtype=np.int32)
    for i, track in enumerate(tracks)
}

np.savez(
    "03_tracks.npz",
    **track_data,
)

print()
print("Saved:")
print("  03_tracks.npz")
