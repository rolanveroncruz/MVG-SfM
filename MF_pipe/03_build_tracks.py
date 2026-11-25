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
# ------------------------------------------------------------

pair_keys = [
    key
    for key in data.files
    if key.startswith("matches_")
]

pair_keys.sort(
    key=lambda key: int(key.split("_")[1])
)

print()
print("Match pairs:")

for key in pair_keys:
    print(" ", key)

# ------------------------------------------------------------
# Build feature tracks
# ------------------------------------------------------------

tracks = []
first_pair = pair_keys[0]
# Maps (frame_index, keypoint_index) → track index
observation_to_track = {}

frame1, frame2 = map(
    int,
    first_pair.removeprefix("matches_").split("_"),
)

matches = data[first_pair]

for query_idx, train_idx, distance in matches:

    track = [
        (frame1, int(query_idx)),
        (frame2, int(train_idx)),
    ]

    track_index = len(tracks)
    tracks.append(track)

    observation_to_track[(frame1, query_idx)] = track_index
    observation_to_track[(frame2, train_idx)] = track_index

print()
print(f"Initial tracks: {len(tracks)}")

# ------------------------------------------------------------
# Extend tracks through remaining frame pairs
# ------------------------------------------------------------

for pair_key in pair_keys[1:]:

    frame1, frame2 = map(
        int,
        pair_key.removeprefix("matches_").split("_"),
    )

    matches = data[pair_key]

    for query_idx, train_idx, distance in matches:

        query_idx = int(query_idx)
        train_idx = int(train_idx)

        existing_track = observation_to_track.get(
            (frame1, query_idx)
        )

        if existing_track is not None:

            tracks[existing_track].append(
                (frame2, train_idx)
            )

            observation_to_track[
                (frame2, train_idx)
            ] = existing_track

        else:

            track_index = len(tracks)

            tracks.append([
                (frame1, query_idx),
                (frame2, train_idx),
            ])

            observation_to_track[
                (frame1, query_idx)
            ] = track_index

            observation_to_track[
                (frame2, train_idx)
            ] = track_index


# ------------------------------------------------------------
# Check for duplicate observations
# ------------------------------------------------------------

observation_counts = {}

for track in tracks:
    for observation in track:
        observation_counts[observation] = (
            observation_counts.get(observation, 0) + 1
        )

duplicates = {
    observation: count
    for observation, count in observation_counts.items()
    if count > 1
}

print()
print(f"Duplicate observations: {len(duplicates)}")

if duplicates:
    for observation, count in list(duplicates.items())[:10]:
        print(f"  {observation}: appears in {count} tracks")

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

print()
print("First 10 tracks:")

for i, track in enumerate(tracks[:10]):
    print(f"Track {i}: {track}")


# ------------------------------------------------------------
# Save tracks
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