"""G1 joint/body order maps — single source of truth.

The retargeted dataset (motion.npz) stores joints and bodies in IsaacLab BFS
order; mjlab/MuJoCo uses XML DFS order. The pretrained textop WBC consumes and
emits IL-ordered joint vectors.

Usage: ``mj_data = il_data[:, IL2MJ]`` and ``il_data = mj_data[:, MJ2IL]``.
"""

# fmt: off
IL2MJ = [
    0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18,
    2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28,
]
MJ2IL = [
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17,
    24, 18, 25, 19, 26, 20, 27, 21, 28,
]
# fmt: on

# 14 tracked bodies: (body_name, IL body index in motion.npz body arrays).
# IL indices verified by FK position-matching against MuJoCo.
# fmt: off
G1_TRACKED_BODIES: tuple[tuple[str, int], ...] = (
    ("pelvis",                   0),
    ("left_hip_roll_link",       5),
    ("left_knee_link",          11),
    ("left_ankle_roll_link",    21),
    ("right_hip_roll_link",      6),
    ("right_knee_link",         12),
    ("right_ankle_roll_link",   22),
    ("torso_link",              10),
    ("left_shoulder_roll_link", 19),
    ("left_elbow_link",         27),
    ("left_wrist_yaw_link",     33),
    ("right_shoulder_roll_link",20),
    ("right_elbow_link",        28),
    ("right_wrist_yaw_link",    34),
)
# fmt: on
G1_TRACKED_BODY_NAMES = tuple(name for name, _ in G1_TRACKED_BODIES)
