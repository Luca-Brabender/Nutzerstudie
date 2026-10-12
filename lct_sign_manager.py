import carla

# Sequenz 1 (Track A - Baseline)
TRACK_SEQ_A = [
    'Right', 'Left', 'Middle', 'Middle', 'Left', 'Right',
    'Middle', 'Right', 'Left', 'Left', 'Middle', 'Left',
    'Right', 'Left', 'Middle', 'Middle', 'Right', 'Middle'
]

# Sequenz 2 (Track B - DualTask 1)
TRACK_SEQ_B = [
    'Left', 'Middle', 'Right', 'Left', 'Right', 'Middle',
    'Left', 'Left', 'Middle', 'Right', 'Middle', 'Right',
    'Left', 'Right', 'Middle', 'Left', 'Middle', 'Right'
]

# Sequenz 3 (Track C - DualTask 2)
TRACK_SEQ_C = [
    'Middle', 'Right', 'Left', 'Right', 'Middle', 'Left',
    'Right', 'Middle', 'Left', 'Middle', 'Right', 'Left',
    'Left', 'Middle', 'Right', 'Right', 'Left', 'Middle'
]

# Sequenz 4 (Track D - DualTask 3)
TRACK_SEQ_D = [
    'Right', 'Middle', 'Left', 'Middle', 'Right', 'Left',
    'Middle', 'Left', 'Right', 'Right', 'Middle', 'Left',
    'Right', 'Left', 'Middle', 'Left', 'Right', 'Middle'
]

# Mapping auf die 4 Maps
ALL_SEQUENCES = [TRACK_SEQ_A, TRACK_SEQ_B, TRACK_SEQ_C, TRACK_SEQ_D]
