"""
Employee Registry Configuration
"""


class EmployeeRegistryConfig:
    """Configuration for the employee registry module."""

    # Feature extraction
    FEATURE_DIM = 2048
    BATCH_SIZE = 8

    # Feature collection criteria
    MIN_TRACK_LENGTH = 15          # Minimum frames before collecting features
    MIN_DETECTION_CONFIDENCE = 0.7 # Minimum detection score for feature collection
    MAX_OCCLUSION_RATIO = 0.3      # Maximum allowed occlusion ratio

    # Feature deduplication
    DEDUP_SIMILARITY_THRESHOLD = 0.85  # Features with cosine similarity > this are considered duplicates

    # Identity matching
    MATCH_THRESHOLD = 0.4              # Maximum distance to consider a match
    MIN_GALLERY_SIZE = 1               # Minimum features in gallery before matching

    # Position prior: near candidates get a distance discount, far ones a penalty.
    # The radius is the distance the employee could have walked while absent (speed x absence).
    POSITION_SPEED = 8.0               # Predicted movement speed in px/frame during absence (~200 px/s @25fps)
    POSITION_RADIUS_MIN = 100.0        # Radius floor (absorbs box jitter for very short absences)
    POSITION_RADIUS_MAX = 400.0        # Radius cap (beyond this the position carries no discriminative information)
    POSITION_BONUS = 0.08              # Max discount at zero distance (0 disables)
    POSITION_PENALTY = 0.08            # Max penalty reached at twice the radius (0 disables)

    # Gallery management
    MAX_GALLERY_SIZE = 4               # Maximum features per employee (FIFO eviction)
    TIME_WEIGHT_DECAY = 0.002          # Exponential decay for time weighting (per frame)
    MIN_TIME_WEIGHT = 0.7              # Lower bound for time weight; caps absence penalty at ~1.43x

    # Post-match pruning
    PRUNE_DISTANCE_THRESHOLD = 0.6     # Remove gallery features with distance > this after match

    # Re-activation identity verification
    REID_VERIFY_THRESHOLD = 0.45       # Check A: max distance between curr_feat and old smooth_feat
    REID_VERIFY_THRESHOLD_B = 0.5      # Check B: max distance between curr_feat and employee gallery

    # Model paths (to be set from args)
    CONFIG_FILE = ""
    WEIGHTS_PATH = ""
    DEVICE = "cuda"
