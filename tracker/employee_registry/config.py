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

    # Gallery management
    MAX_GALLERY_SIZE = 10              # Maximum features per employee (FIFO eviction)
    TIME_WEIGHT_DECAY = 0.002          # Exponential decay for time weighting (per frame)

    # Post-match pruning
    PRUNE_DISTANCE_THRESHOLD = 0.6     # Remove gallery features with distance > this after match

    # Model paths (to be set from args)
    CONFIG_FILE = ""
    WEIGHTS_PATH = ""
    DEVICE = "cuda"
