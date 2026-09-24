"""
Face Registry Configuration
"""


class FaceRegistryConfig:
    """Configuration for the face registry module."""

    # Face feature extraction
    FEATURE_DIM = 128              # MobileFaceNet output dimension
    MIN_FACE_SIZE = 40             # Minimum face size in pixels
    MIN_DETECTION_SCORE = 0.5      # Minimum face detection confidence
    MIN_QUALITY_SCORE = 0.3        # Minimum face quality score

    # Feature deduplication
    DEDUP_SIMILARITY_THRESHOLD = 0.85

    # Identity matching
    MATCH_THRESHOLD = 0.4
    MIN_GALLERY_SIZE = 1

    # Gallery management
    MAX_GALLERY_SIZE = 3
    TIME_WEIGHT_DECAY = 0.002

    # Post-match pruning
    PRUNE_DISTANCE_THRESHOLD = 0.5

    # insightface model config
    DET_SIZE = (640, 640)
    DET_SCORE = 0.5
    PROVIDERS = ['CPUExecutionProvider']  # CPU模式（RTX 5060需要CUDA 13.x，nightly暂不可用）
