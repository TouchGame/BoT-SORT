"""
Face Registry Configuration
"""


class FaceRegistryConfig:
    """Configuration for the face registry module."""

    # Face feature extraction
    FEATURE_DIM = 512              # ResNet50 output dimension (buffalo_l)
    MIN_FACE_SIZE = 20             # Minimum face size in pixels
    MIN_DETECTION_SCORE = 0.5      # Minimum face detection confidence
    MIN_QUALITY_SCORE = 0.3        # Minimum face quality score

    # Feature deduplication
    DEDUP_SIMILARITY_THRESHOLD = 0.85  # 相似度 > 0.85 视为重复，新特征拒收
    SLOT_MIN_SIMILARITY = 0.5          # 相似度 < 0.5 视为捕获错误，拒绝进入非空槽位（与 MATCH_THRESHOLD 对齐）

    # Identity matching
    MATCH_THRESHOLD = 0.5
    MIN_GALLERY_SIZE = 1

    # Gallery management
    MAX_GALLERY_SIZE = 3

    # Pose-based slot management (3 slots: front, left, right)
    FRONT_MAX_YAW = 20.0           # 正脸：|yaw| <= 20°
    SIDE_MIN_YAW = 20.0            # 侧脸：20° < |yaw| < 45°
    MAX_YAW = 45.0                 # 超过45°拒绝
    SLOT_MAX_SIZE = 1              # 每个槽位最多1个特征

    # Post-match pruning
    PRUNE_DISTANCE_THRESHOLD = 0.5

    # insightface model config
    DET_SIZE = (640, 640)
    DET_SCORE = 0.5
    PROVIDERS = ['CPUExecutionProvider']  # CPU模式（RTX 5060需要CUDA 13.x，nightly暂不可用）
