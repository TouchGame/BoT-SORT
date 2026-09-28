"""
Face Analysis Wrapper

Wraps insightface FaceAnalysis for face detection and feature extraction.
Uses SCRFD for detection and MobileFaceNet for feature extraction.
"""

import logging
from typing import Optional, Tuple

import numpy as np

from .config import FaceRegistryConfig

logger = logging.getLogger('bot_sort')


class FaceAnalysisWrapper:
    """
    Wrapper for insightface FaceAnalysis.

    Provides face detection within person bounding boxes and
    face feature extraction with quality filtering.
    """

    def __init__(self, config: Optional[FaceRegistryConfig] = None):
        self.config = config or FaceRegistryConfig()
        self.app = None
        self._init_model()

    def _init_model(self):
        """Initialize insightface FaceAnalysis model."""
        try:
            from insightface.app import FaceAnalysis
            self.app = FaceAnalysis(name='buffalo_l', providers=self.config.PROVIDERS)
            self.app.prepare(ctx_id=0, det_size=self.config.DET_SIZE, det_thresh=self.config.DET_SCORE)
            logger.info("FaceAnalysis model loaded successfully (buffalo_l)")
        except Exception as e:
            logger.error(f"Failed to load FaceAnalysis model: {e}")
            self.app = None

    def _calculate_yaw_angle(self, landmark: np.ndarray) -> float:
        """
        通过人脸关键点计算yaw角度（水平旋转）。
        只负责计算原始角度，分类由 _classify_pose 根据配置阈值判断。

        Args:
            landmark: 5个关键点 [[左眼x,左眼y], [右眼x,右眼y], [鼻尖x,鼻尖y], [左嘴角x,左嘴角y], [右嘴角x,右嘴角y]]

        Returns:
            yaw角度（度）：正=右转，负=左转，0=正脸
        """
        left_eye = landmark[0]
        right_eye = landmark[1]
        nose = landmark[2]

        eye_center = (left_eye + right_eye) / 2
        eye_distance = np.linalg.norm(right_eye - left_eye)
        if eye_distance < 1e-6:
            return 0.0

        nose_offset = nose[0] - eye_center[0]
        offset_ratio = nose_offset / eye_distance
        offset_ratio = np.clip(offset_ratio, -0.99, 0.99)
        yaw_angle = np.degrees(np.arcsin(offset_ratio))

        return yaw_angle

    def _classify_pose(self, yaw_angle: float) -> str:
        """
        根据yaw角度分类人脸姿态。

        Returns:
            'front' | 'left' | 'right' | 'unknown'
            - |yaw| <= FRONT_MAX_YAW → front
            - FRONT_MAX_YAW < |yaw| < MAX_YAW → left/right
            - |yaw| >= MAX_YAW → unknown (拒绝)
        """
        abs_yaw = abs(yaw_angle)
        if abs_yaw <= self.config.FRONT_MAX_YAW:
            return 'front'
        elif abs_yaw >= self.config.MAX_YAW:
            return 'unknown'
        elif yaw_angle < 0:
            return 'left'
        else:
            return 'right'

    def extract_face(self, img: np.ndarray, person_bbox: np.ndarray, video_time: float = 0.0) -> Optional[Tuple[np.ndarray, float, str, float]]:
        """
        Detect face within person bbox and extract face feature.

        Args:
            img: Original image (BGR format)
            person_bbox: Person bounding box [x1, y1, x2, y2]
            video_time: Current video timestamp in seconds (for logging)

        Returns:
            (face_feature, quality_score, pose_label, yaw_angle) or None if no valid face found
            pose_label: 'front' | 'left' | 'right'
        """
        if self.app is None:
            return None

        x1, y1, x2, y2 = person_bbox.astype(int)
        h, w = img.shape[:2]

        # Expand bbox by 20% upward to include full head
        box_h = y2 - y1
        expand_h = int(box_h * 0.2)
        y1_expanded = max(0, y1 - expand_h)

        # Clip to image bounds
        x1_clip = max(0, x1)
        y1_clip = max(0, y1_expanded)
        x2_clip = min(w, x2)
        y2_clip = min(h, y2)

        if x2_clip <= x1_clip or y2_clip <= y1_clip:
            return None

        # Crop person region
        person_crop = img[y1_clip:y2_clip, x1_clip:x2_clip].copy()

        if person_crop.size == 0:
            return None

        # Detect faces in the cropped region
        faces = self.app.get(person_crop)

        if len(faces) == 0:
            return None

        # Take the largest face
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

        # Quality filtering
        face_bbox = face.bbox
        face_w = face_bbox[2] - face_bbox[0]
        face_h = face_bbox[3] - face_bbox[1]

        if face_w < self.config.MIN_FACE_SIZE or face_h < self.config.MIN_FACE_SIZE:
            return None

        det_score = face.det_score
        if det_score < self.config.MIN_DETECTION_SCORE:
            return None

        # Calculate yaw angle and classify pose (use 5-point landmarks: left_eye, right_eye, nose, left_mouth, right_mouth)
        yaw_angle = self._calculate_yaw_angle(face.kps)
        pose_label = self._classify_pose(yaw_angle)

        # Skip unknown pose (yaw exceeds MAX_YAW)
        if pose_label == 'unknown':
            return None

        # Get face feature (512-dim for buffalo_l, L2-normalized by insightface)
        face_feat = face.embedding

        logger.debug(f"[{video_time:.2f}s] FACE EXTRACTED: face_size=({face_w:.0f}x{face_h:.0f}), "
                   f"det_score={det_score:.3f}, pose={pose_label}, yaw={yaw_angle:.1f}°")
        return face_feat, float(det_score), pose_label, float(yaw_angle)

    def is_available(self) -> bool:
        """Check if face analysis model is loaded."""
        return self.app is not None
