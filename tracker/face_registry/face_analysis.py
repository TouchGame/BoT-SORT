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
            self.app = FaceAnalysis(name='buffalo_s', providers=self.config.PROVIDERS)
            self.app.prepare(ctx_id=0, det_size=self.config.DET_SIZE, det_thresh=self.config.DET_SCORE)
            logger.info("FaceAnalysis model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load FaceAnalysis model: {e}")
            self.app = None

    def extract_face(self, img: np.ndarray, person_bbox: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
        """
        Detect face within person bbox and extract face feature.

        Args:
            img: Original image (BGR format)
            person_bbox: Person bounding box [x1, y1, x2, y2]

        Returns:
            (face_feature, quality_score) or None if no valid face found
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

        # Get face feature (128-dim, already normalized by insightface)
        face_feat = face.embedding

        return face_feat, float(det_score)

    def is_available(self) -> bool:
        """Check if face analysis model is loaded."""
        return self.app is not None
