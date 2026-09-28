"""
Face Registry

Manages employee identity through face feature galleries.
Handles feature collection, deduplication, and identity matching.
In-memory only — each run starts with a fresh registry.

Gallery structure: {employee_id: {'front': [(feat, frame_id)], 'left': [...], 'right': [...]}}
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import FaceRegistryConfig

logger = logging.getLogger(__name__)


class FaceRegistry:
    """
    Manages employee identity through in-memory face feature galleries.

    Each employee has a gallery of face feature vectors collected from
    high-quality face detections. Features are organized by pose (front/left/right)
    to ensure diversity and coverage.
    """

    def __init__(self, config: Optional[FaceRegistryConfig] = None):
        self.config = config or FaceRegistryConfig()
        # gallery structure: {employee_id: {'front': [(feat, frame_id)], 'left': [...], 'right': [...]}}
        self.galleries: Dict[str, Dict[str, List[Tuple[np.ndarray, int]]]] = defaultdict(
            lambda: {'front': [], 'left': [], 'right': []}
        )
        self.metadata: Dict[str, Dict] = {}

    def add_employee(self, employee_id: str, name: str = "") -> None:
        if employee_id in self.galleries:
            return

        self.galleries[employee_id] = {'front': [], 'left': [], 'right': []}
        self.metadata[employee_id] = {
            "name": name,
            "feature_count": 0
        }

    def add_features(self, employee_id: str, features: np.ndarray, frame_id: int = 0, pose_label: str = 'front', video_time: float = 0.0) -> int:
        """
        Add face features to an employee's gallery slot based on pose.

        Args:
            employee_id: Employee identifier
            features: Array of shape (N, 512) or (512,)
            frame_id: Current frame number for time weighting
            pose_label: 'front' | 'left' | 'right'
            video_time: Current video timestamp in seconds (for logging)

        Returns:
            Number of features actually added (after deduplication)
        """
        if employee_id not in self.galleries:
            self.add_employee(employee_id)

        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Get the slot for this pose
        if pose_label not in ['front', 'left', 'right']:
            logger.warning(f"Invalid pose_label: {pose_label}, using 'front'")
            pose_label = 'front'

        slot = self.galleries[employee_id][pose_label]
        added_count = 0

        for feat in features:
            if len(slot) > 0:
                slot_features = np.stack([g[0] for g in slot])
                distances = self._cosine_distance(feat, slot_features)
                min_similarity = 1.0 - float(np.min(distances))

                if min_similarity > self.config.DEDUP_SIMILARITY_THRESHOLD:
                    # Duplicate of existing reference — keep the old one
                    continue

                if min_similarity < self.config.SLOT_MIN_SIMILARITY:
                    # Too different from slot reference — likely a mis-captured face
                    logger.debug(f"[{video_time:.2f}s] FACE SLOT REJECT: employee={employee_id}, pose={pose_label}, "
                               f"similarity={min_similarity:.3f} < {self.config.SLOT_MIN_SIMILARITY} (likely mis-capture)")
                    continue

            # Replace the oldest feature if slot is full
            if len(slot) >= self.config.SLOT_MAX_SIZE:
                slot.pop(0)

            slot.append((feat, frame_id))
            added_count += 1

        if added_count > 0:
            total_count = sum(len(self.galleries[employee_id][p]) for p in ['front', 'left', 'right'])
            self.metadata[employee_id]["feature_count"] = total_count
            logger.debug(f"[{video_time:.2f}s] FACE ADD: employee={employee_id}, pose={pose_label}, added={added_count}, "
                       f"slot_size={len(slot)}, total={total_count}, frame={frame_id}")

        return added_count

    def identify(self, feature: np.ndarray, exclude_ids: Optional[set] = None, current_frame: int = 0, return_all: bool = False, face_size: Optional[float] = None) -> Tuple[Optional[str], float, int, Optional[List[Tuple[str, float]]]]:
        """
        Match a face feature against all employee galleries (all pose slots).

        Args:
            feature: Feature vector of shape (128,)
            exclude_ids: Optional set of employee_ids to skip
            current_frame: Current frame number for time weighting
            return_all: If True, also return distances to all employees
            face_size: Detected face min side in pixels; small faces use a relaxed threshold

        Returns:
            Tuple of (employee_id, weighted_distance, best_feature_index) or (None, inf, -1)
            If return_all=True, returns (employee_id, weighted_distance, best_feature_index, all_distances)
        """
        if feature.ndim != 1:
            feature = feature.flatten()

        best_id = None
        best_dist = float('inf')
        best_idx = -1
        all_distances = []

        for employee_id, pose_galleries in self.galleries.items():
            if exclude_ids and employee_id in exclude_ids:
                continue

            # Find minimum distance for this employee across all pose slots
            emp_best_dist = float('inf')
            emp_best_idx = -1

            for pose_label in ['front', 'left', 'right']:
                slot = pose_galleries[pose_label]
                if len(slot) < self.config.MIN_GALLERY_SIZE:
                    continue

                slot_features = np.stack([g[0] for g in slot])
                distances = self._cosine_distance(feature, slot_features)

                min_idx = np.argmin(distances)
                min_dist = distances[min_idx]

                if min_dist < emp_best_dist:
                    emp_best_dist = min_dist
                    emp_best_idx = int(min_idx)

            # Record distance for this employee if it has any features
            if emp_best_dist < float('inf'):
                all_distances.append((employee_id, float(emp_best_dist)))

                if emp_best_dist < best_dist:
                    best_dist = emp_best_dist
                    best_id = employee_id
                    best_idx = emp_best_idx

        all_distances.sort(key=lambda x: x[1])

        match_threshold = self._match_threshold(face_size)

        if best_dist < match_threshold:
            if return_all:
                return best_id, best_dist, best_idx, all_distances
            return best_id, best_dist, best_idx

        logger.debug(f"FACE NO MATCH: best_dist={best_dist:.4f} >= threshold={match_threshold}, "
                   f"best_id={best_id}, all_distances={[(e, f'{d:.4f}') for e, d in all_distances]}")

        if return_all:
            return None, best_dist, -1, all_distances
        return None, best_dist, -1

    def identify_best_match(self, face_buffer: List[Tuple[np.ndarray, int, float, str, float]], exclude_ids: Optional[set] = None, current_frame: int = 0) -> Tuple[Optional[str], float, int, Optional[np.ndarray], List[Tuple[str, float]]]:
        """
        Match multiple face features against all employee galleries.
        Returns the best match across all buffered features.

        Args:
            face_buffer: List of (feature, frame_id, quality_score, pose_label, face_size) tuples
            exclude_ids: Optional set of employee_ids to skip
            current_frame: Current frame number for time weighting

        Returns:
            Tuple of (employee_id, best_distance, gallery_feature_idx, best_face_feature, all_distances)
            best_face_feat is the matched face if a match exists, otherwise the
            highest-quality buffered face (so callers can seed an empty gallery).
        """
        if not face_buffer:
            return None, float('inf'), -1, None, []

        best_id = None
        best_dist = float('inf')
        best_idx = -1
        best_face_feat = None
        best_face_by_quality = max(face_buffer, key=lambda x: x[2])[0]
        all_distances_dict = {}  # {emp_id: min_dist}

        for face_feat, frame_id, quality_score, pose_label, face_size in face_buffer:
            emp_id, dist, idx, emp_distances = self.identify(face_feat, exclude_ids, current_frame, return_all=True, face_size=face_size)
            # Update all_distances with minimum distance per employee
            for e, d in emp_distances:
                if e not in all_distances_dict or d < all_distances_dict[e]:
                    all_distances_dict[e] = d
            # Prefer real matches over smaller raw distances: with per-size
            # thresholds a non-matching entry can be closer than a matching one
            is_match = emp_id is not None
            was_match = best_id is not None
            if (is_match and not was_match) or (is_match == was_match and dist < best_dist):
                best_dist = dist
                best_id = emp_id
                best_idx = idx
                best_face_feat = face_feat

        if best_face_feat is None:
            # No gallery match — return best quality buffered face so the
            # caller can still seed a gallery with it (inf < inf is False)
            best_face_feat = best_face_by_quality

        all_distances = sorted(all_distances_dict.items(), key=lambda x: x[1])
        return best_id, best_dist, best_idx, best_face_feat, all_distances

    def get_employee_info(self, employee_id: str) -> Optional[Dict]:
        return self.metadata.get(employee_id)

    def list_employees(self) -> List[str]:
        return list(self.galleries.keys())

    def clear_gallery(self, employee_id: str) -> None:
        if employee_id in self.galleries:
            self.galleries[employee_id] = {'front': [], 'left': [], 'right': []}
            self.metadata[employee_id]["feature_count"] = 0

    def refresh_feature(self, employee_id: str, feature_idx: int, new_feature: np.ndarray, frame_id: int, pose_label: str = 'front') -> None:
        """Update a matched feature's vector and refresh its timestamp."""
        if employee_id in self.galleries:
            slot = self.galleries[employee_id].get(pose_label, [])
            if 0 <= feature_idx < len(slot):
                slot[feature_idx] = (new_feature, frame_id)
                logger.debug(f"FACE REFRESH: employee={employee_id}, pose={pose_label}, idx={feature_idx}, frame={frame_id}")

    def prune_gallery(self, employee_id: str, feature: np.ndarray) -> int:
        """
        Remove gallery features that are too distant from the given feature.
        Operates on all pose slots.

        Returns:
            Number of features removed
        """
        if feature.ndim != 1:
            feature = feature.flatten()

        if employee_id not in self.galleries:
            return 0

        total_removed = 0
        for pose_label in ['front', 'left', 'right']:
            slot = self.galleries[employee_id][pose_label]
            if len(slot) <= 1:
                continue

            slot_features = np.stack([g[0] for g in slot])
            distances = self._cosine_distance(feature, slot_features)

            kept = [(f, fid) for (f, fid), d in zip(slot, distances) if d <= self.config.PRUNE_DISTANCE_THRESHOLD]
            removed = len(slot) - len(kept)
            total_removed += removed

            if removed > 0:
                self.galleries[employee_id][pose_label] = kept

        if total_removed > 0:
            total_count = sum(len(self.galleries[employee_id][p]) for p in ['front', 'left', 'right'])
            self.metadata[employee_id]["feature_count"] = total_count

        return total_removed

    def _match_threshold(self, face_size: Optional[float]) -> float:
        """Small faces have noisier features, so they get a relaxed threshold."""
        if face_size is not None and face_size < self.config.SMALL_FACE_SIZE:
            return self.config.SMALL_FACE_MATCH_THRESHOLD
        return self.config.MATCH_THRESHOLD

    def _cosine_distance(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a_norm = a / (np.linalg.norm(a) + 1e-8)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)

        similarities = np.dot(b_norm, a_norm)
        distances = 1 - similarities

        return distances
