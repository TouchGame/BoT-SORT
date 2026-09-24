"""
Employee Registry

Manages employee identity through feature galleries.
Handles feature collection, deduplication, and identity matching.
In-memory only — each run starts with a fresh registry.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import EmployeeRegistryConfig


class EmployeeRegistry:
    """
    Manages employee identity through in-memory feature galleries.

    Each employee has a gallery of feature vectors collected from high-quality
    tracking sequences. New detections are matched against galleries to assign
    or recover employee identities.
    """

    def __init__(self, config: Optional[EmployeeRegistryConfig] = None):
        self.config = config or EmployeeRegistryConfig()
        self.galleries: Dict[str, List[Tuple[np.ndarray, int]]] = defaultdict(list)
        self.metadata: Dict[str, Dict] = {}

    def add_employee(self, employee_id: str, name: str = "") -> None:
        if employee_id in self.galleries:
            return

        self.galleries[employee_id] = []
        self.metadata[employee_id] = {
            "name": name,
            "feature_count": 0
        }

    def add_features(self, employee_id: str, features: np.ndarray, frame_id: int = 0) -> int:
        """
        Add features to an employee's gallery with deduplication.

        Args:
            employee_id: Employee identifier
            features: Array of shape (N, 2048) or (2048,)
            frame_id: Current frame number for time weighting

        Returns:
            Number of features actually added (after deduplication)
        """
        if employee_id not in self.galleries:
            self.add_employee(employee_id)

        if features.ndim == 1:
            features = features.reshape(1, -1)

        added_count = 0

        for feat in features:
            if self._is_duplicate(employee_id, feat):
                continue

            self.galleries[employee_id].append((feat, frame_id))
            added_count += 1

            if len(self.galleries[employee_id]) > self.config.MAX_GALLERY_SIZE:
                self.galleries[employee_id].pop(0)

        if added_count > 0:
            self.metadata[employee_id]["feature_count"] = len(self.galleries[employee_id])

        return added_count

    def identify(self, feature: np.ndarray, exclude_ids: Optional[set] = None, current_frame: int = 0) -> Tuple[Optional[str], float, int]:
        """
        Match a feature against all employee galleries with time weighting.

        Args:
            feature: Feature vector of shape (2048,)
            exclude_ids: Optional set of employee_ids to skip (e.g., those with active tracks)
            current_frame: Current frame number for time weighting

        Returns:
            Tuple of (employee_id, weighted_distance, best_feature_index) or (None, inf, -1) if no match
        """
        if feature.ndim != 1:
            feature = feature.flatten()

        best_id = None
        best_dist = float('inf')
        best_idx = -1

        for employee_id, gallery in self.galleries.items():
            if exclude_ids and employee_id in exclude_ids:
                continue

            if len(gallery) < self.config.MIN_GALLERY_SIZE:
                continue

            gallery_features = np.stack([g[0] for g in gallery])
            gallery_frames = np.array([g[1] for g in gallery])
            distances = self._cosine_distance(feature, gallery_features)

            age = current_frame - gallery_frames
            time_weights = np.exp(-self.config.TIME_WEIGHT_DECAY * age)
            weighted_distances = distances / time_weights

            min_idx = np.argmin(weighted_distances)
            min_dist = weighted_distances[min_idx]

            if min_dist < best_dist:
                best_dist = min_dist
                best_id = employee_id
                best_idx = int(min_idx)

        if best_dist < self.config.MATCH_THRESHOLD:
            return best_id, best_dist, best_idx

        return None, best_dist, -1

    def get_employee_info(self, employee_id: str) -> Optional[Dict]:
        return self.metadata.get(employee_id)

    def list_employees(self) -> List[str]:
        return list(self.galleries.keys())

    def clear_gallery(self, employee_id: str) -> None:
        if employee_id in self.galleries:
            self.galleries[employee_id] = []
            self.metadata[employee_id]["feature_count"] = 0

    def refresh_feature(self, employee_id: str, feature_idx: int, new_feature: np.ndarray, frame_id: int) -> None:
        """Update a matched feature's vector and refresh its timestamp."""
        if employee_id in self.galleries and 0 <= feature_idx < len(self.galleries[employee_id]):
            self.galleries[employee_id][feature_idx] = (new_feature, frame_id)

    def prune_gallery(self, employee_id: str, feature: np.ndarray) -> int:
        """
        Remove gallery features that are too distant from the given feature.
        Keeps features with distance <= PRUNE_DISTANCE_THRESHOLD.

        Returns:
            Number of features removed
        """
        if feature.ndim != 1:
            feature = feature.flatten()

        gallery = self.galleries.get(employee_id, [])
        if len(gallery) <= 1:
            return 0

        gallery_features = np.stack([g[0] for g in gallery])
        distances = self._cosine_distance(feature, gallery_features)

        kept = [(f, fid) for (f, fid), d in zip(gallery, distances) if d <= self.config.PRUNE_DISTANCE_THRESHOLD]
        removed = len(gallery) - len(kept)

        if removed > 0:
            self.galleries[employee_id] = kept
            self.metadata[employee_id]["feature_count"] = len(kept)

        return removed

    def _is_duplicate(self, employee_id: str, feature: np.ndarray) -> bool:
        if len(self.galleries[employee_id]) == 0:
            return False

        gallery_features = np.stack([g[0] for g in self.galleries[employee_id]])
        distances = self._cosine_distance(feature, gallery_features)
        min_dist = np.min(distances)

        similarity = 1 - min_dist
        return similarity > self.config.DEDUP_SIMILARITY_THRESHOLD

    def _cosine_distance(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a_norm = a / (np.linalg.norm(a) + 1e-8)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)

        similarities = np.dot(b_norm, a_norm)
        distances = 1 - similarities

        return distances
