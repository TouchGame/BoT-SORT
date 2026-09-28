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
        self.last_positions: Dict[str, Tuple[float, float, int]] = {}

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
            gallery = self.galleries[employee_id]

            # Dedup: if a similar feature exists, refresh its timestamp instead of adding
            if len(gallery) > 0:
                gallery_features = np.stack([g[0] for g in gallery])
                distances = self._cosine_distance(feat, gallery_features)
                min_idx = int(np.argmin(distances))
                if 1.0 - float(distances[min_idx]) > self.config.DEDUP_SIMILARITY_THRESHOLD:
                    gallery[min_idx] = (gallery[min_idx][0], frame_id)
                    continue

            gallery.append((feat, frame_id))
            added_count += 1

            if len(gallery) > self.config.MAX_GALLERY_SIZE:
                gallery.pop(0)

        if added_count > 0:
            self.metadata[employee_id]["feature_count"] = len(self.galleries[employee_id])

        return added_count

    @staticmethod
    def foot_point(tlbr) -> Tuple[float, float]:
        """Bottom-center of a tlbr box — the person's ground contact point."""
        return ((float(tlbr[0]) + float(tlbr[2])) / 2.0, float(tlbr[3]))

    def update_position(self, employee_id: str, foot_xy: Tuple[float, float], frame_id: int) -> None:
        """Record the employee's latest foot point and the frame it was seen at."""
        self.last_positions[employee_id] = (float(foot_xy[0]), float(foot_xy[1]), int(frame_id))

    def position_bonus(self, employee_id: str, query_pos: Optional[Tuple[float, float]],
                       query_frame: Optional[int] = None) -> float:
        """Position term subtracted from the distance: positive discounts near candidates, negative penalizes far ones.

        The radius scales with the employee's absence — it is the distance they could
        have walked since last seen: R = clamp(POSITION_SPEED * (query_frame - last_seen), MIN, MAX).

        d <= R:      +POSITION_BONUS * (1 - d / R)
        R < d < 2R:  -POSITION_PENALTY * (d - R) / R
        d >= 2R:     -POSITION_PENALTY
        """
        if query_pos is None or query_frame is None or employee_id not in self.last_positions:
            return 0.0

        last_x, last_y, last_frame = self.last_positions[employee_id]
        absent = max(0, int(query_frame) - last_frame)
        radius = min(max(self.config.POSITION_SPEED * absent,
                         self.config.POSITION_RADIUS_MIN),
                     self.config.POSITION_RADIUS_MAX)
        dist = float(np.hypot(last_x - query_pos[0], last_y - query_pos[1]))
        if dist <= radius:
            return self.config.POSITION_BONUS * (1.0 - dist / radius)
        ramp = min(1.0, (dist - radius) / radius)
        return -self.config.POSITION_PENALTY * ramp

    def identify(self, feature: np.ndarray, exclude_ids: Optional[set] = None, current_frame: int = 0,
                 query_pos: Optional[Tuple[float, float]] = None,
                 query_frame: Optional[int] = None) -> Tuple[Optional[str], float, int]:
        """
        Match a feature against all employee galleries with time weighting.

        Args:
            feature: Feature vector of shape (2048,)
            exclude_ids: Optional set of employee_ids to skip (e.g., those with active tracks)
            current_frame: Current frame number for time weighting
            query_pos: Optional foot point of the query track; candidates last seen near
                it get a distance discount, far ones a penalty (subtracted from distance)
            query_frame: Optional frame at which query_pos was observed (defaults to current_frame);
                drives the absence-scaled radius of the position term

        Returns:
            Tuple of (employee_id, weighted_distance, best_feature_index) or (None, inf, -1) if no match
        """
        if feature.ndim != 1:
            feature = feature.flatten()

        if query_frame is None:
            query_frame = current_frame

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
            time_weights = np.maximum(np.exp(-self.config.TIME_WEIGHT_DECAY * age), self.config.MIN_TIME_WEIGHT)
            weighted_distances = distances / time_weights - self.position_bonus(employee_id, query_pos, query_frame)

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

    def remove_employee(self, employee_id: str) -> None:
        """Remove an employee entirely — gallery, metadata and position record."""
        self.galleries.pop(employee_id, None)
        self.metadata.pop(employee_id, None)
        self.last_positions.pop(employee_id, None)

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

    def _cosine_distance(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a_norm = a / (np.linalg.norm(a) + 1e-8)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)

        similarities = np.dot(b_norm, a_norm)
        distances = 1 - similarities

        return distances
