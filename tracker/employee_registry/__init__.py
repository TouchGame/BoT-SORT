"""
Employee Registry Module

Provides persistent employee identity management through feature galleries.
"""

from .config import EmployeeRegistryConfig
from .feature_extractor import FeatureExtractor
from .registry import EmployeeRegistry

__all__ = [
    "EmployeeRegistryConfig",
    "FeatureExtractor",
    "EmployeeRegistry",
]
