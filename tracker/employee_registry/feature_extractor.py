"""
Feature Extractor for Employee Registry

Independent module for extracting appearance features from person detections.
Copied from fast_reid_interfece.py to allow independent evolution.
Future: can be replaced with face features or other modalities.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from fast_reid.fastreid.config import get_cfg
from fast_reid.fastreid.modeling.meta_arch import build_model
from fast_reid.fastreid.utils.checkpoint import Checkpointer


def setup_cfg(config_file, opts):
    """Load config from file and command-line arguments."""
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    cfg.MODEL.BACKBONE.PRETRAIN = False
    cfg.freeze()
    return cfg


def postprocess(features):
    """Normalize feature to compute cosine distance."""
    features = F.normalize(features)
    features = features.cpu().data.numpy()
    return features


class FeatureExtractor:
    """
    Extracts appearance features from person detection crops.

    This is an independent copy of FastReIDInterface, allowing the employee
    registry to evolve separately from the tracking ReID module.
    """

    def __init__(self, config_file, weights_path, device, batch_size=8):
        """
        Initialize the feature extractor.

        Args:
            config_file: Path to FastReID config file
            weights_path: Path to pretrained weights
            device: 'cuda' or 'cpu'
            batch_size: Batch size for inference
        """
        super(FeatureExtractor, self).__init__()

        if device != 'cpu':
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.batch_size = batch_size
        self.cfg = setup_cfg(config_file, ['MODEL.WEIGHTS', weights_path])

        self.model = build_model(self.cfg)
        self.model.eval()
        Checkpointer(self.model).load(weights_path)

        if self.device != 'cpu':
            self.model = self.model.eval().to(device='cuda').half()
        else:
            self.model = self.model.eval()

        self.pH, self.pW = self.cfg.INPUT.SIZE_TEST

    def extract(self, image, detections):
        """
        Extract features from detection crops.

        Args:
            image: Input image (H, W, 3) in BGR format
            detections: Numpy array of shape (N, 4+) with [x1, y1, x2, y2, ...]

        Returns:
            Numpy array of shape (N, 2048) with L2-normalized features
        """
        if detections is None or np.size(detections) == 0:
            return np.zeros((0, 2048))

        H, W, _ = np.shape(image)

        batch_patches = []
        patches = []

        for d in range(np.size(detections, 0)):
            tlbr = detections[d, :4].astype(np.int_)
            tlbr[0] = max(0, tlbr[0])
            tlbr[1] = max(0, tlbr[1])
            tlbr[2] = min(W - 1, tlbr[2])
            tlbr[3] = min(H - 1, tlbr[3])

            patch = image[tlbr[1]:tlbr[3], tlbr[0]:tlbr[2], :]
            patch = patch[:, :, ::-1]  # BGR to RGB
            patch = cv2.resize(patch, tuple(self.cfg.INPUT.SIZE_TEST[::-1]),
                             interpolation=cv2.INTER_LINEAR)

            patch = torch.as_tensor(patch.astype("float32").transpose(2, 0, 1))
            patch = patch.to(device=self.device).half()

            patches.append(patch)

            if (d + 1) % self.batch_size == 0:
                patches = torch.stack(patches, dim=0)
                batch_patches.append(patches)
                patches = []

        if len(patches):
            patches = torch.stack(patches, dim=0)
            batch_patches.append(patches)

        features = np.zeros((0, 2048))

        for patches in batch_patches:
            patches_ = torch.clone(patches)
            pred = self.model(patches)
            pred[torch.isinf(pred)] = 1.0

            feat = postprocess(pred)

            # Handle NaN features
            nans = np.isnan(np.sum(feat, axis=1))
            if np.isnan(feat).any():
                for n in range(np.size(nans)):
                    if nans[n]:
                        patch_np = patches_[n, ...]
                        patch_np_ = torch.unsqueeze(patch_np, 0)
                        pred_ = self.model(patch_np_)
                        feat[n] = postprocess(pred_)[0]

            features = np.vstack((features, feat))

        return features
