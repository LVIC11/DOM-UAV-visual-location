import dataclasses
from typing import Tuple

import numpy as np
import torch

from lightglue import LightGlue
from svl.keypoint_pipeline.base import KeyPointMatcher
from svl.keypoint_pipeline.typing import ImageKeyPoints, LightGlueConfig, SuperGlueConfig


class LightGlueMatcher(KeyPointMatcher):
    """LightGlue keypoint matcher — faster, more accurate successor to SuperGlue.

    LightGlue: Local Feature Matching at Light Speed.
    Philipp Lindenberger, Paul-Edouard Sarlin, Marc Pollefeys. ICCV 2023.

    Parameters
    ----------
    config : LightGlueConfig
        configuration for LightGlue
    """

    def __init__(self, config: LightGlueConfig) -> None:
        super().__init__()
        self.config = config
        self.device = config.device
        self.matcher = LightGlue(
            features=config.features,
            n_layers=config.n_layers,
            flash=config.flash,
            depth_confidence=config.depth_confidence,
            width_confidence=config.width_confidence,
            filter_threshold=config.filter_threshold,
        )
        self.matcher = self.matcher.eval()
        self.matcher = self.matcher.to(config.device)

    def match_keypoints(
        self, keypoints_1: ImageKeyPoints, keypoints_2: ImageKeyPoints
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Match keypoints between two sets of descriptors using LightGlue.

        Parameters
        ----------
        keypoints_1 : ImageKeyPoints
            keypoints from the first image
        keypoints_2 : ImageKeyPoints
            keypoints from the second image

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            matches and confidence
        """
        # Build LightGlue-format input dict for image0
        kpts1 = keypoints_1.torch().to(self.device).add_batch_dimension()
        feats0 = {
            "keypoints": kpts1.keypoints,
            "descriptors": kpts1.descriptors,
            "image_size": torch.tensor(
                list(keypoints_1.image_size), device=self.device
            ).unsqueeze(0),
        }
        if kpts1.scores is not None:
            feats0["keypoint_scores"] = kpts1.scores

        # Build LightGlue-format input dict for image1
        kpts2 = keypoints_2.torch().to(self.device).add_batch_dimension()
        feats1 = {
            "keypoints": kpts2.keypoints,
            "descriptors": kpts2.descriptors,
            "image_size": torch.tensor(
                list(keypoints_2.image_size), device=self.device
            ).unsqueeze(0),
        }
        if kpts2.scores is not None:
            feats1["keypoint_scores"] = kpts2.scores

        with torch.no_grad():
            preds = self.matcher({"image0": feats0, "image1": feats1})
        matches = preds["matches0"][0].cpu().numpy()
        confidence = preds["matching_scores0"][0].cpu().numpy()
        return matches, confidence


class SuperGlueMatcher(KeyPointMatcher):
    """SuperGlue keypoint matcher (legacy, kept for backward compatibility).

    Prefer LightGlueMatcher for new projects — it is faster and more accurate.

    Parameters
    ----------
    config : SuperGlueConfig
        configuration for SuperGlue
    """

    def __init__(self, config: SuperGlueConfig) -> None:
        super().__init__()
        from superglue_lib.models.superglue import SuperGlue

        self.config = config
        self.device = config.device
        self.matcher = SuperGlue(dataclasses.asdict(config))
        self.matcher = self.matcher.eval()
        self.matcher = self.matcher.to(config.device)

    def match_keypoints(
        self, keypoints_1: ImageKeyPoints, keypoints_2: ImageKeyPoints
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Match keypoints between two sets of descriptors using SuperGlue.

        Parameters
        ----------
        keypoints_1 : ImageKeyPoints
            keypoints from the first image
        keypoints_2 : ImageKeyPoints
            keypoints from the second image

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            matches and confidence
        """
        from superglue_lib.models.superglue import SuperGlue

        inputs = {}
        # SuperGlue matching needs images only to get the image size
        # Workaround: create dummy images with the image_size specified in the keypoints
        inputs["image0"] = torch.randn(1, 1, *keypoints_1.image_size)
        inputs["image1"] = torch.randn(1, 1, *keypoints_2.image_size)

        # Convert keypoints to tensor
        keypoints_1 = (
            keypoints_1.torch()
            .to(self.device)
            .add_batch_dimension()
            .to_dict(suffix_idx=0)
        )
        keypoints_1["descriptors0"] = keypoints_1["descriptors0"].transpose(-2, -1)

        keypoints_2 = (
            keypoints_2.torch()
            .to(self.device)
            .add_batch_dimension()
            .to_dict(suffix_idx=1)
        )
        keypoints_2["descriptors1"] = keypoints_2["descriptors1"].transpose(-2, -1)

        inputs.update(**keypoints_1)
        inputs.update(**keypoints_2)

        with torch.no_grad():
            preds = self.matcher(inputs)
        matches = preds["matches0"].cpu().numpy().squeeze()
        confidence = preds["matching_scores0"].cpu().numpy().squeeze()
        return matches, confidence
