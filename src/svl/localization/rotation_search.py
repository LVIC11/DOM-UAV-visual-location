"""Physical query rotation with exact inverse pixels and existing SVL geometry."""

from dataclasses import dataclass

import cv2
import numpy as np

from svl.localization.geometry import estimate_planar_pnp, satellite_pixels_to_metric


def transform_points(points, matrix):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if not len(points):
        return points.copy()
    return cv2.perspectiveTransform(points.reshape(-1, 1, 2), matrix).reshape(-1, 2)


def rotation_geometry(shape, clockwise_deg):
    h, w = shape[:2]
    if min(h, w) < 2 or not np.isfinite(clockwise_deg):
        raise ValueError("Invalid image shape or rotation")
    angle = float(clockwise_deg) % 360.0
    cardinal = {
        0: (np.eye(3), (w, h)),
        90: (np.array([[0, -1, h - 1], [1, 0, 0], [0, 0, 1.0]]), (h, w)),
        180: (np.array([[-1, 0, w - 1], [0, -1, h - 1], [0, 0, 1.0]]), (w, h)),
        270: (np.array([[0, 1, 0], [-1, 0, w - 1], [0, 0, 1.0]]), (h, w)),
    }
    for value, result in cardinal.items():
        if np.isclose(angle, value):
            return result
    affine = np.eye(3)
    affine[:2] = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), -angle, 1.0)
    corners = transform_points([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], affine)
    lower, upper = corners.min(axis=0), corners.max(axis=0)
    affine[:2, 2] -= lower
    size = tuple((np.ceil(upper - lower).astype(int) + 1).tolist())
    return affine, size


def rotate_query(image, clockwise_deg, long_side=None):
    """No crop; coordinate origins and resize grid stay explicit in the affine."""
    matrix, (w, h) = rotation_geometry(image.shape, clockwise_deg)
    if long_side is not None:
        if long_side < 2:
            raise ValueError("Query long side must be >=2")
        nw, nh = max(2, round(w * long_side / max(w, h))), max(
            2, round(h * long_side / max(w, h))
        )
        matrix = np.diag([(nw - 1) / (w - 1), (nh - 1) / (h - 1), 1.0]) @ matrix
        w, h = nw, nh
    rotated = cv2.warpAffine(
        image,
        matrix[:2],
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return rotated, matrix


@dataclass
class RotationCandidate:
    angle: float
    long_side: int
    tile: object
    image: np.ndarray
    affine: np.ndarray
    query_points: np.ndarray
    raw_points: np.ndarray
    satellite_points: np.ndarray
    query_indices: np.ndarray
    satellite_indices: np.ndarray
    confidence: np.ndarray
    homography: np.ndarray
    homography_mask: np.ndarray
    pose: object = None
    pnp_input_count: int = 0
    center_prediction: object = None

    @property
    def has_prediction(self):
        return self.pose is not None or self.center_prediction is not None

    @property
    def score(self):
        if self.pose is not None:
            return (
                1,
                self.pose.inlier_count,
                self.pose.inlier_ratio,
                -self.pose.reprojection_mean_px,
            )
        if self.center_prediction is not None:
            return (1, len(self.raw_points), 0.0, 0.0)
        return (0, int(self.homography_mask.sum()), 0.0, 0.0)


def fit_homography_center(candidate, min_inliers=4):
    """Legacy image-center projection; this is NOT a 3D camera-position solve."""
    if int(candidate.homography_mask.sum()) < min_inliers:
        return dict(stage="insufficient_homography_inliers_for_center")
    h, w = candidate.image.shape[:2]
    source = np.array([[w / 2.0, h / 2.0]], dtype=np.float64)
    homogeneous = candidate.homography @ np.array([w / 2.0, h / 2.0, 1.0])
    if not np.isfinite(homogeneous).all() or abs(homogeneous[2]) < 1e-9:
        return dict(stage="homography_center_nonfinite")
    projected = homogeneous[:2] / homogeneous[2]
    sh, sw = candidate.tile.image.shape[:2]
    if not (0 <= projected[0] < sw and 0 <= projected[1] < sh):
        return dict(stage="homography_center_outside_tile")
    # The historical predictor used integer satellite pixels before interpolation.
    x, y = int(projected[0]), int(projected[1])
    tile = candidate.tile
    residuals = np.linalg.norm(
        transform_points(candidate.query_points, candidate.homography)
        - candidate.satellite_points,
        axis=1,
    )
    candidate.center_prediction = dict(
        source_query_center=source[0].tolist(),
        satellite_center_pixel=[x, y],
        satellite_center_pixel_float=projected.tolist(),
        latitude=float(
            tile.top_left.lat + y / sh * (tile.bottom_right.lat - tile.top_left.lat)
        ),
        longitude=float(
            tile.top_left.long + x / sw * (tile.bottom_right.long - tile.top_left.long)
        ),
        homography_reprojection_mean_px=float(
            residuals[candidate.homography_mask].mean()
        ),
        position_represents="image_center_ground_projection_not_3d_camera_position",
    )
    return dict(stage="homography_center_passed", **candidate.center_prediction)


def fit_candidate_pose(candidate, calibration, config):
    tile = candidate.tile
    ground = satellite_pixels_to_metric(
        candidate.satellite_points,
        tile.image.shape,
        tile.top_left.lat,
        tile.top_left.long,
        tile.bottom_right.lat,
        tile.bottom_right.long,
        config.reference_latitude,
        config.reference_longitude,
    )
    use_prefilter = getattr(config, "pnp_homography_prefilter", False)
    selection = (
        candidate.homography_mask if use_prefilter else np.ones(len(ground), dtype=bool)
    )
    candidate.pnp_input_count = int(selection.sum())
    diagnostic = dict(
        pnp_input_matches=candidate.pnp_input_count,
        pnp_prefilter="homography_inliers" if use_prefilter else "all_superglue_matches",
    )
    if candidate.pnp_input_count < config.pnp_min_inliers:
        diagnostic["stage"] = "insufficient_homography_inliers_for_pnp"
        return diagnostic
    raw, plane = candidate.raw_points[selection], ground[selection]
    _, seed_mask = cv2.findHomography(
        plane - plane.mean(axis=0),
        raw,
        cv2.RANSAC,
        config.pnp_ransac_threshold_px,
        maxIters=3000,
        confidence=0.999,
    )
    seed_count = int(seed_mask.sum()) if seed_mask is not None else 0
    diagnostic.update(pnp_seed_inliers=seed_count, pnp_seed_ratio=seed_count / len(raw))
    pose = estimate_planar_pnp(
        raw,
        plane,
        calibration,
        min_inliers=config.pnp_min_inliers,
        min_inlier_ratio=config.pnp_min_inlier_ratio,
        ransac_threshold_px=config.pnp_ransac_threshold_px,
        min_bbox_area_ratio=config.pnp_min_bbox_area_ratio,
        min_camera_height_m=config.pnp_min_camera_height_m,
        max_camera_height_m=config.pnp_max_camera_height_m,
        max_optical_tilt_deg=config.pnp_max_optical_tilt_deg,
        tilt_regularization_px_per_deg=config.pnp_tilt_regularization_px_per_deg,
    )
    diagnostic["stage"] = "pnp_passed" if pose is not None else "pnp_geometry_rejected"
    if pose is None:
        diagnostic["rejection_detail"] = (
            "raw_pixel_consensus_count_or_ratio"
            if seed_count < config.pnp_min_inliers
            or seed_count / len(raw) < config.pnp_min_inlier_ratio
            else "pose_depth_tilt_reprojection_or_coverage"
        )
    else:
        full_mask = np.zeros(len(ground), dtype=bool)
        full_mask[np.flatnonzero(selection)[pose.inlier_mask]] = True
        pose.inlier_mask = full_mask
        candidate.pose = pose
        diagnostic.update(
            pnp_inliers=pose.inlier_count,
            raw_pnp_consensus_ratio=pose.inlier_count / len(ground),
            reprojection_mean_px=pose.reprojection_mean_px,
        )
    return diagnostic


def match_rotation(
    image, angle, long_side, detector, matcher, reader, calibration, config
):
    method = getattr(config, "position_method", "planar_pnp")
    if method not in ("planar_pnp", "homography_center"):
        raise ValueError("Unknown position method: " + method)
    query, affine = rotate_query(image, angle, long_side)
    features = detector.detect_and_describe_keypoints(query)
    best = None
    diagnostics = []
    for tile in reader:
        row = dict(
            rotation_cw=float(angle),
            long_side=int(long_side),
            tile=tile.name,
            query_features=len(features),
            raw_matches=0,
            homography_inliers=0,
            stage="insufficient_matches",
        )
        diagnostics.append(row)
        if min(len(features), len(tile.key_points)) < 4:
            continue
        matches, confidence = matcher.match_keypoints(features, tile.key_points)
        matches, confidence = np.atleast_1d(matches), np.atleast_1d(confidence)
        ids = np.flatnonzero(matches >= 0)
        qp = features.keypoints[ids]
        raw = transform_points(qp, np.linalg.inv(affine))
        h, w = image.shape[:2]
        valid = (
            (raw[:, 0] >= 0)
            & (raw[:, 0] <= w - 1)
            & (raw[:, 1] >= 0)
            & (raw[:, 1] <= h - 1)
        )
        ids, qp, raw = ids[valid], qp[valid], raw[valid]
        sp = tile.key_points.keypoints[matches[ids]]
        row["raw_matches"] = len(qp)
        if len(qp) < 4:
            continue
        H, mask = cv2.findHomography(
            qp,
            sp,
            config.homography_method,
            config.homography_threshold,
            maxIters=config.homography_max_iter,
            confidence=config.homography_confidence,
        )
        row["stage"] = "homography_failed"
        if H is None or mask is None:
            continue
        mask = mask.ravel().astype(bool)
        row["homography_inliers"] = int(mask.sum())
        candidate = RotationCandidate(
            float(angle),
            int(long_side),
            tile,
            query,
            affine,
            qp,
            raw,
            sp,
            ids,
            matches[ids],
            confidence[ids],
            H,
            mask,
        )
        row["stage"] = "insufficient_matches_for_pnp"
        if method == "homography_center":
            row.update(
                fit_homography_center(
                    candidate, getattr(config, "homography_min_inliers", 4)
                )
            )
        elif len(qp) >= config.pnp_min_inliers:
            row.update(fit_candidate_pose(candidate, calibration, config))
        if best is None or candidate.score > best.score:
            best = candidate
    return best, diagnostics


def export_match(candidate, frame, timestamp, calibration, config):
    if candidate is None or not candidate.has_prediction:
        raise ValueError("Cannot export unverified geometry as a satellite constraint")
    pose = (
        candidate.pose.to_dict(config.reference_latitude, config.reference_longitude)
        if candidate.pose
        else None
    )
    mask = candidate.pose.inlier_mask if pose is not None else candidate.homography_mask
    method = "planar_pnp" if pose is not None else "homography_center"
    tile = candidate.tile
    h, w = tile.image.shape[:2]
    points = []
    for i in np.flatnonzero(mask):
        raw, query, sat = (
            candidate.raw_points[i],
            candidate.query_points[i],
            candidate.satellite_points[i],
        )
        points.append(
            dict(
                query_idx=int(candidate.query_indices[i]),
                sat_idx=int(candidate.satellite_indices[i]),
                query_x=float(raw[0]),
                query_y=float(raw[1]),
                query_rotated_x=float(query[0]),
                query_rotated_y=float(query[1]),
                query_rotation_cw=candidate.angle,
                query_input_width=candidate.image.shape[1],
                query_input_height=candidate.image.shape[0],
                query_unrotated_width=calibration.image_width,
                query_unrotated_height=calibration.image_height,
                sat_x=float(sat[0]),
                sat_y=float(sat[1]),
                sat_lat=float(
                    tile.top_left.lat
                    + sat[1] / h * (tile.bottom_right.lat - tile.top_left.lat)
                ),
                sat_lon=float(
                    tile.top_left.long
                    + sat[0] / w * (tile.bottom_right.long - tile.top_left.long)
                ),
                confidence=float(candidate.confidence[i]),
                is_pnp_inlier=pose is not None,
                is_homography_inlier=bool(candidate.homography_mask[i]),
            )
        )
    return dict(
        frame=frame,
        position_method=method,
        position_represents="camera_position"
        if pose is not None
        else candidate.center_prediction["position_represents"],
        timestamp=float(timestamp),
        matched_image=tile.name,
        query_rotation_cw=candidate.angle,
        query_coordinate_frame="unrotated_original",
        original_to_matching_pixels=candidate.affine.tolist(),
        matching_to_original_pixels=np.linalg.inv(candidate.affine).tolist(),
        num_matches=len(points),
        num_inliers=int(mask.sum()),
        num_homography_inliers=int(candidate.homography_mask.sum()),
        num_superglue_matches=len(candidate.raw_points),
        num_pnp_input_matches=candidate.pnp_input_count,
        pnp_inlier_ratio_denominator=None
        if pose is None
        else (
            "homography_inliers"
            if getattr(config, "pnp_homography_prefilter", False)
            else "all_superglue_matches"
        ),
        raw_superglue_pnp_inlier_ratio=candidate.pose.inlier_count
        / len(candidate.raw_points)
        if pose is not None
        else None,
        predicted_lat=pose["camera_lat"]
        if pose is not None
        else candidate.center_prediction["latitude"],
        predicted_lon=pose["camera_lon"]
        if pose is not None
        else candidate.center_prediction["longitude"],
        planar_pose=pose,
        homography_center=candidate.center_prediction,
        keypoints=points,
    )
