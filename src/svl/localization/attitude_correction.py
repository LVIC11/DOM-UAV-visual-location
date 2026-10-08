"""Correct visual look-at GPS to drone GPS using body attitude and altitude.

When the drone camera is not perfectly nadir (pitch/roll ≠ 0), the visual
matching pipeline outputs the "look-at" point — where the camera optical axis
intersects the ground. This is offset from the drone's true ground projection.

Given body orientation (quaternion from SLAM/INS) and camera extrinsics, we
compute the horizontal offset and subtract it from the look-at GPS.
"""
from typing import Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R


# Camera extrinsics from nav_stereo_imu.yaml: body_T_cam0 (left camera)
# Rotation matrix: camera → body
R_BODY_CAM = np.array([
    [0.9993451, -0.02374258, -0.02730688],
    [-0.02453817, -0.99927298, -0.02917852],
    [-0.02659425, 0.02982947, -0.99920116],
], dtype=np.float64)


class AttitudeCorrector:
    """Correct look-at GPS to drone GPS using attitude and altitude."""

    def __init__(self, ba_loop_path: str):
        """Load BA_loop trajectory.

        Parameters
        ----------
        ba_loop_path : str
            Path to BA_loop.txt (timestamp tx ty tz qw qx qy qz per line).
        """
        data = np.loadtxt(ba_loop_path)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        self.ba_ts = data[:, 0]         # timestamps
        self.ba_pos = data[:, 1:4]      # tx, ty, tz
        self.ba_quat = data[:, 4:8]     # qw, qx, qy, qz

    def get_pose_at(self, timestamp: float) -> Tuple[np.ndarray, np.ndarray]:
        """Get body position and orientation quaternion at given timestamp.

        Returns (position_xyz, quat_wxyz).
        """
        idx = np.argmin(np.abs(self.ba_ts - timestamp))
        return self.ba_pos[idx].copy(), self.ba_quat[idx].copy()

    def compute_offset(
        self,
        body_quat_wxyz: np.ndarray,
        altitude_m: float,
    ) -> Tuple[float, float]:
        """Compute horizontal ground offset (east_m, north_m) from look-at to drone.

        Uses body-X (forward) direction which is stable even when quaternion flips.
        Only pitch (forward tilt) is used; roll is assumed zero since BA_loop roll
        is unreliable.

        Parameters
        ----------
        body_quat_wxyz : np.ndarray
            Body orientation quaternion [qw, qx, qy, qz] (world ← body).
        altitude_m : float
            Drone altitude above ground in meters.

        Returns
        -------
        (delta_east_m, delta_north_m)
            Horizontal offset: drone = look_at - offset.
        """
        qw, qx, qy, qz = body_quat_wxyz
        R_body_world = R.from_quat([qx, qy, qz, qw]).as_matrix()

        # Body-X (forward) in world frame — stable direction
        body_x_world = R_body_world @ np.array([1.0, 0.0, 0.0])

        # Camera optical axis in body frame
        d_body = R_BODY_CAM @ np.array([0.0, 0.0, 1.0])

        # Pitch: forward tilt from horizontal
        # body_x_world[2] = sin(pitch), negative means nose-down
        pitch = np.arcsin(np.clip(-body_x_world[2], -1.0, 1.0))

        # Optical axis tilted forward by pitch: d_world ≈ [sin(pitch), 0, -cos(pitch)]
        # in the body-forward-aligned frame. Rotate by the heading.
        # Simplified: offset direction = body-X horizontal direction (ignore yaw noise)
        forward_horiz = np.array([body_x_world[0], body_x_world[1]])
        norm = np.linalg.norm(forward_horiz)
        if norm < 1e-6:
            return 0.0, 0.0
        forward_dir = forward_horiz / norm

        # Offset magnitude: AGL * tan(pitch)
        tilt = abs(pitch)
        if tilt < 1e-6:
            return 0.0, 0.0

        offset_mag = altitude_m * np.tan(tilt)

        # Offset direction is along the forward direction
        return offset_mag * forward_dir[0], offset_mag * forward_dir[1]

    def correct_gps(
        self,
        look_at_lat: float,
        look_at_lon: float,
        image_timestamp: float,
        altitude_m: float,
    ) -> Tuple[float, float]:
        """Correct a look-at GPS to drone GPS.

        Parameters
        ----------
        look_at_lat, look_at_lon : float
            GPS of the look-at point (from visual matching).
        image_timestamp : float
            Timestamp of the drone image.
        altitude_m : float
            Drone altitude above ground in meters.

        Returns
        -------
        (drone_lat, drone_lon)
        """
        _, quat = self.get_pose_at(image_timestamp)
        de, dn = self.compute_offset(quat, altitude_m)

        # Convert meters to degrees
        dlat = -dn / 111320.0
        dlon = -de / (111320.0 * np.cos(np.radians(look_at_lat)))

        return look_at_lat + dlat, look_at_lon + dlon
