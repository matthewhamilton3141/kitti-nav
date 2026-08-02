"""KITTI raw-drive access: rectified stereo, calibration, and ground-truth poses.

A thin adapter over **pykitti** (MIT, Lee Clement, <https://github.com/utiasSTARS/pykitti>)
rather than a hand-rolled parser — see `ATTRIBUTION.md`. pykitti already handles the awkward
parts (calibration file parsing, OXTS packet decoding, the Mercator projection from
lat/lon to local metres), so this module only does what pykitti deliberately leaves open:

  * expose the *rectified* pinhole intrinsics and the stereo baseline as plain scalars,
    which is what the depth and PnP maths actually want;
  * convert OXTS IMU ground truth into **camera-2 poses in a local frame with the first
    pose at the origin**, so an estimated trajectory can be compared to it directly.

That second point is the fiddly bit. OXTS gives `T_w_imu` (IMU in a global ENU-ish frame),
but visual odometry estimates the *camera*'s motion, and the two are ~1.1 m apart with a
90-degree-ish rotation between their axes. Comparing VO output against raw OXTS without
composing `T_cam2_imu` is a classic way to manufacture a large fake error.

KITTI camera convention: +x right, +y **down**, +z forward.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "kitti_raw"


@dataclass(frozen=True)
class Intrinsics:
    """Rectified pinhole intrinsics of the left colour camera (cam2)."""

    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]], dtype=np.float64)


def invert_se3(T: np.ndarray) -> np.ndarray:
    """Inverse of a 4x4 rigid transform, without a general matrix inverse."""
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4, dtype=T.dtype)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


class KittiDrive:
    """One KITTI raw drive: stereo imagery, calibration, lidar, and ground-truth poses."""

    def __init__(self, date: str = "2011_09_26", drive: str = "0009",
                 base_dir: Optional[Path | str] = None, n_frames: Optional[int] = None):
        import pykitti                                    # imported lazily: heavy-ish, optional

        self.base_dir = Path(base_dir) if base_dir is not None else DEFAULT_DATA_DIR
        self.date, self.drive = date, drive
        if not (self.base_dir / date).exists():
            raise FileNotFoundError(
                f"no KITTI data at {self.base_dir / date}. "
                f"Run: python3 scripts/fetch_kitti.py --date {date} --drive {drive}")

        frames = range(n_frames) if n_frames else None
        self._data = pykitti.raw(str(self.base_dir), date, drive, frames=frames)
        self._n = len(self._data.timestamps)

    def __len__(self) -> int:
        return self._n

    # -- calibration ---------------------------------------------------------------------

    @cached_property
    def intrinsics(self) -> Intrinsics:
        P = self._data.calib.P_rect_20
        return Intrinsics(fx=float(P[0, 0]), fy=float(P[1, 1]),
                          cx=float(P[0, 2]), cy=float(P[1, 2]))

    @cached_property
    def baseline(self) -> float:
        """Stereo baseline (m) between the rectified colour cameras (cam2 -> cam3).

        Recovered from the rectified projection matrices: the right camera's `P` carries a
        translation term `-fx * b`, so the separation is the difference of those terms over
        `fx`. Reading it off the raw extrinsics instead would ignore rectification.
        """
        P2, P3 = self._data.calib.P_rect_20, self._data.calib.P_rect_30
        return float(abs(P3[0, 3] - P2[0, 3]) / P2[0, 0])

    # -- imagery -------------------------------------------------------------------------

    def gray_pair(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """Rectified (left, right) colour frames as uint8 grayscale — what stereo/ORB want."""
        import cv2

        left, right = self._data.get_rgb(i)
        to_gray = lambda im: cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2GRAY)  # noqa: E731
        return to_gray(left), to_gray(right)

    def rgb_left(self, i: int) -> np.ndarray:
        """Rectified left colour frame as an RGB uint8 array."""
        return np.asarray(self._data.get_rgb(i)[0])

    def velodyne(self, i: int) -> np.ndarray:
        """Raw Velodyne scan as `(N, 4)` — x, y, z, reflectance — in the lidar frame."""
        return self._data.get_velo(i)

    @cached_property
    def T_cam2_velo(self) -> np.ndarray:
        """Lidar -> camera-2 transform, for projecting scans into the camera/BEV frame."""
        return np.asarray(self._data.calib.T_cam2_velo, dtype=np.float64)

    # -- ground truth --------------------------------------------------------------------

    @cached_property
    def gt_poses(self) -> np.ndarray:
        """Ground-truth camera-2 poses as `(N, 4, 4)` camera-to-world, first pose at identity.

        `T_w_cam2 = T_w_imu @ inv(T_cam2_imu)`, then left-multiplied by the inverse of the
        first pose so the trajectory starts at the origin in the initial camera frame —
        the same convention the estimated trajectory uses, making them directly comparable.
        """
        T_imu_cam2 = invert_se3(np.asarray(self._data.calib.T_cam2_imu, dtype=np.float64))
        poses = np.stack([np.asarray(o.T_w_imu, dtype=np.float64) @ T_imu_cam2
                          for o in self._data.oxts])
        return invert_se3(poses[0]) @ poses

    @cached_property
    def speeds(self) -> np.ndarray:
        """Forward speed (m/s) per frame, straight from the OXTS packets."""
        return np.array([o.packet.vf for o in self._data.oxts], dtype=np.float64)

    @property
    def path_length(self) -> float:
        """Total ground-truth distance travelled (m) — the denominator for drift-percent."""
        xyz = self.gt_poses[:, :3, 3]
        return float(np.sum(np.linalg.norm(np.diff(xyz, axis=0), axis=1)))

    def frames(self) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
        """Yield `(index, left_gray, right_gray)` for the whole drive."""
        for i in range(len(self)):
            left, right = self.gray_pair(i)
            yield i, left, right
