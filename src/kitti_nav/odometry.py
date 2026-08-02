"""Stereo visual odometry: ORB features + PnP RANSAC, frame to frame.

Provenance: the geometry and the front-end contract are adapted from this author's
`gsplat-rt`, `src/slam/rgbd_odometry.py` (ORB detect/describe -> ratio-tested BF matching ->
back-project -> `solvePnPRansac` -> compose). What changed for driving:

  * **Depth is stereo, not learned-monocular.** `gsplat-rt` needed a whole metric-scale
    recovery stage; here `fx*b/disparity` is metric by construction, so that stage is gone.
  * **The motion is different in kind.** Handheld indoor footage rotates a lot and
    translates little; a car does the opposite — long, fast, nearly-straight translation.
    That makes forward-axis depth error the dominant error source rather than rotation, and
    it is why the depth-validity gating below matters so much more here.
  * **Ratio and inlier thresholds are re-tuned**, and matches are gated on *trustworthy*
    depth (see `min_depth`/`max_depth` in `stereo.py`) rather than merely non-zero depth.

Pose convention (shared with `gsplat-rt` so the two are comparable): poses are 4x4
camera-to-world SE(3). `solvePnP` returns the extrinsic mapping cam_i coordinates into
cam_{i+1} coordinates (`T_rel`), so the next camera-to-world pose is `P_{i+1} = P_i @ inv(T_rel)`.

OpenCV is Apache 2.0. See `ATTRIBUTION.md`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Tuple

import numpy as np

from .kitti import Intrinsics, invert_se3


@dataclass
class TrackResult:
    """Outcome of one tracking step — `ok=False` means the pose is a fallback, not a fix."""

    pose: np.ndarray            # (4, 4) camera-to-world
    n_matches: int
    n_inliers: int
    ok: bool


class Frontend(Protocol):
    """Detect/describe + match contract, kept pluggable so a learned front-end can drop in.

    `gsplat-rt` used exactly this seam to swap ORB for SuperPoint+LightGlue without touching
    the geometry below. Nothing here is GPU-bound yet, so ORB is the only implementation.
    """

    def detect(self, gray: np.ndarray) -> Tuple[np.ndarray, object]: ...

    def match(self, desc0: object, desc1: object) -> np.ndarray: ...


class ORBFrontend:
    """CPU baseline: ORB keypoints + Hamming BF matching with Lowe's ratio test."""

    def __init__(self, n_features: int = 2000, ratio: float = 0.8):
        import cv2

        self.ratio = ratio
        self._orb = cv2.ORB_create(nfeatures=n_features)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def detect(self, gray: np.ndarray) -> Tuple[np.ndarray, object]:
        kp, des = self._orb.detectAndCompute(gray, None)
        if des is None or len(kp) == 0:
            return np.empty((0, 2), np.float32), None
        return np.array([k.pt for k in kp], dtype=np.float32), des

    def match(self, desc0: object, desc1: object) -> np.ndarray:
        """Ratio-tested matches as an `(M, 2)` array of `[idx0, idx1]`."""
        if desc0 is None or desc1 is None or len(desc0) < 2 or len(desc1) < 2:
            return np.empty((0, 2), np.int32)
        knn = self._matcher.knnMatch(desc0, desc1, k=2)
        good = [[m.queryIdx, m.trainIdx] for m, n in knn
                if m.distance < self.ratio * n.distance]
        return np.array(good, dtype=np.int32) if good else np.empty((0, 2), np.int32)


@dataclass
class OdometryConfig:
    min_matches: int = 20
    min_inliers: int = 8
    ransac_reproj_px: float = 2.0
    ransac_iterations: int = 200
    max_speed: float = 40.0     # m/s; a step implying more than this is rejected as a blunder
    dt: float = 0.1             # KITTI raw is 10 Hz — only used for the sanity gate above


class StereoOdometry:
    """Stateful frame-to-frame stereo visual odometer.

    Call `track(left_gray, depth)` with each frame in order. The first call seeds the
    reference frame and returns identity. On a degenerate step (too few matches, failed
    PnP, or an implausibly large jump) the previous relative motion is re-applied — a
    constant-velocity fallback, which for a car is a genuinely good prior — and
    `TrackResult.ok` is False so the caller can count how often it happened.
    """

    def __init__(self, intrinsics: Intrinsics, cfg: Optional[OdometryConfig] = None,
                 frontend: Optional[Frontend] = None):
        self.K = intrinsics
        self._Kmat = intrinsics.matrix
        self.cfg = cfg or OdometryConfig()
        self._frontend = frontend if frontend is not None else ORBFrontend()

        self._pose = np.eye(4, dtype=np.float64)
        self._last_rel = np.eye(4, dtype=np.float64)
        self._prev: Optional[tuple] = None       # (xy, des, depth)
        self.trajectory: List[np.ndarray] = []
        self.n_fallbacks = 0

    def _backproject(self, xy: np.ndarray, depth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Pixels + depth map -> (3-D points in the camera frame, validity mask).

        Nearest-neighbour depth lookup: interpolating across a depth discontinuity would
        invent a point floating between the foreground object and the background, which is
        exactly the kind of outlier PnP handles worst.
        """
        u, v = xy[:, 0], xy[:, 1]
        row = np.clip(np.rint(v).astype(int), 0, depth.shape[0] - 1)
        col = np.clip(np.rint(u).astype(int), 0, depth.shape[1] - 1)
        z = depth[row, col].astype(np.float64)

        valid = z > 0.0                          # 0 is stereo.py's invalid sentinel
        x = (u - self.K.cx) * z / self.K.fx
        y = (v - self.K.cy) * z / self.K.fy
        return np.stack([x, y, z], axis=-1).astype(np.float64), valid

    def track(self, gray: np.ndarray, depth: np.ndarray) -> TrackResult:
        """Estimate the next camera-to-world pose from a left frame and its stereo depth."""
        import cv2

        xy, des = self._frontend.detect(gray)

        if self._prev is None:
            self._prev = (xy, des, depth)
            self.trajectory.append(self._pose.copy())
            return TrackResult(self._pose.copy(), 0, 0, True)

        prev_xy, prev_des, prev_depth = self._prev
        matches = self._frontend.match(prev_des, des)

        ok, n_inliers = False, 0
        T_rel = self._last_rel                   # constant-velocity fallback

        if len(matches) >= self.cfg.min_matches:
            pts3d, valid = self._backproject(prev_xy[matches[:, 0]], prev_depth)
            obj = pts3d[valid]
            img = xy[matches[:, 1]][valid].astype(np.float64)

            if len(obj) >= self.cfg.min_matches:
                retval, rvec, tvec, inliers = cv2.solvePnPRansac(
                    obj, img, self._Kmat, None,
                    reprojectionError=self.cfg.ransac_reproj_px,
                    iterationsCount=self.cfg.ransac_iterations,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
                if retval and inliers is not None and len(inliers) >= self.cfg.min_inliers:
                    n_inliers = int(len(inliers))
                    R, _ = cv2.Rodrigues(rvec)
                    candidate = np.eye(4)
                    candidate[:3, :3], candidate[:3, 3] = R, tvec.ravel()

                    # Blunder gate: a 10 Hz step implying > max_speed is a bad PnP solution,
                    # not a real motion. Rejecting it and coasting on the previous motion is
                    # far cheaper than letting one wild pose corrupt the whole trajectory.
                    if np.linalg.norm(candidate[:3, 3]) <= self.cfg.max_speed * self.cfg.dt:
                        T_rel, ok = candidate, True

        if not ok:
            self.n_fallbacks += 1

        self._pose = self._pose @ invert_se3(T_rel)
        self._last_rel = T_rel
        self._prev = (xy, des, depth)
        self.trajectory.append(self._pose.copy())
        return TrackResult(self._pose.copy(), int(len(matches)), n_inliers, ok)

    @property
    def positions(self) -> np.ndarray:
        """Estimated camera positions as `(N, 3)`."""
        return np.stack([P[:3, 3] for P in self.trajectory])


# --- trajectory evaluation ----------------------------------------------------------------

@dataclass
class TrajectoryError:
    """Standard VO error summary. `ate_rmse` is the headline; `drift_percent` is comparable
    across sequences of different lengths, which raw ATE is not."""

    ate_rmse: float          # m, RMS position error after alignment
    ate_mean: float          # m
    ate_max: float           # m
    final_drift: float       # m, error at the last pose
    path_length: float       # m, ground-truth distance travelled
    drift_percent: float     # final_drift / path_length * 100
    n_poses: int = field(default=0)

    def __str__(self) -> str:
        return (f"ATE rmse {self.ate_rmse:.2f} m | mean {self.ate_mean:.2f} m | "
                f"max {self.ate_max:.2f} m | final drift {self.final_drift:.2f} m "
                f"over {self.path_length:.1f} m ({self.drift_percent:.2f}%)")


def evaluate_trajectory(estimated: np.ndarray, ground_truth: np.ndarray) -> TrajectoryError:
    """Compare estimated vs ground-truth positions, both `(N, 3)` in the same start frame.

    Deliberately **no Umeyama/Sim(3) alignment**. That alignment is standard for *monocular*
    VO because monocular reconstruction is only defined up to scale, so fitting a scale
    factor is measuring what the method can actually determine. Stereo VO is metric, so
    fitting a scale here would quietly absorb genuine scale error and flatter the result.
    Both trajectories already start at the origin in the same frame, so raw differences are
    the honest comparison.
    """
    est = np.asarray(estimated, dtype=np.float64)[:, :3]
    gt = np.asarray(ground_truth, dtype=np.float64)[:, :3]
    n = min(len(est), len(gt))
    est, gt = est[:n], gt[:n]

    err = np.linalg.norm(est - gt, axis=1)
    path = float(np.sum(np.linalg.norm(np.diff(gt, axis=0), axis=1)))
    final = float(err[-1])

    return TrajectoryError(
        ate_rmse=float(np.sqrt(np.mean(err ** 2))),
        ate_mean=float(np.mean(err)),
        ate_max=float(np.max(err)),
        final_drift=final,
        path_length=path,
        drift_percent=float(final / path * 100.0) if path > 0 else float("nan"),
        n_poses=n,
    )
