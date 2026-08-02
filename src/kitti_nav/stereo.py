"""Stereo disparity -> metric depth, via OpenCV's semi-global block matcher.

This is the piece that makes the visual odometry here fundamentally easier than the
monocular case in this author's `gsplat-rt`. There, depth came from a learned monocular
network whose output is *relative* — a whole sub-system (`src/slam/monocular_scale.py`)
existed to recover metric scale, and residual scale drift was the dominant error. Stereo
sidesteps that entirely: with a known baseline `b` and focal length `fx`,

    depth = fx * b / disparity

is metric by construction. No scale estimation, no scale drift. The trade is that depth
is only available where the matcher finds correspondence — sky, texture-less road, and
occluded pixels come back invalid, so every consumer must respect the validity mask.

OpenCV is Apache 2.0; SGBM is Hirschmüller's semi-global matching (2005). See
`ATTRIBUTION.md`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StereoDepthConfig:
    """SGBM parameters, defaulted for KITTI's 1242x375 rectified colour pairs.

    `num_disparities` must be a multiple of 16. At fx≈721 px and b≈0.53 m, a 128-level
    search bottoms out at `721*0.53/128 ≈ 3.0 m` — closer than anything the car will
    legitimately see on the road, so 128 is enough headroom for this dataset.
    """

    num_disparities: int = 128
    block_size: int = 5
    uniqueness_ratio: int = 10
    speckle_window_size: int = 100
    speckle_range: int = 2
    disp12_max_diff: int = 1
    min_depth: float = 1.0        # m; nearer than this is almost always a matching artefact
    max_depth: float = 80.0       # m; beyond this stereo depth is too noisy to trust for PnP


class StereoDepth:
    """Computes metric depth maps from rectified stereo pairs."""

    def __init__(self, fx: float, baseline: float,
                 cfg: StereoDepthConfig | None = None):
        import cv2

        self.fx, self.baseline = float(fx), float(baseline)
        self.cfg = cfg or StereoDepthConfig()

        # P1/P2 penalise small/large disparity changes between neighbours. The 8*c*b^2 and
        # 32*c*b^2 forms are the values OpenCV's own documentation recommends; c = 1 channel
        # because we match on grayscale.
        b = self.cfg.block_size
        self._matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=self.cfg.num_disparities,
            blockSize=b,
            P1=8 * 1 * b * b,
            P2=32 * 1 * b * b,
            disp12MaxDiff=self.cfg.disp12_max_diff,
            uniquenessRatio=self.cfg.uniqueness_ratio,
            speckleWindowSize=self.cfg.speckle_window_size,
            speckleRange=self.cfg.speckle_range,
            mode=cv2.StereoSGBM_MODE_SGBM_3WAY,
        )

    def disparity(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Sub-pixel disparity in pixels; non-positive where the match failed.

        OpenCV returns disparity as int16 scaled by 16 (4 fractional bits), so the /16.0
        is a unit conversion, not a magic number.
        """
        return self._matcher.compute(left, right).astype(np.float32) / 16.0

    def depth(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Metric depth (m) from a rectified pair; **0 marks invalid**, never a real depth.

        Using 0 as the invalid sentinel (rather than NaN) matches the convention the PnP
        back-projection expects and keeps the array plain float32.
        """
        disp = self.disparity(left, right)
        depth = np.zeros_like(disp, dtype=np.float32)

        valid = disp > 0.0
        depth[valid] = self.fx * self.baseline / disp[valid]

        # Reject depths outside the range where stereo is trustworthy. Far depths come from
        # sub-pixel disparities where quantisation error explodes: at 80 m a single 1/16-px
        # disparity step already moves the estimate by metres.
        depth[(depth < self.cfg.min_depth) | (depth > self.cfg.max_depth)] = 0.0
        return depth


def depth_from_disparity(disparity: np.ndarray, fx: float, baseline: float) -> np.ndarray:
    """`fx * b / d`, elementwise, with non-positive disparity mapped to 0 (invalid)."""
    disp = np.asarray(disparity, dtype=np.float64)
    out = np.zeros(disp.shape, dtype=np.float64)
    valid = disp > 0.0
    out[valid] = fx * baseline / disp[valid]
    return out
