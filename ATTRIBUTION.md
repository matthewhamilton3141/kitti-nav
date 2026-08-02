# Attribution

Everything this project builds on, with its license. Anything adapted from an upstream
source also carries a provenance header in the file that uses it.

## Datasets

### KITTI (raw data)
- **Source:** <https://www.cvlibs.net/datasets/kitti/raw_data.php>
- **Authors:** Andreas Geiger, Philip Lenz, Christoph Stiller, Raquel Urtasun — Karlsruhe
  Institute of Technology & Toyota Technological Institute at Chicago.
- **License:** Creative Commons Attribution-NonCommercial-ShareAlike 3.0
  ([CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/)).
  **This is a non-commercial license.** This repo is a personal research/learning project
  and is used accordingly. No KITTI data is redistributed here — `data/` is gitignored and
  everything is fetched from the official host by `scripts/fetch_kitti.py`.
- **Citation:**
  > A. Geiger, P. Lenz, C. Stiller, R. Urtasun. *Vision meets Robotics: The KITTI Dataset.*
  > International Journal of Robotics Research (IJRR), 2013.

  > A. Geiger, P. Lenz, R. Urtasun. *Are we ready for Autonomous Driving? The KITTI Vision
  > Benchmark Suite.* CVPR, 2012.

## Libraries

### pykitti
- **Source:** <https://github.com/utiasSTARS/pykitti>
- **Author:** Lee Clement (UTIAS Space & Terrestrial Autonomous Robotic Systems Lab).
- **License:** MIT.
- **Used for:** loading KITTI raw sequences — calibration, OXTS GPS/IMU ground truth,
  stereo imagery, and Velodyne scans. Adopted rather than hand-rolling a parser.

### OpenCV
- **License:** Apache 2.0. Used for the ORB feature front-end, SGBM stereo matching, and
  the exact Euclidean distance transform behind the BEV clearance field.

### Stable-Baselines3
- **Source:** <https://github.com/DLR-RM/stable-baselines3>
- **Authors:** Antonin Raffin et al. (DLR-RM).
- **License:** MIT.
- **Used for:** the PPO implementation that trains the driving policy. Adopted rather than
  writing PPO from scratch — a from-scratch implementation would be a source of subtle bugs
  competing with the part of this project that is actually novel.
- **Citation:**
  > A. Raffin, A. Hill, A. Gleave, A. Kanervisto, M. Ernestus, N. Dormann.
  > *Stable-Baselines3: Reliable Reinforcement Learning Implementations.*
  > Journal of Machine Learning Research, 2021.

### Gymnasium
- **Source:** <https://github.com/Farama-Foundation/Gymnasium> (Farama Foundation).
- **License:** MIT. The RL environment API; confined to `src/kitti_nav/nav_gym.py` so the
  environment core stays dependency-free.

### PyTorch
- **License:** BSD-3-Clause. The policy network backing PPO (CPU only here).

## Prior work by this author

### gsplat-rt
- **Source:** <https://github.com/matthewhamilton3141/gsplat-rt> — this author's real-time
  TensorRT Gaussian-splat SLAM project.
- **Reused here:** the *design* of the safety shield (`src/isaac/nav_sim.py` —
  `predict_pose` / `clearance_at` / `safety_shield`), the ORB + PnP odometry geometry
  (`src/slam/rgbd_odometry.py`), the pluggable front-end seam, and the pure-NumPy,
  laptop-testable core contract.
- **What changed:** the shield's kinematics *and its safety argument* are re-derived for an
  Ackermann vehicle — a one-step lookahead is unsound at driving speeds, so the braking-
  rollout formulation replaces it. Stereo depth removes the monocular scale-recovery stage
  entirely. Circular obstacles are generalised to an `ObstacleField` so real occupancy
  grids work natively.
- Files descending from gsplat-rt state so in their module docstring, along with what
  differs and why. The README has a fuller account.

## Concepts referenced (not code)

- **Braking-aware safety filtering** generalizes the inevitable-collision-state idea
  (Fraichard & Asama, 2004) and the reachability arguments behind Responsibility-Sensitive
  Safety (Shalev-Shwartz, Shammah & Shashua, 2017). Implemented from the concept; no code
  taken from any implementation.
- **Multi-circle footprint approximation** of a rectangular vehicle for fast collision
  checking is standard practice in production planning stacks (Apollo, Autoware).
