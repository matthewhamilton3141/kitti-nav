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
- **License:** Apache 2.0. Used for the ORB feature front-end and stereo matching.

## Prior work by this author

### gsplat-rt
- **Source:** this author's real-time Gaussian-splat SLAM repo (`~/Documents/gsplat-rt`).
- **Reused here:** the *design* of the one-step-lookahead safety shield
  (`src/isaac/nav_sim.py` — `predict_pose` / `clearance_at` / `safety_shield`) and the
  pure-NumPy, laptop-testable env contract. The kinematics are re-derived for an
  Ackermann/bicycle vehicle rather than ported verbatim; files that descend from
  gsplat-rt say so in their module docstring.

## Concepts referenced (not code)

- **Braking-aware safety filtering** generalizes the inevitable-collision-state idea
  (Fraichard & Asama, 2004) and the reachability arguments behind Responsibility-Sensitive
  Safety (Shalev-Shwartz, Shammah & Shashua, 2017). Implemented from the concept; no code
  taken from any implementation.
- **Multi-circle footprint approximation** of a rectangular vehicle for fast collision
  checking is standard practice in production planning stacks (Apollo, Autoware).
