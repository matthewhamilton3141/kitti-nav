#!/usr/bin/env python3
"""Does free-space carving forget a moving actor's trail? Credited by tracklet labels.

    python3 scripts/eval_carving_credit.py                 # the headline number
    python3 scripts/eval_carving_credit.py --window 5 --min-disp 2.0
    python3 scripts/eval_carving_credit.py --types Car Truck Van

`eval_mapping.py` measured carving against a *ground-truth-pose map*, and found it a
non-result: it carves both maps, so a moving actor's smear is de-smeared in both and cancels,
and drive 0009 is "too static" to have much to forget. Both halves of that are now checkable —
0009 ships object tracklets, and 12 of its 98 objects genuinely move — so carving can be
credited directly rather than through an occupancy-agreement wash.

The construction, per moving actor and per reference frame it is present through a full window:

  * **swept** — the actor's labelled footprint at every frame in the window, each transformed
    into the reference frame exactly as `fuse_scans` transforms that frame's points. This is
    where the actor *was*.
  * **trail** — swept minus where the actor *is* at the reference frame. In an accumulated map
    these cells hold the actor's ghost: occupied by an earlier scan, empty road now.
  * **credit** — of the trail cells the fused (un-carved) map marks occupied, what fraction
    does carving retire? A high number means carving forgets the actor where it no longer is.
  * **actor kept** — the control: of the cells the actor *actually* occupies at the reference
    frame, what fraction does carving keep? This must stay high, or carving is erasing a real,
    present obstacle — the one failure that could cause a crash.

Ground-truth poses throughout, so this isolates carving's behaviour from odometry drift.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kitti_nav.bev import BEVConfig, box_corners, rasterize_box, rasterize_polygon
from kitti_nav.kitti import KittiDrive, Tracklet
from kitti_nav.mapping import (
    KITTI_EGO_BOX,
    MapConfig,
    drop_ego_returns,
    fuse_map,
    relative_lidar_transform,
    transform_points,
    window_indices,
)


def _actor_footprint(drive: KittiDrive, t: Tracklet, frame: int, ref: int,
                     bev_cfg: BEVConfig) -> np.ndarray:
    """The actor's footprint at `frame`, rasterised in `ref`'s velo frame (empty if absent)."""
    box = t.box_at(frame)
    if box is None:
        return np.zeros(bev_cfg.shape, bool)
    k = t.index_of(frame)
    corners = np.column_stack([box_corners(box), np.full(4, t.tz[k])])       # (4, 3) at box height
    if frame == ref:
        return rasterize_box(box, bev_cfg)                                   # ref frame is identity
    T = relative_lidar_transform(drive.gt_poses[ref], drive.gt_poses[frame], drive.T_cam2_velo)
    return rasterize_polygon(transform_points(corners, T)[:, :2], bev_cfg)


def credit_at(drive: KittiDrive, t: Tracklet, ref: int, mcfg: MapConfig,
              bev_cfg: BEVConfig, min_trail: int = 5) -> dict | None:
    """Trail-retirement credit for actor `t` at reference frame `ref`, or None if not scorable."""
    idx = window_indices(ref, mcfg, drive.n_velodyne)
    if idx[-1] != ref or any(t.index_of(f) is None for f in idx):
        return None                                          # actor not present all window long

    scans = [drop_ego_returns(drive.velodyne(i), KITTI_EGO_BOX) for i in idx]
    poses = [drive.gt_poses[i] for i in idx]
    fused = fuse_map(scans, poses, drive.T_cam2_velo, ref=-1,
                     cfg=replace(mcfg, carve=False), bev_cfg=bev_cfg).occupied > 0
    carved = fuse_map(scans, poses, drive.T_cam2_velo, ref=-1,
                      cfg=replace(mcfg, carve=True), bev_cfg=bev_cfg).occupied > 0

    swept = np.zeros(bev_cfg.shape, bool)
    for f in idx:
        swept |= _actor_footprint(drive, t, f, ref, bev_cfg)
    at_ref = _actor_footprint(drive, t, ref, ref, bev_cfg)

    trail = swept & ~at_ref
    fused_trail = trail & fused                              # the actor's ghost the map holds
    if fused_trail.sum() < min_trail:
        return None
    retired = fused_trail & ~carved
    actor_fused = at_ref & fused
    return {
        "trail_cells": int(fused_trail.sum()),
        "retired": int(retired.sum()),
        "actor_cells": int(actor_fused.sum()),
        "actor_kept": int((at_ref & carved).sum()),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--date", default="2011_09_26")
    p.add_argument("--drive", default="0009")
    p.add_argument("--window", type=int, default=5, help="scans fused per map")
    p.add_argument("--persistence", type=float, default=2.0, help="carve_persistence")
    p.add_argument("--min-disp", type=float, default=2.0,
                   help="world displacement (m) for an actor to count as moving")
    p.add_argument("--types", nargs="+", default=None,
                   help="restrict to these object types (e.g. Car Truck)")
    p.add_argument("--every", type=int, default=1, help="score every Nth eligible frame")
    args = p.parse_args()

    drive = KittiDrive(args.date, args.drive)
    bev_cfg = BEVConfig()
    mcfg = MapConfig(window=args.window, carve_persistence=args.persistence)

    movers = drive.moving_tracklets(args.min_disp)
    if args.types:
        movers = [t for t in movers if t.object_type in set(args.types)]
    print(f"drive {args.drive}: {len(drive.tracklets)} tracklets, "
          f"{len(movers)} moving (>{args.min_disp:g} m), window {args.window}\n")

    rows: list[dict] = []
    per_actor: list[tuple[Tracklet, list[dict]]] = []
    for t in movers:
        got = []
        for ref in range(t.first_frame, t.last_frame + 1, args.every):
            r = credit_at(drive, t, ref, mcfg, bev_cfg)
            if r is not None:
                got.append(r)
                rows.append(r)
        if got:
            per_actor.append((t, got))

    if not rows:
        print("no scorable actor/frame — try a shorter window or lower --min-disp")
        return 1

    print(f"{'actor':<12} {'frames':>7} {'trail px':>9} {'retired':>8} {'credit':>7} "
          f"{'actor kept':>11}")
    print("-" * 60)
    for t, got in per_actor:
        trail = sum(r["trail_cells"] for r in got)
        retired = sum(r["retired"] for r in got)
        acell = sum(r["actor_cells"] for r in got)
        akept = sum(r["actor_kept"] for r in got)
        print(f"{t.object_type:<12} {len(got):>7} {trail:>9} {retired:>8} "
              f"{retired / max(trail, 1):>6.0%} {akept / max(acell, 1):>10.0%}")

    trail = sum(r["trail_cells"] for r in rows)
    retired = sum(r["retired"] for r in rows)
    acell = sum(r["actor_cells"] for r in rows)
    akept = sum(r["actor_kept"] for r in rows)
    print("-" * 60)
    print(f"{'ALL':<12} {len(rows):>7} {trail:>9} {retired:>8} "
          f"{retired / max(trail, 1):>6.0%} {akept / max(acell, 1):>10.0%}")
    print(f"\ncarving retires {retired / max(trail, 1):.0%} of moving-actor trail cells, "
          f"while keeping {akept / max(acell, 1):.0%} of where the actor actually is.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
