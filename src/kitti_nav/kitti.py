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


@dataclass(frozen=True)
class Tracklet:
    """One KITTI-labelled object across the frames it is visible in.

    Poses are in the **velodyne frame** — the same frame the BEV grid is built in — so a box
    drops straight into occupancy with no transform. `l`/`w`/`h` are length (along the object's
    own x), width (y), height (z); `yaw` (the pose's `rz`) rotates the footprint in the ground
    plane. The per-frame arrays are aligned so index `k` is frame `first_frame + k`; KITTI
    tracklets are contiguous over their visible span, so there are no gaps to reason about.
    """

    object_type: str
    l: float
    w: float
    h: float
    first_frame: int
    tx: np.ndarray                 # (T,) per-frame centre translation in the velodyne frame
    ty: np.ndarray
    tz: np.ndarray
    yaw: np.ndarray                # (T,) rz, footprint heading in the velo ground plane
    state: np.ndarray             # (T,) KITTI pose state (0 unset, 1 interpolated, 2 labelled)

    @property
    def frames(self) -> np.ndarray:
        return self.first_frame + np.arange(len(self.tx))

    @property
    def last_frame(self) -> int:
        return self.first_frame + len(self.tx) - 1

    def index_of(self, frame: int) -> Optional[int]:
        """Position of `frame` within this tracklet's span, or None if it is not present."""
        k = frame - self.first_frame
        return int(k) if 0 <= k < len(self.tx) else None

    def box_at(self, frame: int) -> Optional[np.ndarray]:
        """Ground-plane box `(cx, cy, yaw, l, w)` in the velo frame at `frame`, or None."""
        k = self.index_of(frame)
        if k is None:
            return None
        return np.array([self.tx[k], self.ty[k], self.yaw[k], self.l, self.w], float)


def _parse_tracklets(xml_path: Path) -> list["Tracklet"]:
    """Parse a KITTI `tracklet_labels.xml` into `Tracklet`s, poses left in the velo frame.

    The file is a boost-serialisation dump; we read only the geometry (`objectType`, `h/w/l`,
    `first_frame`, and the per-frame `tx/ty/tz/rz` + `state`), ignoring occlusion/truncation
    bookkeeping. A stdlib ElementTree parse keeps this dependency-free.
    """
    import xml.etree.ElementTree as ET

    root = ET.parse(xml_path).getroot()
    container = root.find(".//tracklets")
    if container is None:
        return []

    def _f(node: "ET.Element", tag: str) -> float:
        return float(node.find(tag).text)

    out: list[Tracklet] = []
    for tr in container.findall("item"):
        if tr.find("objectType") is None:
            continue                                   # not a tracklet entry
        poses = tr.find("poses")
        items = poses.findall("item") if poses is not None else []
        if not items:
            continue
        cols = {k: np.array([_f(it, k) for it in items], float)
                for k in ("tx", "ty", "tz", "rz")}
        state = np.array([_f(it, "state") for it in items], float)
        out.append(Tracklet(
            object_type=tr.find("objectType").text,
            l=_f(tr, "l"), w=_f(tr, "w"), h=_f(tr, "h"),
            first_frame=int(_f(tr, "first_frame")),
            tx=cols["tx"], ty=cols["ty"], tz=cols["tz"], yaw=cols["rz"], state=state))
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

    @property
    def n_velodyne(self) -> int:
        """Number of Velodyne scans, which is **not always** `len(self)`.

        Real drives drop lidar frames: drive 0009 ships 447 images and OXTS packets but only
        443 scans. Anything iterating lidar must bound on this, not on the frame count.
        """
        return len(self._data.velo_files)

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
        """Raw Velodyne scan as `(N, 4)` — x, y, z, reflectance — in the lidar frame.

        Raises with a useful message past the last scan, rather than letting pykitti's
        internal file list raise a bare `IndexError` from inside a loop (see `n_velodyne`).
        """
        if not 0 <= i < self.n_velodyne:
            raise IndexError(
                f"velodyne frame {i} out of range: this drive has {self.n_velodyne} scans "
                f"for {len(self)} image frames (KITTI drives drop lidar frames)")
        return self._data.get_velo(i)

    @cached_property
    def T_cam2_velo(self) -> np.ndarray:
        """Lidar -> camera-2 transform, for projecting scans into the camera/BEV frame."""
        return np.asarray(self._data.calib.T_cam2_velo, dtype=np.float64)

    @cached_property
    def rear_axle_in_lidar(self) -> np.ndarray:
        """`(x, y)` of the vehicle's rear axle in the Velodyne frame — where to put the car.

        The BEV grid is built in the lidar frame, but the bicycle model's pose is the **rear
        axle**, and the Velodyne is roof-mounted about 0.81 m ahead of it and 0.31 m to one
        side. Placing the vehicle footprint at the lidar origin therefore pushes a 4.77 m
        car most of a metre too far forward, which quietly corrupts every clearance query.

        Taken from `T_velo_imu`'s translation: KITTI's sensor-setup diagram puts the OXTS
        IMU/GPS unit at the rear axle, so the IMU origin is the vehicle reference point.
        """
        return np.asarray(self._data.calib.T_velo_imu, dtype=np.float64)[:2, 3]

    def vehicle_state_in_lidar(self, speed: float = 0.0, steer: float = 0.0):
        """A `VehicleState` correctly placed in this drive's BEV/lidar frame."""
        from .vehicle import VehicleState

        x, y = self.rear_axle_in_lidar
        return VehicleState(x=float(x), y=float(y), yaw=0.0, v=float(speed), steer=steer)

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

    # -- object tracklets ----------------------------------------------------------------

    @property
    def tracklet_path(self) -> Path:
        return (self.base_dir / self.date / f"{self.date}_drive_{self.drive}_sync"
                / "tracklet_labels.xml")

    @cached_property
    def tracklets(self) -> list[Tracklet]:
        """Hand-labelled objects for this drive, in the velodyne frame.

        Raises if the labels are absent — they ship separately from the drive itself; see
        `scripts/fetch_kitti.py --tracklets`. Not every drive has them, but 0009 does (98
        objects, 12 of which genuinely move in world coordinates).
        """
        if not self.tracklet_path.exists():
            raise FileNotFoundError(
                f"no tracklet labels at {self.tracklet_path}. "
                f"Run: python3 scripts/fetch_kitti.py --date {self.date} "
                f"--drive {self.drive} --tracklets")
        return _parse_tracklets(self.tracklet_path)

    def tracklet_boxes(self, i: int) -> list[tuple[Tracklet, np.ndarray]]:
        """`(tracklet, box)` for every object present at frame `i`; box is `box_at`'s array."""
        return [(t, t.box_at(i)) for t in self.tracklets if t.index_of(i) is not None]

    def tracklet_world_track(self, t: Tracklet) -> np.ndarray:
        """Object centres in world coordinates, `(T, 3)`, using the drive's ground-truth poses.

        Velo -> world is `gt_poses[frame] @ T_cam2_velo @ centre`. Frames past the pose count
        (KITTI can label slightly beyond the OXTS stream) are dropped, so the result may be
        shorter than the tracklet's span.
        """
        Tcv = self.T_cam2_velo
        poses = self.gt_poses
        pts = []
        for k, fr in enumerate(t.frames):
            if fr >= len(poses):
                break
            c = np.array([t.tx[k], t.ty[k], t.tz[k], 1.0])
            pts.append((poses[fr] @ Tcv @ c)[:3])
        return np.asarray(pts, float)

    def is_moving(self, t: Tracklet, min_disp: float = 2.0) -> bool:
        """Whether `t` actually translates in the world, vs a parked object the ego drives past.

        Judged by net world displacement over the tracklet's span — a parked car's velo-frame
        pose sweeps backward as the ego passes, but its world centre barely moves.
        """
        track = self.tracklet_world_track(t)
        if len(track) < 2:
            return False
        return bool(np.linalg.norm(track[-1] - track[0]) > min_disp)

    def moving_tracklets(self, min_disp: float = 2.0) -> list[Tracklet]:
        """The genuinely-moving subset of `tracklets` (world displacement over `min_disp` m)."""
        return [t for t in self.tracklets if self.is_moving(t, min_disp)]

    def actor_velocity(self, t: Tracklet, frame: int,
                       dt: float = 0.1) -> Optional[np.ndarray]:
        """Tracklet `t`'s velocity `(vx, vy)` in `frame`'s velo frame, or None if unavailable.

        The velo-frame centre at `frame` minus the previous frame's centre brought into this
        frame through the ego motion — so the ego's own translation is removed and what remains
        is the actor's frame-local constant velocity, the feed the dynamic shield's reachable
        set consumes (`dynamics.MovingObstacle`). Returns None when either frame is outside the
        tracklet's span. Uses ground-truth poses; the within-frame ego displacement is what
        matters here, not global drift.
        """
        from .mapping import relative_lidar_transform, transform_points

        if t.index_of(frame) is None or t.index_of(frame - 1) is None:
            return None
        Tcv = self.T_cam2_velo
        cf = t.box_at(frame)[:2]
        kprev = t.index_of(frame - 1)
        prev3 = np.array([[t.tx[kprev], t.ty[kprev], t.tz[kprev]]])
        T = relative_lidar_transform(self.gt_poses[frame], self.gt_poses[frame - 1], Tcv)
        cprev = transform_points(prev3, T)[0, :2]
        return (cf - cprev) / dt
