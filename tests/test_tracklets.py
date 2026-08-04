"""Tracklet XML parsing and the `Tracklet` accessors — pure unit, no KITTI download needed.

The dataset-backed side (a real drive's labels, moving-actor classification) lives in
`test_kitti_integration.py`, skipped when the drive is absent. Here we feed the parser a tiny
hand-written label file so its geometry contract is pinned without gigabytes on disk.
"""

import numpy as np
import pytest

from kitti_nav.kitti import Tracklet, _parse_tracklets

# Two objects: a Car visible frames 2-4, and a Pedestrian visible frames 0-1. Only the fields
# the parser reads are filled; the boost-serialisation cruft KITTI ships is deliberately absent.
_XML = """<?xml version="1.0"?>
<boost_serialization>
 <tracklets>
  <count>2</count>
  <item>
   <objectType>Car</objectType>
   <h>1.5</h><w>1.8</w><l>4.2</l>
   <first_frame>2</first_frame>
   <poses>
    <count>3</count>
    <item><tx>1.0</tx><ty>0.0</ty><tz>-1.5</tz><rx>0</rx><ry>0</ry><rz>0.0</rz><state>1</state></item>
    <item><tx>2.0</tx><ty>0.5</ty><tz>-1.5</tz><rx>0</rx><ry>0</ry><rz>0.1</rz><state>1</state></item>
    <item><tx>3.0</tx><ty>1.0</ty><tz>-1.5</tz><rx>0</rx><ry>0</ry><rz>0.2</rz><state>2</state></item>
   </poses>
   <finished>1</finished>
  </item>
  <item>
   <objectType>Pedestrian</objectType>
   <h>1.7</h><w>0.6</w><l>0.7</l>
   <first_frame>0</first_frame>
   <poses>
    <count>2</count>
    <item><tx>5.0</tx><ty>-2.0</ty><tz>-1.6</tz><rx>0</rx><ry>0</ry><rz>1.0</rz><state>2</state></item>
    <item><tx>5.1</tx><ty>-2.1</ty><tz>-1.6</tz><rx>0</rx><ry>0</ry><rz>1.0</rz><state>2</state></item>
   </poses>
   <finished>1</finished>
  </item>
 </tracklets>
</boost_serialization>
"""


@pytest.fixture
def tracklets(tmp_path):
    p = tmp_path / "tracklet_labels.xml"
    p.write_text(_XML)
    return _parse_tracklets(p)


def test_parses_every_object_with_its_geometry(tracklets):
    assert [t.object_type for t in tracklets] == ["Car", "Pedestrian"]
    car = tracklets[0]
    assert (car.l, car.w, car.h) == (4.2, 1.8, 1.5)
    assert car.first_frame == 2 and car.last_frame == 4
    assert np.array_equal(car.frames, [2, 3, 4])
    assert np.allclose(car.tx, [1.0, 2.0, 3.0]) and np.allclose(car.yaw, [0.0, 0.1, 0.2])
    assert np.array_equal(car.state, [1, 1, 2])


def test_index_and_box_track_the_frame(tracklets):
    car = tracklets[0]
    assert car.index_of(1) is None and car.index_of(5) is None    # outside the span
    assert car.index_of(2) == 0 and car.index_of(4) == 2
    assert car.box_at(1) is None
    box = car.box_at(3)                                            # (cx, cy, yaw, l, w)
    assert np.allclose(box, [2.0, 0.5, 0.1, 4.2, 1.8])


def test_empty_container_yields_no_tracklets(tmp_path):
    p = tmp_path / "empty.xml"
    p.write_text("<boost_serialization><tracklets><count>0</count></tracklets></boost_serialization>")
    assert _parse_tracklets(p) == []
