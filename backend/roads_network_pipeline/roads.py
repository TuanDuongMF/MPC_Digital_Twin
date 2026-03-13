"""Stage 4: Road Network Creation - replicates CreateRoadFromJson from C++.

Builds a road network graph from Lanes.json by:
1. Filtering active autonomous haulage lanes by type
2. Building QuadTree with 20 intermediate points per segment
3. Finding road-road intersections (3D closest approach)
4. Inserting intersection nodes into lane point lists
5. Identifying junction points (shared start/end coordinates)
6. Chaining lanes into roads via doubly-linked lists
7. Processing unvisited lanes (Scenario 2: free endpoints, Scenario 3: loops)
8. Creating non-haulage standalone roads
9. Optional coordinate reduction for low fidelity
10. Outputting output.json with nodes and roads ([Y,X,Z] swapped)
"""

import json
import math
import os
from collections import defaultdict

from .geometry import distance_2d

# ── Constants matching C++ CreateRoadFromJson ──
PI = 3.141592653589793
OFFSET_DISTANCE = 20        # dOffsetDistance - perpendicular lane offset (meters)
POINT_MATCH_TOL = 0.0999    # Junction detection tolerance (±0.0999)
INTERSECTION_TOL = 0.1      # ComputeIntersectionPointOfTwoLines tolerance
QT_INTERMEDIATE_PTS = 20    # Intermediate points per segment for QuadTree
COORD_PRECISION = 1         # Decimal places for coordinate key rounding


def _is_truthy(val):
    """Check if a value represents True (handles '1', 'True', True, 1, etc.)."""
    return str(val).strip().lower() in ("1", "true", "yes")


# ── QuadTree for spatial indexing ──

class QuadTree:
    """2D spatial index - replicates C++ Quad class with Point/RoadData/NodeData."""

    __slots__ = ('boundary', 'capacity', 'points', 'divided',
                 'nw', 'ne', 'sw', 'se', 'depth')

    MAX_DEPTH = 20

    def __init__(self, x_min, y_min, x_max, y_max, capacity=8, depth=0):
        self.boundary = (x_min, y_min, x_max, y_max)
        self.capacity = capacity
        self.points = []        # [(x, y, road_id, node_idx)]
        self.divided = False
        self.nw = self.ne = self.sw = self.se = None
        self.depth = depth

    def insert(self, x, y, road_id, node_idx):
        bx0, by0, bx1, by1 = self.boundary
        if x < bx0 or x > bx1 or y < by0 or y > by1:
            return False
        if len(self.points) < self.capacity or self.depth >= self.MAX_DEPTH:
            self.points.append((x, y, road_id, node_idx))
            return True
        if not self.divided:
            self._subdivide()
        return (self.nw.insert(x, y, road_id, node_idx)
                or self.ne.insert(x, y, road_id, node_idx)
                or self.sw.insert(x, y, road_id, node_idx)
                or self.se.insert(x, y, road_id, node_idx))

    def _subdivide(self):
        bx0, by0, bx1, by1 = self.boundary
        mx, my = (bx0 + bx1) / 2, (by0 + by1) / 2
        d = self.depth + 1
        self.nw = QuadTree(bx0, my, mx, by1, self.capacity, d)
        self.ne = QuadTree(mx, my, bx1, by1, self.capacity, d)
        self.sw = QuadTree(bx0, by0, mx, my, self.capacity, d)
        self.se = QuadTree(mx, by0, bx1, my, self.capacity, d)
        self.divided = True
        for p in self.points:
            (self.nw.insert(*p) or self.ne.insert(*p)
             or self.sw.insert(*p) or self.se.insert(*p))
        self.points = []

    def query_range(self, x, y, radius):
        """Find all points within *radius* of (x, y)."""
        results = []
        bx0, by0, bx1, by1 = self.boundary
        if (x - radius > bx1 or x + radius < bx0
                or y - radius > by1 or y + radius < by0):
            return results
        for px, py, rid, nidx in self.points:
            if distance_2d(x, y, px, py) <= radius:
                results.append((px, py, rid, nidx))
        if self.divided:
            results.extend(self.nw.query_range(x, y, radius))
            results.extend(self.ne.query_range(x, y, radius))
            results.extend(self.sw.query_range(x, y, radius))
            results.extend(self.se.query_range(x, y, radius))
        return results


# ── Linked-list node ──

class RoadNode:
    """Doubly-linked list node - replicates C++ struct node."""
    __slots__ = ('data', 'laneid', 'coords', 'prev', 'next_node')

    def __init__(self, node_id, lane_id, coords):
        self.data = node_id          # auto-increment node ID
        self.laneid = lane_id        # lane this point belongs to
        self.coords = list(coords)   # [X, Y, Z]
        self.prev = None
        self.next_node = None


# ── 3D line intersection ──

def _compute_intersection_3d(pt1, pt2, pt3, pt4, tol=INTERSECTION_TOL):
    """3D closest-approach between two line segments.

    Replicates ComputeIntersectionPointOfTwoLines from C++.
    Returns [x, y, z] intersection point or None.
    """
    EPS = 1e-10

    p13 = [pt1[i] - pt3[i] for i in range(3)]
    p43 = [pt4[i] - pt3[i] for i in range(3)]
    p21 = [pt2[i] - pt1[i] for i in range(3)]

    if all(abs(v) < EPS for v in p43):
        return None
    if all(abs(v) < EPS for v in p21):
        return None

    d1343 = sum(p13[i] * p43[i] for i in range(3))
    d4321 = sum(p43[i] * p21[i] for i in range(3))
    d1321 = sum(p13[i] * p21[i] for i in range(3))
    d4343 = sum(p43[i] * p43[i] for i in range(3))
    d2121 = sum(p21[i] * p21[i] for i in range(3))

    denom = d2121 * d4343 - d4321 * d4321
    if abs(denom) < EPS:
        return None

    mua = (d1343 * d4321 - d1321 * d4343) / denom
    mub = (d1343 + d4321 * mua) / d4343

    # Closest point on first line
    ix = pt1[0] + mua * p21[0]
    iy = pt1[1] + mua * p21[1]
    iz = pt1[2] + mua * p21[2]

    # Closest point on second line
    jx = pt3[0] + mub * p43[0]
    jy = pt3[1] + mub * p43[1]
    jz = pt3[2] + mub * p43[2]

    # Parameters must be inside segments (not at endpoints)
    endpoint_margin = 0.05
    if mua <= endpoint_margin or mua >= (1 - endpoint_margin):
        return None
    if mub <= endpoint_margin or mub >= (1 - endpoint_margin):
        return None

    # Verify: partial distances ≈ full segment lengths
    def _dist3(a, b):
        return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))

    ipt = [ix, iy, iz]
    jpt = [jx, jy, jz]

    d_p1_i = _dist3(pt1, ipt)
    d_i_p2 = _dist3(ipt, pt2)
    d_p1_p2 = _dist3(pt1, pt2)
    if abs((d_p1_i + d_i_p2) - d_p1_p2) > tol:
        return None

    d_p3_j = _dist3(pt3, jpt)
    d_j_p4 = _dist3(jpt, pt4)
    d_p3_p4 = _dist3(pt3, pt4)
    if abs((d_p3_j + d_j_p4) - d_p3_p4) > tol:
        return None

    # The two closest points must actually be close
    if _dist3(ipt, jpt) > tol * 2:
        return None

    return ipt


# ── Road Network Builder ──

class RoadNetworkBuilder:
    """Builds road network from lanes.

    Replicates the Lanes-mode branch of docreateroadfromjson() in C++.
    All global maps/state from C++ are instance attributes.
    """

    # Lane types accepted by C++ (autonomous haulage network)
    HAULAGE_TYPES = {"HAULAGE"}
    NON_HAULAGE_TYPES = {
        "DUMP_QUEUE", "LOAD_QUEUE", "LOAD_EXIT", "DUMP_EXIT",
        "DUMP_ENTRY", "LOAD_ENTRY",
        "STATION_QUEUE", "STATION_ENTRY", "STATION_EXIT",
    }
    VALID_TYPES = HAULAGE_TYPES | NON_HAULAGE_TYPES

    def __init__(self):
        # ── mpaLaneID_VecLanePoints ──
        self.lane_points = {}           # lane_oid → [[x,y,z], …]
        # ── mpaLaneID_StartEndPoints ──
        self.lane_start_end = {}        # lane_oid → [start_pt, end_pt]
        # ── Lane attribute maps ──
        self.lane_speed = {}
        self.lane_width = {}
        self.lane_banking = {}
        self.lane_curvature = {}
        self.lane_traction = {}
        # ── mapNonHaulageRoadIDs ──
        self.non_haulage_ids = set()

        # ── Bounds ──
        self.x_min = float("inf")
        self.y_min = float("inf")
        self.x_max = float("-inf")
        self.y_max = float("-inf")

        # ── Distance metrics ──
        self.shortest_lane_dist = float("inf")   # dShortestLaneDistance
        self.longest_lane_dist = float("-inf")    # dLongestLaneDistance

        # ── Road-building state ──
        self._node_id = 1                          # NodeId auto-increment
        self._road_id = 1                          # fallback; C++ uses lane OID
        # mapDuplicatePointsWithLaneID
        self.duplicate_points = {}                 # coord_key → [lane_oid, …]
        # mapLinkedListsOfRoadsNodes
        self.linked_nodes = {}                     # road_id → [RoadNode, …]
        # mapLinkedRoadIds
        self.linked_road_ids = {}                  # road_id → [lane_oid, …]
        # vecAddressedLaneIDs
        self.addressed = set()
        # mapIntersectionPoints
        self.intersection_pts = {}                 # coord_key → True

    # ── helpers ──

    @staticmethod
    def _ck(pt):
        """Coordinate key for hashing (rounded)."""
        return (round(pt[0], COORD_PRECISION),
                round(pt[1], COORD_PRECISION),
                round(pt[2], COORD_PRECISION) if len(pt) > 2 else 0.0)

    @staticmethod
    def _pts_match(a, b):
        """Point matching within ±POINT_MATCH_TOL (C++ tolerance 0.0999)."""
        return abs(a[0] - b[0]) < POINT_MATCH_TOL and abs(a[1] - b[1]) < POINT_MATCH_TOL

    def _next_node_id(self):
        nid = self._node_id
        self._node_id += 1
        return nid

    def _next_road_id(self):
        rid = self._road_id
        self._road_id += 1
        return rid

    # ── CreateLinkedListsForRoads ──

    def _create_linked_list(self, forward, road_id, lane_id):
        """Build doubly-linked list of nodes from a lane's points.

        *forward*=True  → iterate 0…N-1  (start is junction)
        *forward*=False → iterate N-1…0  (end is junction)

        Returns last point traversed (list [x,y,z]).
        """
        pts = self.lane_points.get(lane_id, [])
        if not pts:
            return []

        node_list = self.linked_nodes.setdefault(road_id, [])
        indices = range(len(pts)) if forward else range(len(pts) - 1, -1, -1)

        last_pt = []
        for i in indices:
            rn = RoadNode(self._next_node_id(), lane_id, pts[i])
            if node_list:
                rn.prev = node_list[-1]
                node_list[-1].next_node = rn
            node_list.append(rn)
            last_pt = pts[i]
        return last_pt

    # ── IdentifyLinkedRoads ──

    def _identify_linked(self, last_pt, road_id):
        """Find and chain next unaddressed lane sharing *last_pt*.

        Replicates IdentifyLinkedRoads from C++.
        Returns (new_last_pt, next_lane_id) or ([], None).
        """
        if not last_pt:
            return [], None

        for lid, se in self.lane_start_end.items():
            if lid in self.addressed or lid in self.non_haulage_ids:
                continue

            start_pt, end_pt = se

            for is_start, cand_pt in ((True, start_pt), (False, end_pt)):
                if not self._pts_match(last_pt, cand_pt):
                    continue
                # Skip multi-junction points (handled later)
                key = self._ck(cand_pt)
                if len(self.duplicate_points.get(key, [])) > 1:
                    continue

                self.addressed.add(lid)
                self.linked_road_ids.setdefault(road_id, []).append(lid)
                new_last = self._create_linked_list(is_start, road_id, lid)
                return new_last, lid

        return [], None

    # ── GrowRoadInformation (Scenario 1) ──

    def _grow_roads(self):
        """Chain lanes at junction points to form roads (Scenario 1).

        Replicates GrowRoadInformation from C++.
        """
        # Build endpoint → lane mapping
        ep_map = defaultdict(list)
        for lid, se in self.lane_start_end.items():
            if lid in self.non_haulage_ids:
                continue
            ep_map[self._ck(se[0])].append((lid, "start"))
            ep_map[self._ck(se[1])].append((lid, "end"))

        # Identify junction points (shared by >1 lane)
        for key, entries in ep_map.items():
            if len(entries) > 1:
                self.duplicate_points[key] = [lid for lid, _ in entries]

        # Process each junction
        for coord_key, lane_ids in self.duplicate_points.items():
            for lid in lane_ids:
                if lid in self.addressed or lid in self.non_haulage_ids:
                    continue

                self.addressed.add(lid)

                # Determine direction: junction at start → forward, else reverse
                se = self.lane_start_end[lid]
                is_start = (self._ck(se[0]) == coord_key)

                rid = lid  # C++ uses lane OID as road ID
                self.linked_road_ids[rid] = [lid]
                last_pt = self._create_linked_list(is_start, rid, lid)

                # Chain adjacent lanes
                while True:
                    new_last, next_lid = self._identify_linked(last_pt, rid)
                    if next_lid is None:
                        break
                    last_pt = new_last

    # ── GrowRoadLeftOutRoadInformation (Scenarios 2 & 3) ──

    def _grow_leftout(self, unvisited, eight_shape=False):
        """Process unvisited lanes.

        Scenario 2 (eight_shape=False): start from free (non-junction) endpoints.
        Scenario 3 (eight_shape=True):  process remaining loops / 8-shaped circuits.
        """
        if not eight_shape:
            # Rebuild endpoint map for unvisited haulage lanes
            ep_map = defaultdict(list)
            for lid in unvisited:
                if lid in self.non_haulage_ids:
                    continue
                se = self.lane_start_end.get(lid)
                if not se:
                    continue
                ep_map[self._ck(se[0])].append((lid, "start"))
                ep_map[self._ck(se[1])].append((lid, "end"))

            # Free endpoints = single-connection
            free_eps = {}
            for key, entries in ep_map.items():
                if len(entries) == 1:
                    free_eps[key] = entries[0]

            for _key, (lid, ep_type) in free_eps.items():
                if lid in self.addressed or lid in self.non_haulage_ids:
                    continue
                self.addressed.add(lid)
                is_start = (ep_type == "start")

                rid = lid  # C++ uses lane OID as road ID
                self.linked_road_ids[rid] = [lid]
                last_pt = self._create_linked_list(is_start, rid, lid)

                while True:
                    new_last, next_lid = self._identify_linked(last_pt, rid)
                    if next_lid is None:
                        break
                    last_pt = new_last
        else:
            # Scenario 3: remaining loops
            for lid in list(unvisited):
                if lid in self.addressed or lid in self.non_haulage_ids:
                    continue
                self.addressed.add(lid)

                rid = lid  # C++ uses lane OID as road ID
                self.linked_road_ids[rid] = [lid]
                last_pt = self._create_linked_list(True, rid, lid)

                while True:
                    new_last, next_lid = self._identify_linked(last_pt, rid)
                    if next_lid is None:
                        break
                    last_pt = new_last

    # ── CreateNonHaulageRoads ──

    def _create_non_haulage(self, unvisited):
        """Create standalone roads for non-haulage (queue/exit) lanes."""
        for lid in unvisited:
            if lid in self.addressed:
                continue
            self.addressed.add(lid)
            rid = lid  # C++ uses lane OID as road ID
            self.linked_road_ids[rid] = [lid]
            self._create_linked_list(True, rid, lid)

    # ── InsertNewNodeIntoLanes ──

    def _insert_intersection_node(self, lane_id, seg_idx, point):
        """Insert intersection point into lane's point list after *seg_idx*.

        Replicates InsertNewNodeIntoLanes from C++.
        """
        pts = self.lane_points.get(lane_id)
        if pts is None or seg_idx + 1 > len(pts):
            return
        pts.insert(seg_idx + 1, list(point))
        # Refresh start/end
        self.lane_start_end[lane_id] = [pts[0], pts[-1]]
        self.intersection_pts[self._ck(point)] = True

    # ── CreateIntersectionPoints ──

    def _create_intersections(self):
        """Build QuadTree and detect all road-road intersections.

        Replicates CreateIntersectionPoints from C++:
        - 20 intermediate points per segment inserted into QuadTree
        - For each segment, query QuadTree → find nearby roads → check intersection
        - Insert intersection nodes into both lanes
        """
        if self.x_min == float("inf"):
            return

        margin = 10
        qt = QuadTree(
            self.x_min - margin, self.y_min - margin,
            self.x_max + margin, self.y_max + margin,
        )

        # Populate QuadTree with 20 intermediate points per segment
        haulage_ids = [lid for lid in self.lane_points if lid not in self.non_haulage_ids]
        for lid in haulage_ids:
            pts = self.lane_points[lid]
            for i in range(len(pts) - 1):
                p1, p2 = pts[i], pts[i + 1]
                # C++: for (int t = 1; t <= steps; t++) → 20 points (1..20)
                for s in range(1, QT_INTERMEDIATE_PTS + 1):
                    t = s / QT_INTERMEDIATE_PTS
                    qt.insert(
                        p1[0] + t * (p2[0] - p1[0]),
                        p1[1] + t * (p2[1] - p1[1]),
                        lid, i,
                    )

        # Search radius from C++: dShortestLaneDistance * 0.5
        radius = self.shortest_lane_dist * 0.5 if self.shortest_lane_dist < float("inf") else 50
        radius = max(radius, 10)

        checked_pairs = set()
        total_inserted = 0

        for lid in haulage_ids:
            pts = self.lane_points[lid]
            seg_i = 0
            while seg_i < len(pts) - 1:
                p1, p2 = pts[seg_i], pts[seg_i + 1]

                # Query at both endpoints of this segment
                nearby_roads = set()
                for qp in (p1, p2):
                    for _, _, rid, _ in qt.query_range(qp[0], qp[1], radius):
                        if rid != lid:
                            nearby_roads.add(rid)

                for other_lid in nearby_roads:
                    pair = (min(lid, other_lid), max(lid, other_lid))
                    if pair in checked_pairs:
                        continue
                    checked_pairs.add(pair)

                    other_pts = self.lane_points.get(other_lid, [])
                    # Collect insertions per lane to apply in reverse index order
                    inserts_a = {}   # seg_idx → [x,y,z]
                    inserts_b = {}

                    for ai in range(len(pts) - 1):
                        a1 = pts[ai]
                        a2 = pts[ai + 1]
                        for bi in range(len(other_pts) - 1):
                            b1 = other_pts[bi]
                            b2 = other_pts[bi + 1]
                            sa = [a1[0], a1[1], a1[2] if len(a1) > 2 else 0]
                            ea = [a2[0], a2[1], a2[2] if len(a2) > 2 else 0]
                            sb = [b1[0], b1[1], b1[2] if len(b1) > 2 else 0]
                            eb = [b2[0], b2[1], b2[2] if len(b2) > 2 else 0]

                            ipt = _compute_intersection_3d(sa, ea, sb, eb)
                            if ipt:
                                inserts_a[ai] = ipt
                                inserts_b[bi] = ipt

                    # Insert in reverse order so indices stay valid
                    for idx in sorted(inserts_a, reverse=True):
                        self._insert_intersection_node(lid, idx, inserts_a[idx])
                        total_inserted += 1
                    for idx in sorted(inserts_b, reverse=True):
                        self._insert_intersection_node(other_lid, idx, inserts_b[idx])

                    # Refresh local pts reference after insertion
                    pts = self.lane_points[lid]

                seg_i += 1

        if total_inserted:
            print(f"  Intersection nodes inserted: {total_inserted}")

    # ── Main build ──

    def build(self, lanes_data, fidelity="Low"):
        """Main entry point - replicates docreateroadfromjson (Lanes mode)."""

        all_lanes = lanes_data.get("lanes", [])

        # ── Step 1: Extract and filter lanes ──
        for lane in all_lanes:
            oid_str = lane.get("LANE_OID", "").strip()
            if not oid_str:
                continue
            try:
                oid = int(oid_str)  # C++ uses strtoll
            except ValueError:
                continue

            is_active = lane.get("IS_ACTIVE", "0")
            autonomous = lane.get("AUTONOMOUS", "0")
            lane_type = str(lane.get("TYPE", "")).strip().upper()
            points = lane.get("points", [])

            if not _is_truthy(is_active) or not _is_truthy(autonomous):
                continue
            if lane_type not in self.VALID_TYPES:
                continue
            if not points or len(points) < 2:
                continue

            # Ensure 3D
            pts = []
            for pt in points:
                pts.append([float(pt[0]), float(pt[1]),
                            float(pt[2]) if len(pt) > 2 else 0.0])

            self.lane_points[oid] = pts
            self.lane_start_end[oid] = [pts[0], pts[-1]]

            self.lane_speed[oid] = str(lane.get("SPEED_LIMIT", "0") or "0")
            self.lane_width[oid] = str(lane.get("LANE_WIDTH", "0") or "0")
            self.lane_banking[oid] = str(lane.get("BANKING", "0") or "0")
            self.lane_curvature[oid] = str(lane.get("CURVATURE", "0") or "0")
            self.lane_traction[oid] = str(lane.get("TRACTION", "1") or "1")

            if lane_type in self.NON_HAULAGE_TYPES:
                self.non_haulage_ids.add(oid)

            # Update bounds (dLeast_X/Y, dHighest_X/Y)
            for pt in pts:
                self.x_min = min(self.x_min, pt[0])
                self.y_min = min(self.y_min, pt[1])
                self.x_max = max(self.x_max, pt[0])
                self.y_max = max(self.y_max, pt[1])

            # Lane distance
            d = sum(distance_2d(pts[j][0], pts[j][1], pts[j + 1][0], pts[j + 1][1])
                    for j in range(len(pts) - 1))
            if d > 0:
                self.shortest_lane_dist = min(self.shortest_lane_dist, d)
                self.longest_lane_dist = max(self.longest_lane_dist, d)

        haulage_n = sum(1 for lid in self.lane_points if lid not in self.non_haulage_ids)
        print(f"  Haulage lanes: {haulage_n}, Non-haulage: {len(self.non_haulage_ids)}")

        if not self.lane_points:
            # Fallback: use all active lanes regardless of type
            print("  Warning: No matching lanes. Trying all active autonomous lanes...")
            for lane in all_lanes:
                oid_str = lane.get("LANE_OID", "").strip()
                if not oid_str:
                    continue
                try:
                    oid = int(oid_str)  # C++ uses strtoll
                except ValueError:
                    continue
                is_active = lane.get("IS_ACTIVE", "0")
                autonomous = lane.get("AUTONOMOUS", "0")
                points = lane.get("points", [])
                if not _is_truthy(is_active) or not _is_truthy(autonomous):
                    continue
                if not points or len(points) < 2:
                    continue
                pts = [[float(p[0]), float(p[1]),
                        float(p[2]) if len(p) > 2 else 0.0] for p in points]
                self.lane_points[oid] = pts
                self.lane_start_end[oid] = [pts[0], pts[-1]]
                self.lane_speed[oid] = str(lane.get("SPEED_LIMIT", "0") or "0")
                self.lane_width[oid] = str(lane.get("LANE_WIDTH", "0") or "0")
                self.lane_banking[oid] = str(lane.get("BANKING", "0") or "0")
                self.lane_curvature[oid] = str(lane.get("CURVATURE", "0") or "0")
                self.lane_traction[oid] = str(lane.get("TRACTION", "1") or "1")
                for pt in pts:
                    self.x_min = min(self.x_min, pt[0])
                    self.y_min = min(self.y_min, pt[1])
                    self.x_max = max(self.x_max, pt[0])
                    self.y_max = max(self.y_max, pt[1])
                d = sum(distance_2d(pts[j][0], pts[j][1],
                                    pts[j + 1][0], pts[j + 1][1])
                        for j in range(len(pts) - 1))
                if d > 0:
                    self.shortest_lane_dist = min(self.shortest_lane_dist, d)
                    self.longest_lane_dist = max(self.longest_lane_dist, d)

        if not self.lane_points:
            print("  Warning: No active autonomous lanes found at all.")
            return {"nodes": [], "roads": []}

        # ── Step 2: Intersection detection via QuadTree ──
        self._create_intersections()

        # ── Step 3: Scenario 1 – grow from junction points ──
        self._grow_roads()

        # ── Step 4: Scenario 2 – unvisited lanes with free endpoints ──
        unvisited = [lid for lid in self.lane_points if lid not in self.addressed]
        if unvisited:
            self._grow_leftout(unvisited, eight_shape=False)

        # ── Step 5: Scenario 3 – remaining loops / 8-shapes ──
        unvisited = [lid for lid in self.lane_points if lid not in self.addressed]
        if unvisited:
            self._grow_leftout(unvisited, eight_shape=True)

        # ── Step 6: Non-haulage standalone roads ──
        unvisited = [lid for lid in self.lane_points if lid not in self.addressed]
        if unvisited:
            self._create_non_haulage(unvisited)

        # ── Step 7: Build output JSON ──
        return self._build_output(fidelity)

    # ── JSON output ──

    def _build_output(self, fidelity):
        """Generate nodes[] and roads[] for output.json.

        Coordinates stored as [Y, X, Z] (swapped) matching C++ output.
        Node properties: rolling_resistance=1, traction from lane.
        """
        out_nid = 1
        coord_to_out = {}      # coord_key → output_node_id
        out_nodes = []
        out_roads = []

        for rid, lane_ids in self.linked_road_ids.items():
            node_list = self.linked_nodes.get(rid, [])
            if not node_list:
                continue

            # ── Optional coordinate reduction (Low fidelity) ──
            keep = None
            if fidelity.lower() == "low" and len(node_list) > 3:
                keep = {0, len(node_list) - 1, len(node_list) // 2}
                for idx, rn in enumerate(node_list):
                    if self._ck(rn.coords) in self.intersection_pts:
                        keep.add(idx)

            road_nids = []
            for idx, rn in enumerate(node_list):
                if keep is not None and idx not in keep:
                    continue

                ck = self._ck(rn.coords)
                if ck not in coord_to_out:
                    nid = out_nid
                    out_nid += 1
                    coord_to_out[ck] = nid

                    lid = rn.laneid
                    # C++ uses atoi() for speed_limit → int
                    try:
                        spd = int(float(self.lane_speed.get(lid, "0")))
                    except (ValueError, TypeError):
                        spd = 0
                    try:
                        wid = float(self.lane_width.get(lid, "0"))
                    except (ValueError, TypeError):
                        wid = 0
                    try:
                        bnk = float(self.lane_banking.get(lid, "0"))
                    except (ValueError, TypeError):
                        bnk = 0
                    try:
                        crv = float(self.lane_curvature.get(lid, "0"))
                    except (ValueError, TypeError):
                        crv = 0
                    try:
                        trc = float(self.lane_traction.get(lid, "1"))
                    except (ValueError, TypeError):
                        trc = 1

                    # C++ outputs [Y, X, Z]
                    # curvature stored as object {road_id: value} in C++
                    out_nodes.append({
                        "id": nid,
                        "coords": [rn.coords[1], rn.coords[0],
                                   rn.coords[2] if len(rn.coords) > 2 else 0],
                        "speed_limit": spd,
                        "lane_width": wid,
                        "banking": bnk,
                        "traction": 1,
                        "curvature": {str(rid): crv},
                        "rolling_resistance": 1,
                    })

                onid = coord_to_out[ck]
                if not road_nids or road_nids[-1] != onid:
                    road_nids.append(onid)

            if len(road_nids) < 2:
                continue

            first_lid = lane_ids[0] if lane_ids else None
            try:
                spd = float(self.lane_speed.get(first_lid, "0")) if first_lid else 0
            except (ValueError, TypeError):
                spd = 0

            # Build Lanes object: {lane_id_str: [node_ids]} for each lane in road
            lanes_obj = {}
            for lid in lane_ids:
                lane_node_list = self.linked_nodes.get(rid, [])
                lane_nids = []
                for rn in lane_node_list:
                    if rn.laneid == lid:
                        ck2 = self._ck(rn.coords)
                        onid2 = coord_to_out.get(ck2)
                        if onid2 and (not lane_nids or lane_nids[-1] != onid2):
                            lane_nids.append(onid2)
                if lane_nids:
                    lanes_obj[str(lid)] = lane_nids

            out_roads.append({
                "id": rid,
                "is_generated": False,
                "lanes_num": 1,
                "name": f"Road {rid}",
                "nodes": road_nids,
                "rolling_resistance": 1,
                "speed_limit": spd,
                "ways_num": 1,
                "Lanes": lanes_obj,
            })

        return {"nodes": out_nodes, "roads": out_roads}


# ── Public entry point ──

def create_road_network(output_path, fidelity="Low"):
    """Build road network from Lanes.json, write output.json.

    Main entry point matching C++ docreateroadfromjson().
    """
    print("[Stage 4] Creating road network...")

    lanes_path = os.path.join(output_path, "Lanes.json")
    with open(lanes_path, "r", encoding="utf-8") as f:
        lanes_data = json.load(f)

    builder = RoadNetworkBuilder()
    output = builder.build(lanes_data, fidelity)

    nodes = output["nodes"]
    roads = output["roads"]

    output_file = os.path.join(output_path, "output.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    print(f"  Nodes: {len(nodes)}, Roads: {len(roads)}")
    print(f"  Written: {output_file}")
    print("[Stage 4] Road network complete.")
