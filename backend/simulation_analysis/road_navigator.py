"""
Road Navigator Module

Provides road-constrained navigation for haulers, ensuring sequential node traversal
along roads with proper road switching when necessary.

Business Rules:
1. Hauler must move sequentially through nodes in road order
2. Within each road, hauler visits nodes one by one without skipping
3. Road switching allowed at:
   - Road endpoints (first or last node of a road)
   - Shared nodes (nodes that exist on multiple roads)
4. When current road doesn't contain next matched node, find a valid road that
   contains the sequential path the hauler has traveled
5. Accuracy is highest priority - never skip nodes for performance
"""

import math
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field


@dataclass
class RoadState:
    """Tracks hauler's current state on the road network."""
    current_road_id: Optional[int] = None
    current_node_index: int = -1  # Index within current road's nodes list
    direction: int = 1  # 1 = forward (increasing index), -1 = backward
    visited_nodes: List[int] = field(default_factory=list)  # Sequential history of visited nodes
    road_history: List[int] = field(default_factory=list)  # History of road IDs traversed


@dataclass
class NavigationResult:
    """Result of navigation to next node."""
    node_id: int
    node_name: str
    coords: Tuple[float, float, float]
    road_id: int
    distance_from_gps: float
    road_switched: bool = False
    intermediate_nodes: List[int] = field(default_factory=list)  # Nodes between last and current


class RoadNavigator:
    """
    Manages road-constrained navigation for a single hauler.

    Ensures hauler follows roads sequentially without skipping nodes,
    with proper road switching at endpoints.
    """

    def __init__(
        self,
        nodes: Dict[int, Dict[str, Any]],
        roads: List[Dict[str, Any]],
        max_search_distance: float = 100.0,
    ):
        """
        Initialize RoadNavigator.

        Args:
            nodes: Dictionary mapping node_id to node data
            roads: List of road dictionaries with 'id' and 'nodes' fields
            max_search_distance: Maximum distance to search for matching node (meters)
        """
        self.nodes = nodes
        self.roads = {r["id"]: r for r in roads}
        self.max_search_distance = max_search_distance

        # Build road lookup structures
        self._node_to_roads: Dict[int, List[int]] = {}  # node_id -> list of road_ids containing it
        self._road_node_indices: Dict[int, Dict[int, int]] = {}  # road_id -> {node_id: index}
        self._shared_nodes: Set[int] = set()  # Nodes that exist on multiple roads
        self._build_lookups()

        # Current navigation state
        self.state = RoadState()

    def _build_lookups(self) -> None:
        """Build lookup structures for efficient navigation."""
        for road_id, road in self.roads.items():
            road_nodes = road.get("nodes", [])
            self._road_node_indices[road_id] = {}

            for idx, node_id in enumerate(road_nodes):
                # Map node to roads
                if node_id not in self._node_to_roads:
                    self._node_to_roads[node_id] = []
                if road_id not in self._node_to_roads[node_id]:
                    self._node_to_roads[node_id].append(road_id)

                # Map node to index within road
                self._road_node_indices[road_id][node_id] = idx

        # Identify shared nodes (nodes that exist on multiple roads)
        for node_id, road_ids in self._node_to_roads.items():
            if len(road_ids) > 1:
                self._shared_nodes.add(node_id)

    def reset(self) -> None:
        """Reset navigation state for new hauler session."""
        self.state = RoadState()

    def get_road_history(self) -> List[int]:
        """Get history of road IDs traversed."""
        return self.state.road_history.copy()

    def get_visited_nodes(self) -> List[int]:
        """Get sequential history of visited nodes."""
        return self.state.visited_nodes.copy()

    def _get_node_coords(self, node_id: int) -> Optional[Tuple[float, float, float]]:
        """Get coordinates for a node."""
        node = self.nodes.get(node_id)
        if not node:
            return None
        coords = node.get("coords", [0, 0, 0])
        if len(coords) >= 3:
            return (coords[0], coords[1], coords[2])
        elif len(coords) >= 2:
            return (coords[0], coords[1], 0.0)
        return None

    def _calculate_distance(
        self, x1: float, y1: float, x2: float, y2: float
    ) -> float:
        """Calculate 2D Euclidean distance."""
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def _find_nearest_node_global(
        self, x: float, y: float, z: float
    ) -> Optional[Tuple[int, float]]:
        """
        Find nearest node globally without road constraint.

        Returns:
            Tuple of (node_id, distance) or None if no node within max_search_distance
        """
        min_dist = float('inf')
        nearest_id = None

        for node_id, node in self.nodes.items():
            coords = self._get_node_coords(node_id)
            if coords is None:
                continue

            dist = self._calculate_distance(x, y, coords[0], coords[1])
            if dist < min_dist and dist <= self.max_search_distance:
                min_dist = dist
                nearest_id = node_id

        if nearest_id is None:
            return None
        return (nearest_id, min_dist)

    def _get_road_nodes(self, road_id: int) -> List[int]:
        """Get ordered list of nodes for a road."""
        road = self.roads.get(road_id)
        if not road:
            return []
        return road.get("nodes", [])

    def _is_road_endpoint(self, road_id: int, node_id: int) -> bool:
        """Check if node is an endpoint (first or last) of a road."""
        road_nodes = self._get_road_nodes(road_id)
        if not road_nodes:
            return False
        return node_id == road_nodes[0] or node_id == road_nodes[-1]

    def _is_shared_node(self, node_id: int) -> bool:
        """Check if node exists on multiple roads (shared node)."""
        return node_id in self._shared_nodes

    def _is_valid_switch_point(self, road_id: int, node_id: int) -> bool:
        """
        Check if node is a valid point to switch roads.

        A node is valid for road switching if it's either:
        - An endpoint of the current road
        - A shared node (exists on multiple roads)
        """
        return self._is_road_endpoint(road_id, node_id) or self._is_shared_node(node_id)

    def _get_node_index_in_road(self, road_id: int, node_id: int) -> int:
        """Get index of node within road's node list. Returns -1 if not found."""
        return self._road_node_indices.get(road_id, {}).get(node_id, -1)

    def _find_road_containing_sequence(
        self, node_sequence: List[int]
    ) -> Optional[Tuple[int, int]]:
        """
        Find a road that contains the given sequence of nodes in order.

        Args:
            node_sequence: List of node IDs in traversal order

        Returns:
            Tuple of (road_id, direction) where direction is 1 (forward) or -1 (backward),
            or None if no road contains the sequence
        """
        if not node_sequence:
            return None

        # Get candidate roads that contain the first node
        first_node = node_sequence[0]
        candidate_roads = self._node_to_roads.get(first_node, [])

        for road_id in candidate_roads:
            road_nodes = self._get_road_nodes(road_id)
            if not road_nodes:
                continue

            # Check forward direction
            if self._sequence_in_road(node_sequence, road_nodes, forward=True):
                return (road_id, 1)

            # Check backward direction
            if self._sequence_in_road(node_sequence, road_nodes, forward=False):
                return (road_id, -1)

        return None

    def _sequence_in_road(
        self, sequence: List[int], road_nodes: List[int], forward: bool
    ) -> bool:
        """
        Check if sequence appears in road_nodes in given direction.

        Args:
            sequence: Node sequence to find
            road_nodes: Road's ordered node list
            forward: If True, check forward order; if False, check reverse order
        """
        if not sequence:
            return True

        if forward:
            # Check if sequence appears as consecutive nodes in road_nodes
            try:
                first_idx = road_nodes.index(sequence[0])
            except ValueError:
                return False

            for i, node_id in enumerate(sequence):
                expected_idx = first_idx + i
                if expected_idx >= len(road_nodes):
                    return False
                if road_nodes[expected_idx] != node_id:
                    return False
            return True
        else:
            # Check reverse order
            try:
                first_idx = road_nodes.index(sequence[0])
            except ValueError:
                return False

            for i, node_id in enumerate(sequence):
                expected_idx = first_idx - i
                if expected_idx < 0:
                    return False
                if road_nodes[expected_idx] != node_id:
                    return False
            return True

    def _get_intermediate_nodes(
        self, road_id: int, from_index: int, to_index: int, direction: int
    ) -> List[int]:
        """
        Get nodes between from_index and to_index (exclusive of endpoints).

        Args:
            road_id: Road ID
            from_index: Starting index in road's node list
            to_index: Ending index in road's node list
            direction: 1 for forward, -1 for backward

        Returns:
            List of intermediate node IDs (excluding from and to nodes)
        """
        road_nodes = self._get_road_nodes(road_id)
        if not road_nodes:
            return []

        intermediate = []
        if direction == 1:
            for i in range(from_index + 1, to_index):
                if 0 <= i < len(road_nodes):
                    intermediate.append(road_nodes[i])
        else:
            for i in range(from_index - 1, to_index, -1):
                if 0 <= i < len(road_nodes):
                    intermediate.append(road_nodes[i])

        return intermediate

    def _find_valid_road_for_switch(
        self, target_node_id: int, recent_nodes: List[int]
    ) -> Optional[Tuple[int, int, int]]:
        """
        Find a road that allows valid switch to reach target_node.

        Args:
            target_node_id: The node we want to reach
            recent_nodes: Recent node history to verify continuity

        Returns:
            Tuple of (road_id, direction, node_index) or None
        """
        # Get roads containing target node
        candidate_roads = self._node_to_roads.get(target_node_id, [])

        for road_id in candidate_roads:
            road_nodes = self._get_road_nodes(road_id)
            if not road_nodes:
                continue

            target_index = self._get_node_index_in_road(road_id, target_node_id)
            if target_index < 0:
                continue

            # Check if this road contains recent nodes in sequence
            # This validates the road switch is valid
            if recent_nodes:
                last_node = recent_nodes[-1]
                if last_node in road_nodes:
                    last_index = road_nodes.index(last_node)
                    # Determine direction based on indices
                    if target_index > last_index:
                        direction = 1
                    elif target_index < last_index:
                        direction = -1
                    else:
                        continue  # Same node, skip

                    # Verify sequence is valid
                    if self._verify_switch_validity(road_id, last_index, target_index, direction, recent_nodes):
                        return (road_id, direction, target_index)
                else:
                    # Last node not on this road - check if any recent node is a shared node
                    # that connects to this road (allows switch via shared node)
                    for i in range(len(recent_nodes) - 1, -1, -1):
                        node = recent_nodes[i]
                        if node in road_nodes and self._is_shared_node(node):
                            # Found a shared node that can serve as switch point
                            switch_index = road_nodes.index(node)
                            if target_index > switch_index:
                                direction = 1
                            elif target_index < switch_index:
                                direction = -1
                            else:
                                continue
                            return (road_id, direction, target_index)
            else:
                # No history, any road containing target is valid
                return (road_id, 1, target_index)

        return None

    def _verify_switch_validity(
        self,
        road_id: int,
        from_index: int,
        to_index: int,
        direction: int,
        recent_nodes: List[int],
    ) -> bool:
        """
        Verify that switching to this road maintains valid sequential traversal.

        For road switching to be valid:
        1. The last visited node must exist on the new road
        2. If multiple recent nodes exist on new road, they must be in correct order

        Note: We relaxed the requirement from "all recent nodes must be on new road"
        to "at least the last node must be on new road" to support roads that only
        partially overlap (share some nodes but not all).
        """
        road_nodes = self._get_road_nodes(road_id)
        if not road_nodes:
            return False

        if not recent_nodes:
            return True

        # At minimum, the last visited node should be on the new road
        last_node = recent_nodes[-1]
        if last_node not in road_nodes:
            return False

        # Check which recent nodes are on this road and verify their order
        nodes_on_road = [n for n in recent_nodes if n in road_nodes]
        if len(nodes_on_road) < 2:
            return True

        # Check they appear in correct order (monotonic indices)
        indices = [road_nodes.index(n) for n in nodes_on_road]

        # Indices should be monotonic in the direction of travel
        if direction == 1:
            return all(indices[i] <= indices[i+1] for i in range(len(indices)-1))
        else:
            return all(indices[i] >= indices[i+1] for i in range(len(indices)-1))

    def navigate_to_gps(
        self, x: float, y: float, z: float = 0.0
    ) -> Optional[NavigationResult]:
        """
        Navigate hauler towards GPS position, following road constraints.

        This is the main method that:
        1. Finds nearest node to GPS position
        2. Validates node is reachable via current road
        3. Handles road switching if necessary
        4. Returns nodes that must be visited (including intermediate nodes)

        Args:
            x: Easting coordinate (meters)
            y: Northing coordinate (meters)
            z: Elevation coordinate (meters)

        Returns:
            NavigationResult with next node and any intermediate nodes,
            or None if no valid navigation possible
        """
        # Find nearest node to GPS position
        nearest_result = self._find_nearest_node_global(x, y, z)
        if nearest_result is None:
            return None

        target_node_id, gps_distance = nearest_result

        # First navigation - initialize state
        if self.state.current_road_id is None:
            return self._handle_initial_navigation(target_node_id, gps_distance)

        # Check if target is on current road
        current_road_nodes = self._get_road_nodes(self.state.current_road_id)
        target_in_current_road = target_node_id in current_road_nodes

        if target_in_current_road:
            return self._handle_same_road_navigation(target_node_id, gps_distance)
        else:
            return self._handle_road_switch_navigation(target_node_id, gps_distance)

    def _handle_initial_navigation(
        self, target_node_id: int, gps_distance: float
    ) -> Optional[NavigationResult]:
        """Handle first navigation when no road is set."""
        # Find a road containing this node
        candidate_roads = self._node_to_roads.get(target_node_id, [])
        if not candidate_roads:
            return None

        # Use first available road
        road_id = candidate_roads[0]
        road_nodes = self._get_road_nodes(road_id)
        node_index = self._get_node_index_in_road(road_id, target_node_id)

        # Initialize state
        self.state.current_road_id = road_id
        self.state.current_node_index = node_index
        self.state.direction = 1  # Assume forward initially
        self.state.visited_nodes.append(target_node_id)
        if road_id not in self.state.road_history:
            self.state.road_history.append(road_id)

        coords = self._get_node_coords(target_node_id)
        node = self.nodes.get(target_node_id, {})

        return NavigationResult(
            node_id=target_node_id,
            node_name=node.get("name", f"Node_{target_node_id}"),
            coords=coords or (0, 0, 0),
            road_id=road_id,
            distance_from_gps=gps_distance,
            road_switched=False,
            intermediate_nodes=[],
        )

    def _handle_same_road_navigation(
        self, target_node_id: int, gps_distance: float
    ) -> Optional[NavigationResult]:
        """Handle navigation when target is on current road."""
        road_id = self.state.current_road_id
        road_nodes = self._get_road_nodes(road_id)

        current_index = self.state.current_node_index
        target_index = self._get_node_index_in_road(road_id, target_node_id)

        # Same node - no movement
        if target_index == current_index:
            coords = self._get_node_coords(target_node_id)
            node = self.nodes.get(target_node_id, {})
            return NavigationResult(
                node_id=target_node_id,
                node_name=node.get("name", f"Node_{target_node_id}"),
                coords=coords or (0, 0, 0),
                road_id=road_id,
                distance_from_gps=gps_distance,
                road_switched=False,
                intermediate_nodes=[],
            )

        # Determine direction
        if target_index > current_index:
            direction = 1
        else:
            direction = -1

        # Get intermediate nodes that must be visited
        intermediate = self._get_intermediate_nodes(
            road_id, current_index, target_index, direction
        )

        # Update state
        self.state.current_node_index = target_index
        self.state.direction = direction

        # Add intermediate nodes to visited history
        for node_id in intermediate:
            if not self.state.visited_nodes or self.state.visited_nodes[-1] != node_id:
                self.state.visited_nodes.append(node_id)

        # Add target node to visited history
        if not self.state.visited_nodes or self.state.visited_nodes[-1] != target_node_id:
            self.state.visited_nodes.append(target_node_id)

        coords = self._get_node_coords(target_node_id)
        node = self.nodes.get(target_node_id, {})

        return NavigationResult(
            node_id=target_node_id,
            node_name=node.get("name", f"Node_{target_node_id}"),
            coords=coords or (0, 0, 0),
            road_id=road_id,
            distance_from_gps=gps_distance,
            road_switched=False,
            intermediate_nodes=intermediate,
        )

    def _handle_road_switch_navigation(
        self, target_node_id: int, gps_distance: float
    ) -> Optional[NavigationResult]:
        """
        Handle navigation when target is NOT on current road.

        This implements the road switching logic:
        1. Find a road that contains recent visited nodes OR a shared node
        2. Calculate intermediate nodes:
           a. From current position to switch point (on current road)
           b. From switch point to target (on new road)
        3. Switch to new road
        """
        # Get recent visited nodes for validation
        recent_nodes = self.state.visited_nodes[-5:] if self.state.visited_nodes else []

        # Find valid road for switch
        switch_result = self._find_valid_road_for_switch(target_node_id, recent_nodes)

        if switch_result is None:
            # No valid road found - try fallback strategies
            return self._handle_road_switch_fallback(target_node_id, gps_distance)

        new_road_id, direction, target_index = switch_result
        new_road_nodes = self._get_road_nodes(new_road_id)
        current_road_nodes = self._get_road_nodes(self.state.current_road_id)

        # Calculate intermediate nodes
        intermediate = []

        # Find the switch point (shared node) between current and new road
        switch_node_id = None
        if recent_nodes and recent_nodes[-1] in new_road_nodes:
            # Last visited node is on new road - direct switch
            switch_node_id = recent_nodes[-1]
        else:
            # Find a shared node that connects current road to new road
            for node_id in current_road_nodes:
                if node_id in new_road_nodes and self._is_shared_node(node_id):
                    current_idx = self._get_node_index_in_road(self.state.current_road_id, node_id)
                    # Prefer shared node ahead of current position (in travel direction)
                    if self.state.direction == 1 and current_idx >= self.state.current_node_index:
                        switch_node_id = node_id
                        break
                    elif self.state.direction == -1 and current_idx <= self.state.current_node_index:
                        switch_node_id = node_id
                        break
            # If no shared node in travel direction, try any shared node
            if switch_node_id is None:
                for node_id in current_road_nodes:
                    if node_id in new_road_nodes and self._is_shared_node(node_id):
                        switch_node_id = node_id
                        break

        if switch_node_id:
            # Calculate intermediate nodes on CURRENT road to reach switch point
            switch_idx_current = self._get_node_index_in_road(self.state.current_road_id, switch_node_id)
            if switch_idx_current >= 0 and switch_idx_current != self.state.current_node_index:
                dir_to_switch = 1 if switch_idx_current > self.state.current_node_index else -1
                intermediate_current = self._get_intermediate_nodes(
                    self.state.current_road_id,
                    self.state.current_node_index,
                    switch_idx_current,
                    dir_to_switch
                )
                intermediate.extend(intermediate_current)
                # Add the switch node itself if it's not already in intermediate
                if switch_node_id not in intermediate:
                    intermediate.append(switch_node_id)

            # Calculate intermediate nodes on NEW road from switch point to target
            switch_idx_new = new_road_nodes.index(switch_node_id)
            intermediate_new = self._get_intermediate_nodes(
                new_road_id, switch_idx_new, target_index, direction
            )
            intermediate.extend(intermediate_new)

        # Update state for road switch
        old_road_id = self.state.current_road_id
        self.state.current_road_id = new_road_id
        self.state.current_node_index = target_index
        self.state.direction = direction

        if new_road_id not in self.state.road_history:
            self.state.road_history.append(new_road_id)

        # Add intermediate and target nodes to visited history
        for node_id in intermediate:
            if not self.state.visited_nodes or self.state.visited_nodes[-1] != node_id:
                self.state.visited_nodes.append(node_id)

        if not self.state.visited_nodes or self.state.visited_nodes[-1] != target_node_id:
            self.state.visited_nodes.append(target_node_id)

        coords = self._get_node_coords(target_node_id)
        node = self.nodes.get(target_node_id, {})

        return NavigationResult(
            node_id=target_node_id,
            node_name=node.get("name", f"Node_{target_node_id}"),
            coords=coords or (0, 0, 0),
            road_id=new_road_id,
            distance_from_gps=gps_distance,
            road_switched=(new_road_id != old_road_id),
            intermediate_nodes=intermediate,
        )

    def _handle_road_switch_fallback(
        self, target_node_id: int, gps_distance: float
    ) -> Optional[NavigationResult]:
        """
        Fallback handling when no direct road switch is valid.

        Tries multiple strategies:
        1. Switch at valid switch points (endpoints or shared nodes)
        2. Search for road that connects via shared nodes
        3. Use BFS to find multi-road path
        """
        current_road_nodes = self._get_road_nodes(self.state.current_road_id)
        if not current_road_nodes:
            return None

        current_node_id = current_road_nodes[self.state.current_node_index]

        # Check if current node is a valid switch point (endpoint OR shared node)
        is_valid_switch = self._is_valid_switch_point(self.state.current_road_id, current_node_id)

        if is_valid_switch:
            # At endpoint - allowed to switch to any road containing target
            candidate_roads = self._node_to_roads.get(target_node_id, [])

            for new_road_id in candidate_roads:
                new_road_nodes = self._get_road_nodes(new_road_id)
                if not new_road_nodes:
                    continue

                # Check if current endpoint connects to new road
                if current_node_id in new_road_nodes:
                    # Found connecting road
                    target_index = self._get_node_index_in_road(new_road_id, target_node_id)
                    current_index_in_new = new_road_nodes.index(current_node_id)

                    direction = 1 if target_index > current_index_in_new else -1
                    intermediate = self._get_intermediate_nodes(
                        new_road_id, current_index_in_new, target_index, direction
                    )

                    # Update state
                    self.state.current_road_id = new_road_id
                    self.state.current_node_index = target_index
                    self.state.direction = direction

                    if new_road_id not in self.state.road_history:
                        self.state.road_history.append(new_road_id)

                    for node_id in intermediate:
                        if not self.state.visited_nodes or self.state.visited_nodes[-1] != node_id:
                            self.state.visited_nodes.append(node_id)

                    if not self.state.visited_nodes or self.state.visited_nodes[-1] != target_node_id:
                        self.state.visited_nodes.append(target_node_id)

                    coords = self._get_node_coords(target_node_id)
                    node = self.nodes.get(target_node_id, {})

                    return NavigationResult(
                        node_id=target_node_id,
                        node_name=node.get("name", f"Node_{target_node_id}"),
                        coords=coords or (0, 0, 0),
                        road_id=new_road_id,
                        distance_from_gps=gps_distance,
                        road_switched=True,
                        intermediate_nodes=intermediate,
                    )

        # Last resort: find any connecting path (may require multiple road switches)
        return self._find_multi_road_path(target_node_id, gps_distance)

    def _find_multi_road_path(
        self, target_node_id: int, gps_distance: float
    ) -> Optional[NavigationResult]:
        """
        Find path through multiple roads to reach target.

        Uses BFS to find shortest path through road network.
        """
        current_road_nodes = self._get_road_nodes(self.state.current_road_id)
        if not current_road_nodes:
            return None

        current_node_id = current_road_nodes[self.state.current_node_index]

        # BFS to find path
        visited_nodes_set: Set[int] = set()
        queue: List[Tuple[int, List[int], List[int]]] = []  # (node_id, path, roads)

        queue.append((current_node_id, [current_node_id], [self.state.current_road_id]))
        visited_nodes_set.add(current_node_id)

        max_iterations = 1000
        iterations = 0

        while queue and iterations < max_iterations:
            iterations += 1
            node_id, path, road_path = queue.pop(0)

            # Check if we reached target
            if node_id == target_node_id:
                # Found path - update state and return
                intermediate = path[1:-1] if len(path) > 2 else []

                # Update road to last road in path
                final_road_id = road_path[-1]
                self.state.current_road_id = final_road_id
                self.state.current_node_index = self._get_node_index_in_road(final_road_id, target_node_id)

                for rid in road_path:
                    if rid not in self.state.road_history:
                        self.state.road_history.append(rid)

                for nid in intermediate:
                    if not self.state.visited_nodes or self.state.visited_nodes[-1] != nid:
                        self.state.visited_nodes.append(nid)

                if not self.state.visited_nodes or self.state.visited_nodes[-1] != target_node_id:
                    self.state.visited_nodes.append(target_node_id)

                coords = self._get_node_coords(target_node_id)
                node = self.nodes.get(target_node_id, {})

                return NavigationResult(
                    node_id=target_node_id,
                    node_name=node.get("name", f"Node_{target_node_id}"),
                    coords=coords or (0, 0, 0),
                    road_id=final_road_id,
                    distance_from_gps=gps_distance,
                    road_switched=True,
                    intermediate_nodes=intermediate,
                )

            # Explore neighbors via roads
            node_roads = self._node_to_roads.get(node_id, [])
            for road_id in node_roads:
                road_nodes = self._get_road_nodes(road_id)
                node_idx = road_nodes.index(node_id) if node_id in road_nodes else -1
                if node_idx < 0:
                    continue

                # Add adjacent nodes
                for adj_idx in [node_idx - 1, node_idx + 1]:
                    if 0 <= adj_idx < len(road_nodes):
                        adj_node = road_nodes[adj_idx]
                        if adj_node not in visited_nodes_set:
                            visited_nodes_set.add(adj_node)
                            new_path = path + [adj_node]
                            new_roads = road_path if road_id == road_path[-1] else road_path + [road_id]
                            queue.append((adj_node, new_path, new_roads))

        # No path found - return None
        return None

    def get_current_state(self) -> Dict[str, Any]:
        """Get current navigation state as dictionary."""
        return {
            "current_road_id": self.state.current_road_id,
            "current_node_index": self.state.current_node_index,
            "direction": self.state.direction,
            "visited_nodes_count": len(self.state.visited_nodes),
            "road_history": self.state.road_history.copy(),
            "last_visited_nodes": self.state.visited_nodes[-5:] if self.state.visited_nodes else [],
        }
