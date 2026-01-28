"""
Node Matcher Module

Matches GPS coordinates to road network nodes for event generation.
Uses spatial indexing for efficient nearest-neighbor lookup.
"""

import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class MatchedNode:
    """Result of node matching operation."""
    node_id: int
    node_name: str
    distance: float  # Distance from GPS point to node (meters)
    coords: Tuple[float, float, float]  # Node coordinates
    road_id: Optional[int] = None
    is_trolley: bool = False


class NodeMatcher:
    """
    Matches GPS coordinates to road network nodes.
    
    Uses a spatial grid for efficient nearest-neighbor lookup.
    """
    
    def __init__(
        self,
        nodes: List[Dict[str, Any]],
        roads: List[Dict[str, Any]] = None,
        grid_size: float = 50.0,
    ):
        """
        Initialize NodeMatcher with road network data.
        
        Args:
            nodes: List of node dictionaries with 'id', 'coords' fields
            roads: Optional list of road dictionaries for road association
            grid_size: Size of spatial grid cells in meters
        """
        self.nodes = {n["id"]: n for n in nodes}
        self.roads = roads or []
        self.grid_size = grid_size
        
        # Build spatial index
        self._spatial_grid: Dict[Tuple[int, int], List[int]] = {}
        self._build_spatial_index()
        
        # Build node-to-road mapping
        self._node_to_road: Dict[int, int] = {}
        self._build_node_road_mapping()
    
    def _build_spatial_index(self) -> None:
        """Build spatial grid index for fast nearest-neighbor lookup."""
        for node_id, node in self.nodes.items():
            coords = node.get("coords", [0, 0, 0])
            if len(coords) >= 2:
                grid_x = int(coords[0] / self.grid_size)
                grid_y = int(coords[1] / self.grid_size)
                key = (grid_x, grid_y)
                
                if key not in self._spatial_grid:
                    self._spatial_grid[key] = []
                self._spatial_grid[key].append(node_id)
    
    def _build_node_road_mapping(self) -> None:
        """Build mapping from node ID to road ID and node adjacency."""
        self._node_adjacency: Dict[int, List[int]] = {}  # node_id -> list of adjacent node IDs
        
        for road in self.roads:
            road_id = road.get("id")
            road_nodes = road.get("nodes", [])
            
            for i, node_id in enumerate(road_nodes):
                if node_id not in self._node_to_road:
                    self._node_to_road[node_id] = road_id
                
                # Build adjacency list
                if node_id not in self._node_adjacency:
                    self._node_adjacency[node_id] = []
                
                # Add previous node as adjacent
                if i > 0:
                    prev_node = road_nodes[i - 1]
                    if prev_node not in self._node_adjacency[node_id]:
                        self._node_adjacency[node_id].append(prev_node)
                
                # Add next node as adjacent
                if i < len(road_nodes) - 1:
                    next_node = road_nodes[i + 1]
                    if next_node not in self._node_adjacency[node_id]:
                        self._node_adjacency[node_id].append(next_node)
    
    def find_nearest_node(
        self,
        x: float,
        y: float,
        z: float = 0.0,
        max_distance: float = 100.0,
    ) -> Optional[MatchedNode]:
        """
        Find the nearest node to a GPS coordinate.
        
        Args:
            x: Easting coordinate (meters)
            y: Northing coordinate (meters)
            z: Elevation coordinate (meters)
            max_distance: Maximum distance to consider (meters)
        
        Returns:
            MatchedNode if found within max_distance, None otherwise
        """
        grid_x = int(x / self.grid_size)
        grid_y = int(y / self.grid_size)
        
        # Search in 3x3 neighborhood of grid cells
        candidates = []
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                key = (grid_x + dx, grid_y + dy)
                if key in self._spatial_grid:
                    candidates.extend(self._spatial_grid[key])
        
        if not candidates:
            # Fall back to full search if no candidates in nearby cells
            candidates = list(self.nodes.keys())
        
        # Find nearest node
        min_dist = float('inf')
        nearest_node = None
        
        for node_id in candidates:
            node = self.nodes.get(node_id)
            if not node:
                continue
            
            coords = node.get("coords", [0, 0, 0])
            if len(coords) < 2:
                continue
            
            # Calculate 2D distance (ignore elevation for matching)
            dist = math.sqrt(
                (coords[0] - x) ** 2 +
                (coords[1] - y) ** 2
            )
            
            if dist < min_dist and dist <= max_distance:
                min_dist = dist
                nearest_node = node
        
        if nearest_node is None:
            return None
        
        node_id = nearest_node["id"]
        coords = nearest_node.get("coords", [0, 0, 0])
        
        return MatchedNode(
            node_id=node_id,
            node_name=nearest_node.get("name", f"Node_{node_id}"),
            distance=min_dist,
            coords=tuple(coords[:3]) if len(coords) >= 3 else (coords[0], coords[1], 0.0),
            road_id=self._node_to_road.get(node_id),
            is_trolley=False,  # Trolley detection would require trolley zone data
        )
    
    def find_next_node_constrained(
        self,
        x: float,
        y: float,
        z: float,
        current_node_id: int,
        max_distance: float = 100.0,
    ) -> Optional[MatchedNode]:
        """
        Find next node constrained to adjacent nodes on the road network.
        
        This ensures the hauler follows the road topology instead of jumping
        to any nearest node.
        
        Args:
            x: Easting coordinate (meters)
            y: Northing coordinate (meters)
            z: Elevation coordinate (meters)
            current_node_id: Current node ID the hauler is at
            max_distance: Maximum distance to consider (meters)
        
        Returns:
            MatchedNode if found, None otherwise
        """
        # Get adjacent nodes from current position
        adjacent_nodes = self._node_adjacency.get(current_node_id, [])
        
        if not adjacent_nodes:
            # If no adjacency info, fall back to nearest node
            return self.find_nearest_node(x, y, z, max_distance)
        
        # Include current node (might still be at same node)
        candidates = [current_node_id] + adjacent_nodes
        
        # Find the candidate node closest to GPS position
        min_dist = float('inf')
        best_node = None
        
        for node_id in candidates:
            node = self.nodes.get(node_id)
            if not node:
                continue
            
            coords = node.get("coords", [0, 0, 0])
            if len(coords) < 2:
                continue
            
            dist = math.sqrt(
                (coords[0] - x) ** 2 +
                (coords[1] - y) ** 2
            )
            
            if dist < min_dist and dist <= max_distance:
                min_dist = dist
                best_node = node
        
        if best_node is None:
            # If no adjacent node is close enough, try nearest node
            # but only if it's within much smaller distance (likely road change)
            return self.find_nearest_node(x, y, z, max_distance / 2)
        
        node_id = best_node["id"]
        coords = best_node.get("coords", [0, 0, 0])
        
        return MatchedNode(
            node_id=node_id,
            node_name=best_node.get("name", f"Node_{node_id}"),
            distance=min_dist,
            coords=tuple(coords[:3]) if len(coords) >= 3 else (coords[0], coords[1], 0.0),
            road_id=self._node_to_road.get(node_id),
            is_trolley=False,
        )
    
    def get_adjacent_nodes(self, node_id: int) -> List[int]:
        """Get list of adjacent node IDs for a given node."""
        return self._node_adjacency.get(node_id, [])
    
    def find_nodes_along_path(
        self,
        path: List[Tuple[float, float, float]],
        min_node_distance: float = 10.0,
        max_search_distance: float = 50.0,
    ) -> List[MatchedNode]:
        """
        Find sequence of nodes along a GPS path.
        
        Args:
            path: List of (x, y, z) coordinates
            min_node_distance: Minimum distance between consecutive matched nodes
            max_search_distance: Maximum distance from path point to node
        
        Returns:
            List of MatchedNode objects representing the path through the network
        """
        if not path:
            return []
        
        matched_nodes = []
        last_node_id = None
        last_node_coords = None
        
        for point in path:
            x, y, z = point[0], point[1], point[2] if len(point) > 2 else 0.0
            
            match = self.find_nearest_node(x, y, z, max_search_distance)
            if match is None:
                continue
            
            # Skip if same node as last match
            if match.node_id == last_node_id:
                continue
            
            # Check minimum distance from last matched node
            if last_node_coords is not None:
                dist_from_last = math.sqrt(
                    (match.coords[0] - last_node_coords[0]) ** 2 +
                    (match.coords[1] - last_node_coords[1]) ** 2
                )
                if dist_from_last < min_node_distance:
                    continue
            
            matched_nodes.append(match)
            last_node_id = match.node_id
            last_node_coords = match.coords
        
        return matched_nodes
    
    def get_node_by_id(self, node_id: int) -> Optional[Dict[str, Any]]:
        """Get node data by ID."""
        return self.nodes.get(node_id)
    
    def calculate_segment_length(
        self,
        from_node_id: int,
        to_node_id: int,
    ) -> float:
        """
        Calculate distance between two nodes.
        
        Args:
            from_node_id: Starting node ID
            to_node_id: Ending node ID
        
        Returns:
            Distance in meters
        """
        from_node = self.nodes.get(from_node_id)
        to_node = self.nodes.get(to_node_id)
        
        if not from_node or not to_node:
            return 0.0
        
        from_coords = from_node.get("coords", [0, 0, 0])
        to_coords = to_node.get("coords", [0, 0, 0])
        
        return math.sqrt(
            (to_coords[0] - from_coords[0]) ** 2 +
            (to_coords[1] - from_coords[1]) ** 2 +
            (to_coords[2] - from_coords[2]) ** 2
        )
    
    def calculate_grade(
        self,
        from_node_id: int,
        to_node_id: int,
    ) -> float:
        """
        Calculate grade (slope) between two nodes.
        
        Args:
            from_node_id: Starting node ID
            to_node_id: Ending node ID
        
        Returns:
            Grade as percentage (positive = uphill)
        """
        from_node = self.nodes.get(from_node_id)
        to_node = self.nodes.get(to_node_id)
        
        if not from_node or not to_node:
            return 0.0
        
        from_coords = from_node.get("coords", [0, 0, 0])
        to_coords = to_node.get("coords", [0, 0, 0])
        
        horizontal_dist = math.sqrt(
            (to_coords[0] - from_coords[0]) ** 2 +
            (to_coords[1] - from_coords[1]) ** 2
        )
        
        if horizontal_dist == 0:
            return 0.0
        
        vertical_diff = to_coords[2] - from_coords[2] if len(to_coords) > 2 and len(from_coords) > 2 else 0.0
        
        return (vertical_diff / horizontal_dist) * 100
