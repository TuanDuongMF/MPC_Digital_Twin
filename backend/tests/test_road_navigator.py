"""
Unit tests for RoadNavigator module.

Tests road-constrained navigation logic including:
- Sequential node traversal
- Road switching at endpoints
- Intermediate node generation
- Multi-road path finding
"""

import unittest
from typing import Dict, List, Any

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from simulation_analysis.road_navigator import RoadNavigator, RoadState, NavigationResult


class TestRoadNavigator(unittest.TestCase):
    """Test cases for RoadNavigator class."""

    def setUp(self):
        """Set up test fixtures with a simple road network."""
        # Create a simple network:
        # Road 1: Node 1 -> Node 2 -> Node 3
        # Road 2: Node 3 -> Node 4 -> Node 5
        # Road 3: Node 5 -> Node 6 -> Node 7
        self.nodes = {
            1: {"id": 1, "name": "Node_1", "coords": [0.0, 0.0, 0.0]},
            2: {"id": 2, "name": "Node_2", "coords": [100.0, 0.0, 0.0]},
            3: {"id": 3, "name": "Node_3", "coords": [200.0, 0.0, 0.0]},
            4: {"id": 4, "name": "Node_4", "coords": [300.0, 0.0, 0.0]},
            5: {"id": 5, "name": "Node_5", "coords": [400.0, 0.0, 0.0]},
            6: {"id": 6, "name": "Node_6", "coords": [500.0, 0.0, 0.0]},
            7: {"id": 7, "name": "Node_7", "coords": [600.0, 0.0, 0.0]},
        }

        self.roads = [
            {"id": 1, "name": "Road_1", "nodes": [1, 2, 3]},
            {"id": 2, "name": "Road_2", "nodes": [3, 4, 5]},
            {"id": 3, "name": "Road_3", "nodes": [5, 6, 7]},
        ]

        self.navigator = RoadNavigator(
            nodes=self.nodes,
            roads=self.roads,
            max_search_distance=150.0,
        )

    def test_initial_navigation(self):
        """Test first navigation initializes state correctly."""
        result = self.navigator.navigate_to_gps(5.0, 0.0, 0.0)

        self.assertIsNotNone(result)
        self.assertEqual(result.node_id, 1)
        self.assertEqual(result.road_id, 1)
        self.assertFalse(result.road_switched)
        self.assertEqual(len(result.intermediate_nodes), 0)

        # Check state is initialized
        self.assertEqual(self.navigator.state.current_road_id, 1)
        self.assertEqual(len(self.navigator.state.visited_nodes), 1)
        self.assertIn(1, self.navigator.state.road_history)

    def test_sequential_navigation_same_road(self):
        """Test sequential navigation on same road without skipping nodes."""
        # Navigate to Node 1
        self.navigator.navigate_to_gps(0.0, 0.0, 0.0)

        # Navigate to Node 3 (should include Node 2 as intermediate)
        result = self.navigator.navigate_to_gps(200.0, 0.0, 0.0)

        self.assertIsNotNone(result)
        self.assertEqual(result.node_id, 3)
        self.assertEqual(result.road_id, 1)
        self.assertFalse(result.road_switched)
        # Node 2 should be in intermediate nodes
        self.assertIn(2, result.intermediate_nodes)

    def test_road_switch_at_endpoint(self):
        """Test road switching at road endpoint."""
        # Navigate through Road 1 to Node 3
        self.navigator.navigate_to_gps(0.0, 0.0, 0.0)  # Node 1
        self.navigator.navigate_to_gps(100.0, 0.0, 0.0)  # Node 2
        self.navigator.navigate_to_gps(200.0, 0.0, 0.0)  # Node 3

        # Navigate to Node 4 (should switch to Road 2)
        result = self.navigator.navigate_to_gps(300.0, 0.0, 0.0)

        self.assertIsNotNone(result)
        self.assertEqual(result.node_id, 4)
        self.assertEqual(result.road_id, 2)
        self.assertTrue(result.road_switched)

    def test_road_history_tracking(self):
        """Test road history is tracked correctly."""
        # Navigate through multiple roads
        self.navigator.navigate_to_gps(0.0, 0.0, 0.0)    # Node 1, Road 1
        self.navigator.navigate_to_gps(200.0, 0.0, 0.0)  # Node 3, Road 1
        self.navigator.navigate_to_gps(400.0, 0.0, 0.0)  # Node 5, Road 2
        self.navigator.navigate_to_gps(600.0, 0.0, 0.0)  # Node 7, Road 3

        history = self.navigator.get_road_history()
        self.assertIn(1, history)
        self.assertIn(2, history)
        self.assertIn(3, history)

    def test_visited_nodes_tracking(self):
        """Test visited nodes are tracked in sequence."""
        self.navigator.navigate_to_gps(0.0, 0.0, 0.0)    # Node 1
        self.navigator.navigate_to_gps(100.0, 0.0, 0.0)  # Node 2
        self.navigator.navigate_to_gps(200.0, 0.0, 0.0)  # Node 3

        visited = self.navigator.get_visited_nodes()
        self.assertEqual(visited, [1, 2, 3])

    def test_same_node_navigation(self):
        """Test navigating to same node doesn't duplicate entries."""
        self.navigator.navigate_to_gps(0.0, 0.0, 0.0)  # Node 1
        self.navigator.navigate_to_gps(5.0, 0.0, 0.0)  # Still near Node 1

        visited = self.navigator.get_visited_nodes()
        self.assertEqual(len(visited), 1)

    def test_backward_direction(self):
        """Test backward navigation along road."""
        # Navigate forward first
        self.navigator.navigate_to_gps(0.0, 0.0, 0.0)    # Node 1
        self.navigator.navigate_to_gps(100.0, 0.0, 0.0)  # Node 2
        self.navigator.navigate_to_gps(200.0, 0.0, 0.0)  # Node 3

        # Navigate backward
        result = self.navigator.navigate_to_gps(100.0, 0.0, 0.0)  # Node 2

        self.assertIsNotNone(result)
        self.assertEqual(result.node_id, 2)
        self.assertEqual(self.navigator.state.direction, -1)

    def test_no_node_within_distance(self):
        """Test returns None when no node within max distance."""
        # Far from any node
        result = self.navigator.navigate_to_gps(10000.0, 10000.0, 0.0)
        self.assertIsNone(result)

    def test_reset(self):
        """Test reset clears state."""
        self.navigator.navigate_to_gps(0.0, 0.0, 0.0)
        self.navigator.navigate_to_gps(100.0, 0.0, 0.0)

        self.navigator.reset()

        self.assertIsNone(self.navigator.state.current_road_id)
        self.assertEqual(len(self.navigator.state.visited_nodes), 0)
        self.assertEqual(len(self.navigator.state.road_history), 0)

    def test_get_current_state(self):
        """Test get_current_state returns correct information."""
        self.navigator.navigate_to_gps(0.0, 0.0, 0.0)
        self.navigator.navigate_to_gps(100.0, 0.0, 0.0)

        state = self.navigator.get_current_state()

        self.assertEqual(state["current_road_id"], 1)
        self.assertIn("visited_nodes_count", state)
        self.assertIn("road_history", state)
        self.assertIn("last_visited_nodes", state)


class TestRoadNavigatorComplexNetwork(unittest.TestCase):
    """Test cases for RoadNavigator with more complex road network."""

    def setUp(self):
        """Set up a branching road network."""
        #     Node 4
        #       |
        # Node 1 - Node 2 - Node 3
        #       |
        #     Node 5
        self.nodes = {
            1: {"id": 1, "name": "Node_1", "coords": [0.0, 0.0, 0.0]},
            2: {"id": 2, "name": "Node_2", "coords": [100.0, 0.0, 0.0]},
            3: {"id": 3, "name": "Node_3", "coords": [200.0, 0.0, 0.0]},
            4: {"id": 4, "name": "Node_4", "coords": [100.0, 100.0, 0.0]},
            5: {"id": 5, "name": "Node_5", "coords": [100.0, -100.0, 0.0]},
        }

        self.roads = [
            {"id": 1, "name": "Main_Road", "nodes": [1, 2, 3]},
            {"id": 2, "name": "North_Branch", "nodes": [2, 4]},
            {"id": 3, "name": "South_Branch", "nodes": [2, 5]},
        ]

        self.navigator = RoadNavigator(
            nodes=self.nodes,
            roads=self.roads,
            max_search_distance=150.0,
        )

    def test_branch_navigation(self):
        """Test navigation to branch road."""
        self.navigator.navigate_to_gps(0.0, 0.0, 0.0)    # Node 1
        self.navigator.navigate_to_gps(100.0, 0.0, 0.0)  # Node 2

        # Navigate to Node 4 (north branch)
        result = self.navigator.navigate_to_gps(100.0, 100.0, 0.0)

        self.assertIsNotNone(result)
        self.assertEqual(result.node_id, 4)
        self.assertEqual(result.road_id, 2)

    def test_shared_node_navigation(self):
        """Test navigation through node shared by multiple roads."""
        self.navigator.navigate_to_gps(0.0, 0.0, 0.0)  # Node 1 on Road 1

        # Node 2 is shared by roads 1, 2, 3
        result = self.navigator.navigate_to_gps(100.0, 0.0, 0.0)

        self.assertIsNotNone(result)
        self.assertEqual(result.node_id, 2)


class TestRoadNavigatorEdgeCases(unittest.TestCase):
    """Test edge cases for RoadNavigator."""

    def test_empty_roads(self):
        """Test with empty roads list."""
        nodes = {1: {"id": 1, "name": "Node_1", "coords": [0.0, 0.0, 0.0]}}
        roads = []

        navigator = RoadNavigator(nodes=nodes, roads=roads)
        result = navigator.navigate_to_gps(0.0, 0.0, 0.0)

        # Should still find node but with no road association
        self.assertIsNone(result)  # No road means can't navigate

    def test_single_node_road(self):
        """Test road with single node."""
        nodes = {1: {"id": 1, "name": "Node_1", "coords": [0.0, 0.0, 0.0]}}
        roads = [{"id": 1, "name": "Road_1", "nodes": [1]}]

        navigator = RoadNavigator(nodes=nodes, roads=roads)
        result = navigator.navigate_to_gps(0.0, 0.0, 0.0)

        self.assertIsNotNone(result)
        self.assertEqual(result.node_id, 1)

    def test_circular_road(self):
        """Test navigation on circular road (first node = last node)."""
        nodes = {
            1: {"id": 1, "name": "Node_1", "coords": [0.0, 0.0, 0.0]},
            2: {"id": 2, "name": "Node_2", "coords": [100.0, 0.0, 0.0]},
            3: {"id": 3, "name": "Node_3", "coords": [100.0, 100.0, 0.0]},
        }
        # Circular: 1 -> 2 -> 3 -> 1
        roads = [{"id": 1, "name": "Circle", "nodes": [1, 2, 3, 1]}]

        navigator = RoadNavigator(nodes=nodes, roads=roads)

        navigator.navigate_to_gps(0.0, 0.0, 0.0)      # Node 1
        navigator.navigate_to_gps(100.0, 0.0, 0.0)    # Node 2
        navigator.navigate_to_gps(100.0, 100.0, 0.0)  # Node 3

        # Back to Node 1 (should work on circular road)
        result = navigator.navigate_to_gps(0.0, 0.0, 0.0)

        self.assertIsNotNone(result)
        self.assertEqual(result.node_id, 1)


if __name__ == '__main__':
    unittest.main()
