"""
Event Generator Module

Generates simulation events from processed AMT telemetry data.
Events follow the structure defined in events_structure_specification.md
"""

import copy
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

from .constants import (
    HaulerState,
    LocationType,
    SpeedLimitSource,
    SegmentTypeMapping,
    SegmentToLocation,
    EventType,
    DEFAULT_HAULER_STRUCT,
    DEFAULT_NODE_STRUCT,
    DEFAULT_LOADER_STRUCT,
    PAYLOAD_THRESHOLD,
    DEFAULT_PAYLOAD_CAPACITY,
)
from .node_matcher import NodeMatcher, MatchedNode


class EventGenerator:
    """
    Generates simulation events from processed telemetry data.
    """
    
    def __init__(
        self,
        node_matcher: NodeMatcher,
        simulation_start_time: Optional[datetime] = None,
    ):
        """
        Initialize EventGenerator.
        
        Args:
            node_matcher: NodeMatcher instance for road network lookup
            simulation_start_time: Reference time for event timestamps (default: first message time)
        """
        self.node_matcher = node_matcher
        self.simulation_start_time = simulation_start_time
        
        # Event counter
        self._event_id = 0
        
        # Hauler tracking state
        self._hauler_states: Dict[int, Dict[str, Any]] = {}
    
    def _get_next_event_id(self) -> int:
        """Get next sequential event ID."""
        self._event_id += 1
        return self._event_id
    
    def _time_to_minutes(self, event_time: datetime) -> float:
        """
        Convert datetime to minutes from simulation start.
        
        Args:
            event_time: Event timestamp
        
        Returns:
            Time in minutes from simulation start
        """
        if self.simulation_start_time is None:
            self.simulation_start_time = event_time
            return 0.0
        
        delta = event_time - self.simulation_start_time
        return delta.total_seconds() / 60.0
    
    def _create_hauler_struct(
        self,
        machine_id: int,
        machine_name: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Create hauler struct with default values and overrides.
        
        Args:
            machine_id: Machine ID
            machine_name: Machine name
            **kwargs: Field overrides
        
        Returns:
            Hauler struct dictionary
        """
        hauler = copy.deepcopy(DEFAULT_HAULER_STRUCT)
        hauler["id"] = machine_id
        hauler["uid"] = machine_id
        hauler["name"] = machine_name
        
        # Apply overrides
        for key, value in kwargs.items():
            if key in hauler:
                hauler[key] = value
        
        return hauler
    
    def _create_node_struct(
        self,
        node_id: int,
        node_name: str = None,
        is_trolley: bool = False,
    ) -> Dict[str, Any]:
        """
        Create node struct.
        
        Args:
            node_id: Node ID
            node_name: Node name (auto-generated if None)
            is_trolley: Is trolley node flag
        
        Returns:
            Node struct dictionary
        """
        return {
            "id": node_id,
            "name": node_name or f"Node_{node_id}",
            "isTrolley": is_trolley,
        }
    
    def _create_base_event(
        self,
        event_type: str,
        event_time: datetime,
        log_level: str = "record",
    ) -> Dict[str, Any]:
        """
        Create base event structure.
        
        Args:
            event_type: Event type name
            event_time: Event timestamp
            log_level: Log level ("record" or "debug")
        
        Returns:
            Base event dictionary
        """
        return {
            "eid": self._get_next_event_id(),
            "time": self._time_to_minutes(event_time),
            "etype": event_type,
            "log_level": log_level,
            "hauler": None,
            "node": None,
            "trolley": None,
            "charger": None,
            "loader": None,
            "delay": None,
            "ess": None,
            "zone": None,
            "battery_hauler": None,
            "crusher": None,
        }
    
    def _infer_hauler_state(
        self,
        payload_percent: float,
        speed: float,
        segment_type: str = None,
        is_in_zone: bool = False,
    ) -> HaulerState:
        """
        Infer hauler state from telemetry data.
        
        Args:
            payload_percent: Payload percentage (0-100)
            speed: Current speed (km/h)
            segment_type: Segment type from AMT data
            is_in_zone: Whether hauler is in a zone
        
        Returns:
            Inferred HaulerState
        """
        # Check segment type mapping first
        if segment_type and segment_type in SegmentTypeMapping:
            return SegmentTypeMapping[segment_type]
        
        # Speed-based inference
        if speed <= 1.0:
            # Stopped
            if is_in_zone:
                if payload_percent < PAYLOAD_THRESHOLD:
                    return HaulerState.LOADING
                else:
                    return HaulerState.DUMPING
            else:
                return HaulerState.QUEUING
        
        # Moving
        if payload_percent >= PAYLOAD_THRESHOLD:
            return HaulerState.TRAVEL_LOADED
        else:
            return HaulerState.TRAVEL_UNLOADED
    
    def _infer_location(
        self,
        segment_type: str = None,
        payload_percent: float = 0,
    ) -> Tuple[LocationType, int]:
        """
        Infer location type from segment data.
        
        Args:
            segment_type: Segment type from AMT data
            payload_percent: Payload percentage
        
        Returns:
            Tuple of (LocationType, location_id)
        """
        if segment_type and segment_type in SegmentToLocation:
            location = SegmentToLocation[segment_type]
            # Use -1 for unknown zone IDs
            location_id = -1 if location == LocationType.ROUTE else 0
            return location, location_id
        
        return LocationType.ROUTE, -1
    
    def generate_hauler_init_event(
        self,
        machine_id: int,
        machine_name: str,
        event_time: datetime,
        start_node: MatchedNode,
        payload_percent: float = 0,
        segment_type: str = None,
    ) -> Dict[str, Any]:
        """
        Generate HaulerInit event.
        
        Args:
            machine_id: Machine ID
            machine_name: Machine name
            event_time: Event timestamp
            start_node: Starting node
            payload_percent: Initial payload percentage
            segment_type: Initial segment type
        
        Returns:
            HaulerInit event dictionary
        """
        event = self._create_base_event(EventType.HAULER_INIT, event_time)
        
        hauler_state = self._infer_hauler_state(payload_percent, 0, segment_type)
        location, location_id = self._infer_location(segment_type, payload_percent)
        
        # Convert payload percent to tonnes
        payload_tonnes = (payload_percent / 100.0) * DEFAULT_PAYLOAD_CAPACITY
        
        event["hauler"] = self._create_hauler_struct(
            machine_id=machine_id,
            machine_name=machine_name,
            speed=0.0,
            payload=payload_tonnes,
            hauler_state=hauler_state,
            location=location,
            location_id=location_id,
        )
        
        event["node"] = self._create_node_struct(
            node_id=start_node.node_id,
            node_name=start_node.node_name,
            is_trolley=start_node.is_trolley,
        )
        
        # Initialize hauler tracking state
        self._hauler_states[machine_id] = {
            "last_node_id": start_node.node_id,
            "last_event_time": event_time,
            "distance": 0.0,
            "cycle_distance": 0.0,
            "route_distance": 0.0,
            "cycle_count": 0,
            "route_count": 0,
            "last_payload": payload_percent,
        }
        
        return event
    
    def generate_node_arrive_event(
        self,
        machine_id: int,
        machine_name: str,
        event_time: datetime,
        node: MatchedNode,
        speed: float,
        payload_percent: float,
        segment_type: str = None,
        orientation: str = "forward",
    ) -> Dict[str, Any]:
        """
        Generate HaulerNodeArrive event.

        Args:
            machine_id: Machine ID
            machine_name: Machine name
            event_time: Event timestamp
            node: Arrival node
            speed: Speed at arrival (km/h)
            payload_percent: Payload percentage
            segment_type: Current segment type
            orientation: Direction of travel ("forward" or "reverse")

        Returns:
            HaulerNodeArrive event dictionary
        """
        event = self._create_base_event(EventType.HAULER_NODE_ARRIVE, event_time)
        
        # Get hauler tracking state
        state = self._hauler_states.get(machine_id, {})
        last_node_id = state.get("last_node_id")
        last_event_time = state.get("last_event_time", event_time)
        
        # Calculate segment metrics
        seg_length = 0.0
        grade = 0.0
        if last_node_id:
            seg_length = self.node_matcher.calculate_segment_length(last_node_id, node.node_id)
            grade = self.node_matcher.calculate_grade(last_node_id, node.node_id)
        
        # Update distance tracking
        distance = state.get("distance", 0) + seg_length
        cycle_distance = state.get("cycle_distance", 0) + seg_length
        route_distance = state.get("route_distance", 0) + seg_length
        
        # Calculate time metrics
        delta_time = (event_time - last_event_time).total_seconds() / 60.0
        travel_time = delta_time if speed > 0 else 0
        wait_time = delta_time if speed == 0 else 0
        
        # Infer state
        hauler_state = self._infer_hauler_state(payload_percent, speed, segment_type)
        location, location_id = self._infer_location(segment_type, payload_percent)
        
        # Convert payload
        payload_tonnes = (payload_percent / 100.0) * DEFAULT_PAYLOAD_CAPACITY
        
        # Detect cycle transitions
        cycle_count = state.get("cycle_count", 0)
        route_count = state.get("route_count", 0)
        last_payload = state.get("last_payload", 0)
        
        # New cycle when transitioning from loaded to empty (dump complete)
        if last_payload >= PAYLOAD_THRESHOLD and payload_percent < PAYLOAD_THRESHOLD:
            cycle_distance = 0  # Reset cycle distance
        
        # Increment cycle count when loading starts
        if last_payload < PAYLOAD_THRESHOLD and payload_percent >= PAYLOAD_THRESHOLD:
            cycle_count += 1

        event["hauler"] = self._create_hauler_struct(
            machine_id=machine_id,
            machine_name=machine_name,
            speed=speed,
            payload=payload_tonnes,
            hauler_state=hauler_state,
            location=location,
            location_id=location_id,
            orientation=orientation,
            seglength=seg_length,
            physicalgrade=grade,
            totalgrade=grade,  # Simplified; should include rolling resistance
            distance=distance,
            cycle_distance=cycle_distance,
            route_distance=route_distance,
            cycle_count=cycle_count,
            route_count=route_count,
            hauler_delta_time=delta_time,
            travel_time_node_to_node=travel_time,
            wait_time_node_to_node=wait_time,
        )
        
        event["node"] = self._create_node_struct(
            node_id=node.node_id,
            node_name=node.node_name,
            is_trolley=node.is_trolley,
        )
        
        # Update tracking state
        self._hauler_states[machine_id] = {
            "last_node_id": node.node_id,
            "last_event_time": event_time,
            "distance": distance,
            "cycle_distance": cycle_distance,
            "route_distance": route_distance,
            "cycle_count": cycle_count,
            "route_count": route_count,
            "last_payload": payload_percent,
        }
        
        return event
    
    def generate_node_leave_event(
        self,
        machine_id: int,
        machine_name: str,
        event_time: datetime,
        node: MatchedNode,
        speed: float,
        payload_percent: float,
        segment_type: str = None,
        orientation: str = "forward",
    ) -> Dict[str, Any]:
        """
        Generate HaulerNodeLeave event.

        Args:
            machine_id: Machine ID
            machine_name: Machine name
            event_time: Event timestamp
            node: Departure node
            speed: Departure speed (km/h)
            payload_percent: Payload percentage
            segment_type: Current segment type
            orientation: Direction of travel ("forward" or "reverse")

        Returns:
            HaulerNodeLeave event dictionary
        """
        event = self._create_base_event(EventType.HAULER_NODE_LEAVE, event_time)

        state = self._hauler_states.get(machine_id, {})
        hauler_state = self._infer_hauler_state(payload_percent, speed, segment_type)
        location, location_id = self._infer_location(segment_type, payload_percent)

        payload_tonnes = (payload_percent / 100.0) * DEFAULT_PAYLOAD_CAPACITY

        event["hauler"] = self._create_hauler_struct(
            machine_id=machine_id,
            machine_name=machine_name,
            speed=speed,
            payload=payload_tonnes,
            hauler_state=hauler_state,
            location=location,
            location_id=location_id,
            orientation=orientation,
            distance=state.get("distance", 0),
            cycle_distance=state.get("cycle_distance", 0),
            route_distance=state.get("route_distance", 0),
            cycle_count=state.get("cycle_count", 0),
            route_count=state.get("route_count", 0),
        )
        
        event["node"] = self._create_node_struct(
            node_id=node.node_id,
            node_name=node.node_name,
            is_trolley=node.is_trolley,
        )
        
        return event
    
    def _create_loader_struct(
        self,
        loader_name: str,
        hauler_name: str = "",
        time_duration: float = 0.0,
        power: float = 0.0,
        indv_payload: float = 0.0,
    ) -> Dict[str, Any]:
        """Create loader struct per events_structure_specification.md 3.8."""
        loader = copy.deepcopy(DEFAULT_LOADER_STRUCT)
        loader["name"] = loader_name
        loader["hauler_name"] = hauler_name
        loader["time_duration"] = time_duration
        loader["power"] = power
        loader["indv_payload"] = indv_payload
        return loader
    
    def generate_loader_cycle_events_for_one_bucket(
        self,
        loader_name: str,
        load_zone_id: int,
        hauler_id: int,
        hauler_name: str,
        t_start: datetime,
        t_end: datetime,
        indv_payload_tonnes: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Generate 8 loader cycle events for one bucket (Dig -> Swing -> Load -> Return).
        Times are evenly distributed in [t_start, t_end].
        """
        if t_end <= t_start:
            return []
        total_sec = (t_end - t_start).total_seconds()
        step_sec = total_sec / 8.0
        events_out = []
        cycle_duration_min = total_sec / 60.0
        bucket_duration_min = cycle_duration_min / 8.0
        
        cycle_etypes = [
            EventType.LOADER_CYCLE_DIG_START,
            EventType.LOADER_CYCLE_DIG_END,
            EventType.LOADER_CYCLE_SWING_START,
            EventType.LOADER_CYCLE_SWING_END,
            EventType.LOADER_CYCLE_LOAD_START,
            EventType.LOADER_CYCLE_LOAD_END,
            EventType.LOADER_CYCLE_RETURN_START,
            EventType.LOADER_CYCLE_RETURN_END,
        ]
        for i in range(8):
            t = t_start + timedelta(seconds=step_sec * i)
            loader_struct = self._create_loader_struct(
                loader_name=loader_name,
                hauler_name=hauler_name,
                time_duration=bucket_duration_min,
                power=0.0,
                indv_payload=indv_payload_tonnes,
            )
            ev = self._create_base_event(cycle_etypes[i], t)
            ev["loader"] = loader_struct
            ev["ess"] = None
            if i % 2 == 1:
                # End events: include hauler struct
                ev["hauler"] = self._create_hauler_struct(
                    hauler_id,
                    hauler_name,
                    speed=0.0,
                    payload=0.0,
                    hauler_state=HaulerState.LOADING,
                    location=LocationType.LOAD,
                    location_id=load_zone_id,
                )
            events_out.append(ev)
        
        return events_out
    
    def generate_load_start_event(
        self,
        machine_id: int,
        machine_name: str,
        event_time: datetime,
        payload_percent: float,
    ) -> Dict[str, Any]:
        """Generate HaulerLoadStart event."""
        event = self._create_base_event(EventType.HAULER_LOAD_START, event_time)
        
        state = self._hauler_states.get(machine_id, {})
        
        # Increment cycle count at load start
        cycle_count = state.get("cycle_count", 0) + 1
        
        payload_tonnes = (payload_percent / 100.0) * DEFAULT_PAYLOAD_CAPACITY
        
        event["hauler"] = self._create_hauler_struct(
            machine_id=machine_id,
            machine_name=machine_name,
            speed=0.0,
            payload=payload_tonnes,
            hauler_state=HaulerState.LOADING,
            location=LocationType.LOAD,
            location_id=0,
            cycle_count=cycle_count,
            distance=state.get("distance", 0),
        )
        
        # Update state
        if machine_id in self._hauler_states:
            self._hauler_states[machine_id]["cycle_count"] = cycle_count
            self._hauler_states[machine_id]["cycle_distance"] = 0
        
        return event
    
    def generate_load_end_event(
        self,
        machine_id: int,
        machine_name: str,
        event_time: datetime,
        payload_percent: float,
    ) -> Dict[str, Any]:
        """Generate HaulerLoadEnd event."""
        event = self._create_base_event(EventType.HAULER_LOAD_END, event_time)
        
        state = self._hauler_states.get(machine_id, {})
        payload_tonnes = (payload_percent / 100.0) * DEFAULT_PAYLOAD_CAPACITY
        
        event["hauler"] = self._create_hauler_struct(
            machine_id=machine_id,
            machine_name=machine_name,
            speed=0.0,
            payload=payload_tonnes,
            hauler_state=HaulerState.LOADING,
            location=LocationType.LOAD,
            location_id=0,
            cycle_count=state.get("cycle_count", 0),
            distance=state.get("distance", 0),
        )
        
        return event
    
    def generate_dump_start_event(
        self,
        machine_id: int,
        machine_name: str,
        event_time: datetime,
        payload_percent: float,
    ) -> Dict[str, Any]:
        """Generate HaulerDumpStart event."""
        event = self._create_base_event(EventType.HAULER_DUMP_START, event_time)
        
        state = self._hauler_states.get(machine_id, {})
        payload_tonnes = (payload_percent / 100.0) * DEFAULT_PAYLOAD_CAPACITY
        
        event["hauler"] = self._create_hauler_struct(
            machine_id=machine_id,
            machine_name=machine_name,
            speed=0.0,
            payload=payload_tonnes,
            hauler_state=HaulerState.DUMPING,
            location=LocationType.DUMP,
            location_id=0,
            cycle_count=state.get("cycle_count", 0),
            distance=state.get("distance", 0),
        )
        
        return event
    
    def generate_dump_end_event(
        self,
        machine_id: int,
        machine_name: str,
        event_time: datetime,
        payload_percent: float,
    ) -> Dict[str, Any]:
        """Generate HaulerDumpEnd event."""
        event = self._create_base_event(EventType.HAULER_DUMP_END, event_time)
        
        state = self._hauler_states.get(machine_id, {})
        payload_tonnes = (payload_percent / 100.0) * DEFAULT_PAYLOAD_CAPACITY
        
        event["hauler"] = self._create_hauler_struct(
            machine_id=machine_id,
            machine_name=machine_name,
            speed=0.0,
            payload=payload_tonnes,
            hauler_state=HaulerState.DUMPING,
            location=LocationType.DUMP,
            location_id=0,
            cycle_count=state.get("cycle_count", 0),
            distance=state.get("distance", 0),
        )
        
        # Reset cycle distance after dump
        if machine_id in self._hauler_states:
            self._hauler_states[machine_id]["route_distance"] = 0
        
        return event
    
    def generate_idle_start_event(
        self,
        machine_id: int,
        machine_name: str,
        event_time: datetime,
        node: MatchedNode,
        payload_percent: float,
    ) -> Dict[str, Any]:
        """Generate HaulerIdleStart event."""
        event = self._create_base_event(EventType.HAULER_IDLE_START, event_time)
        
        state = self._hauler_states.get(machine_id, {})
        payload_tonnes = (payload_percent / 100.0) * DEFAULT_PAYLOAD_CAPACITY
        
        event["hauler"] = self._create_hauler_struct(
            machine_id=machine_id,
            machine_name=machine_name,
            speed=0.0,
            payload=payload_tonnes,
            hauler_state=HaulerState.QUEUING,
            location=LocationType.ROUTE,
            location_id=-1,
            cycle_count=state.get("cycle_count", 0),
            distance=state.get("distance", 0),
        )
        
        event["node"] = self._create_node_struct(
            node_id=node.node_id,
            node_name=node.node_name,
            is_trolley=node.is_trolley,
        )
        
        return event
    
    def generate_idle_end_event(
        self,
        machine_id: int,
        machine_name: str,
        event_time: datetime,
        node: MatchedNode,
        payload_percent: float,
    ) -> Dict[str, Any]:
        """Generate HaulerIdleEnd event."""
        event = self._create_base_event(EventType.HAULER_IDLE_END, event_time)
        
        state = self._hauler_states.get(machine_id, {})
        payload_tonnes = (payload_percent / 100.0) * DEFAULT_PAYLOAD_CAPACITY
        
        hauler_state = HaulerState.TRAVEL_LOADED if payload_percent >= PAYLOAD_THRESHOLD else HaulerState.TRAVEL_UNLOADED
        
        event["hauler"] = self._create_hauler_struct(
            machine_id=machine_id,
            machine_name=machine_name,
            speed=0.0,
            payload=payload_tonnes,
            hauler_state=hauler_state,
            location=LocationType.ROUTE,
            location_id=-1,
            cycle_count=state.get("cycle_count", 0),
            distance=state.get("distance", 0),
        )
        
        event["node"] = self._create_node_struct(
            node_id=node.node_id,
            node_name=node.node_name,
            is_trolley=node.is_trolley,
        )
        
        return event
    
    def reset(self) -> None:
        """Reset generator state for new conversion."""
        self._event_id = 0
        self._hauler_states.clear()
        self.simulation_start_time = None
