"""
GPS to Events Converter

Main converter class that orchestrates the conversion of AMT telemetry data
to simulation events format for animation playback.
"""

import json
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple

import math
from .constants import (
    HaulerState,
    LocationType,
    PAYLOAD_THRESHOLD,
    DEFAULT_PAYLOAD_CAPACITY,
)
from .node_matcher import NodeMatcher, MatchedNode
from .event_generator import EventGenerator


# GPS epoch for time conversion
GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)
LEAP_SECONDS = timedelta(seconds=18)


def gps_to_utc(segment_id: int, elapsed_time: float = 0) -> datetime:
    """
    Convert GPS time to UTC datetime.
    
    Args:
        segment_id: GPS timestamp (seconds since GPS epoch)
        elapsed_time: Additional elapsed time in seconds
    
    Returns:
        UTC datetime object
    """
    gps_time = segment_id + elapsed_time
    utc_time = GPS_EPOCH + timedelta(seconds=gps_time) - LEAP_SECONDS
    return utc_time


class GPSToEventsConverter:
    """
    Converts AMT GPS/telemetry data to simulation events.
    
    This class handles the full conversion pipeline:
    1. Load road network model
    2. Process telemetry messages
    3. Match GPS points to road nodes
    4. Generate simulation events
    5. Output events JSON
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_data: Optional[Dict] = None,
        output_version: str = "20250818",
    ):
        """
        Initialize converter.
        
        Args:
            model_path: Path to model JSON file
            model_data: Pre-loaded model dictionary (alternative to model_path)
            output_version: Events format version string
        """
        self.output_version = output_version
        self.model = None
        self.node_matcher = None
        self.event_generator = None
        
        # Load model
        if model_data:
            self.model = model_data
            self._initialize_from_model()
        elif model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> None:
        """
        Load road network model from JSON file.
        
        Args:
            model_path: Path to model JSON file
        """
        with open(model_path, "r", encoding="utf-8") as f:
            self.model = json.load(f)
        
        self._initialize_from_model()
    
    def set_model(self, model_data: Dict) -> None:
        """
        Set model from dictionary.
        
        Args:
            model_data: Model dictionary
        """
        self.model = model_data
        self._initialize_from_model()
    
    def _initialize_from_model(self) -> None:
        """Initialize NodeMatcher and EventGenerator from loaded model."""
        if not self.model:
            return
        
        nodes = self.model.get("nodes", [])
        roads = self.model.get("roads", [])
        
        self.node_matcher = NodeMatcher(nodes, roads)
        self.event_generator = EventGenerator(self.node_matcher)
    
    def _get_nearest_load_zone(self, x: float, y: float) -> Tuple[int, str]:
        """
        Return (load_zone_id, loader_name) for the load zone nearest to (x, y).
        Uses detected_location from model load_zones. Default (1, 'Loader_1_1') if none.
        """
        load_zones = self.model.get("load_zones", []) if self.model else []
        if not load_zones:
            return 1, "Loader_1_1"
        best_zone_id = 1
        best_name = "Loader_1_1"
        best_dist = float("inf")
        for z in load_zones:
            loc = z.get("detected_location") or z.get("settings", {}).get("detected_location")
            if not loc:
                continue
            zx = loc.get("x", 0)
            zy = loc.get("y", 0)
            d = math.sqrt((x - zx) ** 2 + (y - zy) ** 2)
            if d < best_dist:
                best_dist = d
                best_zone_id = z.get("id", 1)
                best_name = f"Loader_{best_zone_id}_1"
        return best_zone_id, best_name
    
    def convert_messages(
        self,
        messages: List[Any],
        machine_id: int,
        machine_name: str,
        min_node_distance: float = 15.0,
        max_search_distance: float = 50.0,
        include_idle_events: bool = True,
        idle_threshold_seconds: float = 30.0,
        coordinates_in_meters: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Convert AMT messages to simulation events.

        Args:
            messages: List of AMTCycleProdInfoMessage objects or message dicts
            machine_id: Machine ID
            machine_name: Machine name
            min_node_distance: Minimum distance between node events (meters)
            max_search_distance: Maximum distance from GPS to node (meters)
            include_idle_events: Whether to generate idle events
            idle_threshold_seconds: Minimum duration to consider as idle
            coordinates_in_meters: If True, coordinates are already in meters.
                                  If False, coordinates are in millimeters and will be converted.

        Returns:
            List of event dictionaries
        """
        if not messages or not self.node_matcher:
            return []

        # Store coordinates_in_meters for use in _extract_message_data
        self._coordinates_in_meters = coordinates_in_meters

        events = []
        last_node_id = None
        last_event_time = None
        last_speed = None
        is_idle = False
        idle_start_time = None
        idle_node = None
        # Payload smoothing: only change when crossing 50% threshold
        # This prevents constant payload fluctuation due to sensor noise
        stable_payload = None
        PAYLOAD_THRESHOLD = 50
        # For load interval (0->100): time of last message when payload was empty
        last_empty_time = None
        # Number of bucket cycles per load (2 = two buckets fill hauler)
        loader_cycles_per_load = 2
        indv_payload_tonnes = (DEFAULT_PAYLOAD_CAPACITY / 1000.0) / loader_cycles_per_load

        for i, msg in enumerate(messages):
            # Extract message data
            msg_data = self._extract_message_data(msg)
            if msg_data is None:
                continue

            x, y, z = msg_data["x"], msg_data["y"], msg_data["z"]
            speed = msg_data["speed"]
            raw_payload = msg_data["payload"]
            event_time = msg_data["time"]
            segment_type = msg_data.get("segment_type")
            orientation = msg_data.get("orientation", "forward")

            # Smooth payload: only update when crossing threshold (empty <-> loaded)
            prev_stable_payload = stable_payload
            if stable_payload is None:
                # Initialize: use 0 for empty, 100 for loaded
                stable_payload = 0 if raw_payload < PAYLOAD_THRESHOLD else 100
            else:
                # Only change when crossing threshold in opposite direction
                is_currently_loaded = stable_payload >= PAYLOAD_THRESHOLD
                is_raw_loaded = raw_payload >= PAYLOAD_THRESHOLD
                if is_currently_loaded != is_raw_loaded:
                    stable_payload = 0 if raw_payload < PAYLOAD_THRESHOLD else 100

            payload = stable_payload
            if stable_payload < PAYLOAD_THRESHOLD:
                last_empty_time = event_time
            
            # Match to node - find nearest node in the road network
            # Roads are created from actual trajectories, so nodes are already
            # in correct sequence order
            matched_node = self.node_matcher.find_nearest_node(
                x, y, z, max_search_distance
            )
            
            if matched_node is None:
                continue
            
            # Generate init event for first message
            if i == 0:
                init_event = self.event_generator.generate_hauler_init_event(
                    machine_id=machine_id,
                    machine_name=machine_name,
                    event_time=event_time,
                    start_node=matched_node,
                    payload_percent=payload,
                    segment_type=segment_type,
                )
                events.append(init_event)
                last_node_id = matched_node.node_id
                last_event_time = event_time
                last_speed = speed
                continue
            
            # Handle idle detection
            if include_idle_events:
                if speed <= 1.0 and last_speed is not None and last_speed > 1.0:
                    # Started idling
                    is_idle = True
                    idle_start_time = event_time
                    idle_node = matched_node
                elif speed > 1.0 and is_idle:
                    # Stopped idling
                    if idle_start_time and idle_node:
                        idle_duration = (event_time - idle_start_time).total_seconds()
                        if idle_duration >= idle_threshold_seconds:
                            # Generate idle start event
                            idle_start_event = self.event_generator.generate_idle_start_event(
                                machine_id=machine_id,
                                machine_name=machine_name,
                                event_time=idle_start_time,
                                node=idle_node,
                                payload_percent=payload,
                            )
                            events.append(idle_start_event)
                            
                            # Generate idle end event
                            idle_end_event = self.event_generator.generate_idle_end_event(
                                machine_id=machine_id,
                                machine_name=machine_name,
                                event_time=event_time,
                                node=matched_node,
                                payload_percent=payload,
                            )
                            events.append(idle_end_event)
                    
                    is_idle = False
                    idle_start_time = None
                    idle_node = None
            
            # Check if we've moved to a new node
            if matched_node.node_id != last_node_id:
                # Check minimum distance from last node
                if last_node_id:
                    seg_length = self.node_matcher.calculate_segment_length(
                        last_node_id, matched_node.node_id
                    )
                    if seg_length < min_node_distance:
                        # Update last_node_id even when skipping to avoid getting stuck
                        # when multiple consecutive nodes are within min_node_distance
                        last_node_id = matched_node.node_id
                        continue
                    
                    # Generate node leave event for the previous node
                    last_node_data = self.node_matcher.get_node_by_id(last_node_id)
                    if last_node_data:
                        last_coords = last_node_data.get("coords", [0, 0, 0])
                        last_matched_node = MatchedNode(
                            node_id=last_node_id,
                            node_name=last_node_data.get("name", f"Node_{last_node_id}"),
                            distance=0.0,
                            coords=tuple(last_coords[:3]) if len(last_coords) >= 3 else (last_coords[0], last_coords[1], 0.0),
                            road_id=None,
                            is_trolley=False,
                        )
                        leave_event = self.event_generator.generate_node_leave_event(
                            machine_id=machine_id,
                            machine_name=machine_name,
                            event_time=event_time,
                            node=last_matched_node,
                            speed=speed,
                            payload_percent=payload,
                            segment_type=segment_type,
                            orientation=orientation,
                        )
                        events.append(leave_event)

                # Generate node arrive event
                arrive_event = self.event_generator.generate_node_arrive_event(
                    machine_id=machine_id,
                    machine_name=machine_name,
                    event_time=event_time,
                    node=matched_node,
                    speed=speed,
                    payload_percent=payload,
                    segment_type=segment_type,
                    orientation=orientation,
                )
                events.append(arrive_event)
                
                last_node_id = matched_node.node_id
                last_event_time = event_time
            
            # Detect payload transition 0 -> 100: emit HaulerLoadStart, loader cycle events, HaulerLoadEnd
            if (
                prev_stable_payload is not None
                and prev_stable_payload < PAYLOAD_THRESHOLD
                and stable_payload >= PAYLOAD_THRESHOLD
            ):
                load_start_time = last_empty_time if last_empty_time is not None else last_event_time
                load_end_time = event_time
                if load_end_time > load_start_time:
                    load_zone_id, loader_name = self._get_nearest_load_zone(x, y)
                    load_start_event = self.event_generator.generate_load_start_event(
                        machine_id=machine_id,
                        machine_name=machine_name,
                        event_time=load_start_time,
                        payload_percent=0,
                    )
                    events.append(load_start_event)
                    interval_sec = (load_end_time - load_start_time).total_seconds()
                    bucket_duration_sec = interval_sec / loader_cycles_per_load
                    for b in range(loader_cycles_per_load):
                        t_bucket_start = load_start_time + timedelta(
                            seconds=bucket_duration_sec * b
                        )
                        t_bucket_end = load_start_time + timedelta(
                            seconds=bucket_duration_sec * (b + 1)
                        )
                        bucket_events = self.event_generator.generate_loader_cycle_events_for_one_bucket(
                            loader_name=loader_name,
                            load_zone_id=load_zone_id,
                            hauler_id=machine_id,
                            hauler_name=machine_name,
                            t_start=t_bucket_start,
                            t_end=t_bucket_end,
                            indv_payload_tonnes=indv_payload_tonnes,
                        )
                        events.extend(bucket_events)
                    load_end_event = self.event_generator.generate_load_end_event(
                        machine_id=machine_id,
                        machine_name=machine_name,
                        event_time=load_end_time,
                        payload_percent=100,
                    )
                    events.append(load_end_event)
            
            last_event_time = event_time
            last_speed = speed
        
        return events
    
    def convert_cycles(
        self,
        cycles: List[Any],
        machine_id: int,
        machine_name: str,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Convert AMT Cycle objects to simulation events.
        
        Args:
            cycles: List of Cycle objects
            machine_id: Machine ID
            machine_name: Machine name
            **kwargs: Additional arguments passed to convert_messages
        
        Returns:
            List of event dictionaries
        """
        all_messages = []
        
        for cycle in cycles:
            if hasattr(cycle, 'messages'):
                all_messages.extend(cycle.messages)
            elif hasattr(cycle, 'segments'):
                for segment in cycle.segments:
                    if hasattr(segment, 'messages'):
                        all_messages.extend(segment.messages)
        
        # Sort messages by time
        all_messages.sort(key=lambda m: self._get_message_time(m))
        
        return self.convert_messages(all_messages, machine_id, machine_name, **kwargs)
    
    def convert_raw_telemetry(
        self,
        telemetry_data: List[Tuple],
        machine_id: int,
        machine_name: str,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Convert raw telemetry tuples to simulation events.
        
        Args:
            telemetry_data: List of telemetry tuples from database
            machine_id: Machine ID
            machine_name: Machine name
            **kwargs: Additional arguments passed to convert_messages
        
        Returns:
            List of event dictionaries
        """
        # Convert tuples to message-like dicts
        messages = []
        for row in telemetry_data:
            if len(row) < 12:
                continue
            
            msg = {
                "machine_id": row[0],
                "segment_id": row[1],
                "cycle_id": row[2],
                "interval": row[3],
                "pathEasting": row[4],
                "pathNorthing": row[5],
                "pathElevation": row[6],
                "expectedSpeed": row[7],
                "actualSpeed": row[8],
                "pathBank": row[9],
                "pathHeading": row[10],
                "leftWidth": row[11] if len(row) > 11 else 0,
                "rightWidth": row[12] if len(row) > 12 else 0,
                "payloadPercent": row[13] if len(row) > 13 else 0,
            }
            messages.append(msg)
        
        return self.convert_messages(messages, machine_id, machine_name, **kwargs)
    
    def _extract_message_data(self, msg: Any) -> Optional[Dict[str, Any]]:
        """
        Extract relevant data from a message object or dict.
        
        Args:
            msg: AMTCycleProdInfoMessage object or dict
        
        Returns:
            Extracted data dict or None if invalid
        """
        # Handle object with attributes
        if hasattr(msg, 'pathEasting'):
            x = getattr(msg, 'pathEasting', 0)
            y = getattr(msg, 'pathNorthing', 0)
            z = getattr(msg, 'pathElevation', 0)
            speed = getattr(msg, 'actualSpeed', 0)
            payload = getattr(msg, 'payloadPercent', 0)
            event_time = getattr(msg, 'actualTime', None)
            segment_id = getattr(msg, 'segmentId', 0)
            elapsed = getattr(msg, 'actualElapsedTime', 0)
            segment_type = None

            if event_time is None:
                event_time = gps_to_utc(segment_id, elapsed / 1000.0 if elapsed else 0)

            # Negative speed means reverse direction
            orientation = "reverse" if speed and speed < 0 else "forward"

            return {
                "x": x,
                "y": y,
                "z": z,
                "speed": abs(speed) if speed else 0,
                "payload": payload if payload is not None and 0 <= payload <= 100 else 0,
                "time": event_time,
                "segment_type": segment_type,
                "orientation": orientation,
            }
        
        # Handle dict
        if isinstance(msg, dict):
            # Coordinates from database or imported data
            x = msg.get("pathEasting", 0)
            y = msg.get("pathNorthing", 0)
            z = msg.get("pathElevation", 0)

            # Use consistent coordinate conversion based on coordinates_in_meters flag
            # This must match the logic used in create_roads_from_trajectories()
            coordinates_in_meters = getattr(self, '_coordinates_in_meters', False)
            if not coordinates_in_meters:
                # Coordinates are in millimeters (database format) - convert to meters
                x = x / 1000.0
                y = y / 1000.0
                z = z / 1000.0

            speed = msg.get("actualSpeed", 0) or 0
            payload = msg.get("payloadPercent", 0)
            segment_id = msg.get("segment_id", msg.get("segmentId", 0))
            # Use 'interval' (time within segment in ms) or 'actualElapsedTime'
            elapsed = msg.get("interval", msg.get("actualElapsedTime", 0)) or 0

            if payload is None or payload > 100:
                payload = 0

            # elapsed is in milliseconds, convert to seconds
            event_time = gps_to_utc(segment_id, elapsed / 1000.0)

            # Negative speed means reverse direction
            orientation = "reverse" if speed < 0 else "forward"

            return {
                "x": x,
                "y": y,
                "z": z,
                "speed": abs(speed),
                "payload": payload,
                "time": event_time,
                "segment_type": None,
                "orientation": orientation,
            }
        
        return None
    
    def _get_message_time(self, msg: Any) -> datetime:
        """Get datetime from message for sorting."""
        if hasattr(msg, 'actualTime') and msg.actualTime:
            return msg.actualTime
        if hasattr(msg, 'segmentId'):
            elapsed = getattr(msg, 'actualElapsedTime', 0) or 0
            return gps_to_utc(msg.segmentId, elapsed / 1000.0)
        if isinstance(msg, dict):
            segment_id = msg.get("segment_id", msg.get("segmentId", 0))
            elapsed = msg.get("actualElapsedTime", 0) or 0
            return gps_to_utc(segment_id, elapsed / 1000.0)
        return datetime.now(timezone.utc)
    
    def create_output(
        self,
        events: List[Dict[str, Any]],
        include_summary: bool = False,
    ) -> Dict[str, Any]:
        """
        Create final output structure.
        
        Args:
            events: List of event dictionaries
            include_summary: Whether to include summary data
        
        Returns:
            Complete output dictionary matching simulation format
        """
        output = {
            "status": True,
            "data": {
                "version": self.output_version,
                "events": events,
            }
        }
        
        if include_summary:
            output["data"]["summary"] = self._generate_summary(events)
        
        return output
    
    def _generate_summary(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics from events."""
        if not events:
            return {}
        
        # Calculate basic statistics
        hauler_ids = set()
        event_types = {}
        total_time = 0
        
        for event in events:
            if event.get("hauler"):
                hauler_ids.add(event["hauler"].get("id"))
            
            etype = event.get("etype", "Unknown")
            event_types[etype] = event_types.get(etype, 0) + 1
            
            total_time = max(total_time, event.get("time", 0))
        
        return {
            "total_events": len(events),
            "total_haulers": len(hauler_ids),
            "simulation_duration_minutes": total_time,
            "event_type_counts": event_types,
        }
    
    def save_events(
        self,
        events: List[Dict[str, Any]],
        output_path: str,
        include_summary: bool = True,
    ) -> None:
        """
        Save events to JSON file.
        
        Args:
            events: List of event dictionaries
            output_path: Output file path
            include_summary: Whether to include summary data
        """
        output = self.create_output(events, include_summary)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, default=str)
    
    def reset(self) -> None:
        """Reset converter state for new conversion."""
        if self.event_generator:
            self.event_generator.reset()
