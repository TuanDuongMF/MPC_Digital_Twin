# Event Generation Algorithm - Technical Documentation

## Overview

This document provides comprehensive technical documentation of the event generation algorithm used to convert raw AMT (Autonomous Mining Truck) telemetry data into a structured events ledger suitable for discrete event simulation (DES) animation playback.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Module Distribution](#module-distribution)
3. [Data Structures](#data-structures)
4. [Core Algorithm Modules](#core-algorithm-modules)
5. [Event Types](#event-types)
6. [Business Logic Rules](#business-logic-rules)
7. [Hauler Movement Rules](#hauler-movement-rules)
8. [State Machines](#state-machines)
9. [Configuration Parameters](#configuration-parameters)
10. [Performance Characteristics](#performance-characteristics)
11. [Data Quality & Validation](#data-quality--validation)

---

## Architecture Overview

### High-Level Pipeline

```
Raw Telemetry Data (GPS, Speed, Payload)
    ↓
Database Extraction (simulation_generator.py)
    ↓
Coordinate Conversion (mm → meters)
    ↓
Road Network Detection/Generation
    ↓
Zone Detection (Load/Dump Areas)
    ↓
GPS → Events Conversion (gps_to_events_converter.py)
    ├─ Road-Constrained Navigation (road_navigator.py)
    ├─ Node Matching (node_matcher.py)
    └─ Event Generation (event_generator.py)
    ↓
Events Ledger Output (JSON)
```

### Component Interaction Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                   simulation_generator.py                     │
│  - Database extraction                                        │
│  - Road/Zone detection                                        │
│  - File I/O orchestration                                     │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                 gps_to_events_converter.py                    │
│  - Main conversion orchestrator                               │
│  - Message processing loop                                    │
│  - Payload smoothing                                          │
└────────────────────────┬─────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│road_navigator│  │node_matcher │  │event_generator│
│  - State    │  │  - Spatial  │  │  - Event    │
│    machine  │  │    indexing │  │    creation │
│  - Road     │  │  - Nearest  │  │  - State    │
│    following│  │    node     │  │    tracking │
└─────────────┘  └─────────────┘  └─────────────┘
```

---

## Module Distribution

| Module | File | Lines | Primary Responsibility |
|--------|------|-------|------------------------|
| Event Generator | `simulation_analysis/event_generator.py` | 729 | Core event creation, hauler state tracking |
| GPS Converter | `simulation_analysis/gps_to_events_converter.py` | 736 | Main orchestrator, message processing |
| Road Navigator | `simulation_analysis/road_navigator.py` | 724 | Road-constrained navigation state machine |
| Simulation Generator | `scripts/simulation_generator.py` | 2,730 | DB extraction, road/zone detection, file I/O |
| Node Matcher | `simulation_analysis/node_matcher.py` | 365 | Spatial indexing, nearest-node lookup |
| Constants | `simulation_analysis/constants.py` | 237 | Enums, state definitions, default structures |

---

## Data Structures

### 3.1 HaulerState Enumeration

The system defines 18 distinct hauler states:

| State | Value | Description |
|-------|-------|-------------|
| `TRAVEL_LOADED` | 0 | Hauler moving with payload |
| `TRAVEL_LOADED_TROLLEY` | 1 | Moving with payload on trolley |
| `TRAVEL_UNLOADED` | 2 | Empty truck moving |
| `TRAVEL_UNLOADED_TROLLEY` | 3 | Empty truck on trolley |
| `LOADING` | 4 | At loading zone |
| `DUMPING` | 5 | At dump zone |
| `CHARGING` | 6 | Battery charging |
| `FUELING` | 7 | Refueling |
| `QUEUING` | 8 | Waiting in queue |
| `DELAY` | 9 | Operational delay |
| `STALL` | 10 | Critical stall |
| `FINISHED` | 11 | Schedule complete |
| `SPOTTING` | 12 | Maneuvering at zone |
| `TROLLEY_QUEUE` | 13 | Waiting for trolley power |
| `CRUSHER_WAIT` | 14 | Waiting at crusher |
| `CHARGING_ALT` | 15 | Alternative charging state |
| `DELAY_AND_CHARGE` | 16 | Combined delay/charge |
| `DELAY_AND_SWAP` | 17 | Combined delay/swap |
| `PASSING_BAY_DELAY` | 18 | Waiting at passing bay |

### 3.2 LocationType Enumeration

| Location | Value | Description |
|----------|-------|-------------|
| `ROUTE` | 0 | On haul road |
| `LOAD` | 1 | Loading zone |
| `DUMP` | 2 | Dump zone |
| `CHARGE` | 3 | Charging station |
| `FUEL` | 4 | Refueling area |
| `SERVICE` | 5 | Service area |

### 3.3 Hauler Data Structure

The hauler struct contains **183 fields** organized into 12 categories:

#### Identity Fields (5 fields)
```python
{
    "id": int,           # Internal hauler ID
    "uid": int,          # Unique identifier
    "name": str,         # User-defined hauler name
    "model_id": int,     # Model specification ID
    "circuit_id": int    # Material movement plan circuit ID
}
```

#### Position & Location Fields (8 fields)
```python
{
    "location": int,        # Current location type (0-5)
    "location_id": int,     # ID of current location zone (-1 for routes)
    "destination": int,     # Destination zone type
    "destination_id": int,  # Destination zone ID
    "next_stop": int,       # Next stop zone type
    "next_stop_id": int,    # Next stop zone ID
    "origin": int,          # Previous zone type
    "origin_id": int        # Previous zone ID
}
```

#### Motion Fields (7 fields)
```python
{
    "speed": float,              # Instantaneous speed (kph)
    "segmentspeed": float,       # Average segment speed (kph)
    "seglength": float,          # Current segment length (m)
    "physicalgrade": float,      # Physical road grade (%)
    "totalgrade": float,         # Effective grade (%)
    "speed_limit": float,        # Current speed limit (kph)
    "speed_limit_source": int    # Speed limit source code
}
```

#### Distance & Time Fields (9 fields)
```python
{
    "distance": float,                  # Total distance traveled (m)
    "route_distance": float,            # Distance since leaving last zone (m)
    "cycle_distance": float,            # Distance since cycle start (m)
    "route_time": float,                # Time since leaving last zone (min)
    "cycle_time": float,                # Time since cycle start (min)
    "hauler_delta_time": float,         # Time since previous event (min)
    "travel_time_node_to_node": float,  # Travel time between nodes (min)
    "wait_time_node_to_node": float,    # Wait time at node (min)
    "smu": float                        # Service meter units (min)
}
```

#### Cycle Counters (3 fields)
```python
{
    "cycle_count": int,           # Increments at each load start
    "route_count": int,           # Increments at each zone departure
    "current_material_plan": int  # Material movement plan line ID
}
```

### 3.4 Event Structure

Base event structure with 10 fields:

```python
{
    "eid": int,           # Sequential event ID
    "time": float,        # Minutes from simulation start
    "etype": str,         # Event type name
    "log_level": str,     # "record" or "debug"
    "hauler": dict,       # Hauler state struct (183 fields)
    "node": dict,         # {"id", "name", "isTrolley"}
    "trolley": None,      # Trolley context (if applicable)
    "charger": None,      # Charger context
    "loader": dict,       # Loader context (for load events)
    "delay": None,
    "ess": None,          # Energy storage system
    "zone": None,
    "battery_hauler": None,
    "crusher": None
}
```

### 3.5 RoadState (Navigation State)

```python
@dataclass
class RoadState:
    current_road_id: Optional[int]     # Currently active road
    current_node_index: int = -1       # Position in road's node list
    direction: int = 1                 # 1=forward, -1=backward
    visited_nodes: List[int] = []      # Sequential history
    road_history: List[int] = []       # Roads traversed
```

### 3.6 MatchedNode (GPS-to-Node Mapping)

```python
@dataclass
class MatchedNode:
    node_id: int
    node_name: str
    distance: float                    # Meters from GPS point
    coords: Tuple[float, float, float] # (x, y, z) in meters
    road_id: Optional[int]
    is_trolley: bool
```

### 3.7 NavigationResult

```python
@dataclass
class NavigationResult:
    node_id: int                       # Target node ID
    node_name: str
    coords: Tuple[float, float, float] # (x, y, z) in meters
    road_id: int                       # Road containing node
    distance_from_gps: float           # Meters from actual GPS
    road_switched: bool                # True if road changed
    intermediate_nodes: List[int]      # Nodes to visit first
```

---

## Core Algorithm Modules

### 4.1 EventGenerator

**File**: `simulation_analysis/event_generator.py`

**Responsibilities**:
- Sequential event ID generation
- Hauler state tracking per machine
- Event creation with proper data structures
- Payload percentage conversion to tonnes

#### State Tracking Per Machine

```python
_hauler_states[machine_id] = {
    "last_node_id": int,
    "last_event_time": datetime,
    "distance": float,           # Cumulative distance
    "cycle_distance": float,     # Distance in current cycle
    "route_distance": float,     # Distance on current route
    "cycle_count": int,
    "route_count": int,
    "last_payload": float        # For transition detection
}
```

#### Key Methods

| Method | Purpose |
|--------|---------|
| `generate_hauler_init_event()` | Creates initial HaulerInit event |
| `generate_node_arrive_event()` | HaulerNodeArrive - calculates segment metrics |
| `generate_node_leave_event()` | HaulerNodeLeave - records departure |
| `generate_load_start_event()` | Marks loading zone entry |
| `generate_load_end_event()` | Marks loading completion |
| `generate_dump_start_event()` | Marks dumping zone entry |
| `generate_dump_end_event()` | Marks dumping completion |
| `generate_idle_start_event()` | Tracks non-motion periods |
| `generate_idle_end_event()` | Ends idle tracking |
| `generate_loader_cycle_events_for_one_bucket()` | Creates 8 loader events per bucket |

#### State Inference Logic

```python
def _infer_hauler_state(speed, payload, in_zone, segment_type):
    # 1. Check segment_type mapping (primary)
    if segment_type == "SPOTTING_AT_SOURCE":
        return SPOTTING

    # 2. Speed-based inference (secondary)
    if speed <= 1:  # km/h
        if in_zone and payload < THRESHOLD:
            return LOADING
        if in_zone and payload >= THRESHOLD:
            return DUMPING
        return QUEUING

    # 3. Motion state (tertiary)
    if payload >= THRESHOLD:
        return TRAVEL_LOADED
    return TRAVEL_UNLOADED
```

#### Payload Handling

- **Input**: payloadPercent (0-100%)
- **Conversion**: `tonnes = (payloadPercent / 100.0) * DEFAULT_PAYLOAD_CAPACITY`
- **Threshold**: 50% (`PAYLOAD_THRESHOLD = 50`)
- **Default capacity**: 227,000 kg (Cat 797F hauler)

### 4.2 NodeMatcher

**File**: `simulation_analysis/node_matcher.py`

**Purpose**: Spatial indexing for GPS-to-node matching

#### Algorithm: Spatial Grid with Adjacency

```python
class NodeMatcher:
    def __init__(self, nodes, roads, grid_size=50.0):
        self._spatial_grid: Dict[Tuple[int, int], List[int]] = {}
        self._node_to_road: Dict[int, int] = {}
        self._node_adjacency: Dict[int, List[int]] = {}
```

#### Spatial Index Building

```python
# Grid cell key calculation
cell_key = (int(x // grid_size), int(y // grid_size))

# Each cell maps to list of node IDs in that cell
# Default 50m grid size
```

#### Nearest Node Lookup Algorithm

```python
def find_nearest_node(x, y, z, max_distance=100.0):
    # 1. Check 3×3 neighborhood of grid cells (9 cells total)
    # 2. Calculate 2D distance (ignoring z) to all candidates
    # 3. Return nearest within max_distance threshold
    # 4. Fallback to full search if no candidates nearby
```

#### 3×3 Grid Neighborhood Search

```
┌───┬───┬───┐
│ 0 │ 1 │ 2 │
├───┼───┼───┤
│ 3 │ X │ 4 │  X = target cell
├───┼───┼───┤
│ 5 │ 6 │ 7 │
└───┴───┴───┘
```

### 4.3 RoadNavigator

**File**: `simulation_analysis/road_navigator.py`

**Purpose**: Ensure haulers follow roads sequentially without jumping

#### Business Rules

1. Haulers must traverse nodes in road order (no skipping)
2. Within a road: linear progression through node sequence
3. Road switching only at endpoints (first or last node)
4. Validate switches maintain sequential continuity

#### Navigation States

| State | Condition | Handler |
|-------|-----------|---------|
| INITIAL | `road_id = None` | `_handle_initial_navigation()` |
| SAME_ROAD | Target on current road | `_handle_same_road_navigation()` |
| ROAD_SWITCH | Target not on current road | `_handle_road_switch_navigation()` |
| FALLBACK | No valid switch found | `_handle_road_switch_fallback()` |

#### Key Algorithms

**Sequence Validation**:
```python
def _sequence_in_road(sequence, road_nodes, forward=True):
    # Checks if node sequence appears consecutively in road
    # Handles both forward and backward traversal
    # Returns False if gap detected
```

**Switch Validity Check**:
```python
def _verify_switch_validity():
    # Validates that recent visited nodes match road sequence
    # Checks last 3 nodes for monotonic index progression
    # Ensures direction consistency
```

**Multi-Road Path Finding**:
```python
def _find_multi_road_path():
    # BFS shortest path through road network
    # Limit 1000 iterations (prevents infinite loops)
    # Returns first path found to target
```

### 4.4 GPSToEventsConverter

**File**: `simulation_analysis/gps_to_events_converter.py`

**Purpose**: Main orchestrator - converts telemetry messages to events

#### Input Parameters

```python
def convert_messages(
    messages: List[Any],              # Telemetry points
    machine_id: int,
    machine_name: str,
    min_node_distance: float = 15.0,  # Minimum spacing between node events
    max_search_distance: float = 50.0, # Max GPS-to-node distance
    include_idle_events: bool = True,
    idle_threshold_seconds: float = 30.0,
    coordinates_in_meters: bool = False,  # If False, convert mm→m
    use_road_constrained_navigation: bool = True
)
```

#### Main Processing Loop

```python
1. Initialize state variables:
   - last_node_id = None
   - last_event_time = None
   - stable_payload = None (smoothed 0 or 100)
   - is_idle = False
   - last_empty_time = None

2. For each message:
   a) Extract: x, y, z, speed, payload, time

   b) PAYLOAD SMOOTHING:
      - Only update when crossing 50% threshold
      - Prevents noise-induced fluctuations

   c) NODE MATCHING:
      if use_road_constrained_navigation:
          nav_result = road_navigator.navigate_to_gps(x, y, z)
          intermediate_nodes = nav_result.intermediate_nodes
          matched_node = MatchedNode(nav_result)
      else:
          matched_node = node_matcher.find_nearest_node(x, y, z)

   d) INITIAL EVENT:
      if first message:
          event = event_generator.generate_hauler_init_event(...)
          events.append(event)
          continue

   e) IDLE DETECTION:
      if speed <= 1 and last_speed > 1:
          is_idle = True
          idle_start_time = time
      elif speed > 1 and is_idle:
          idle_duration = time - idle_start_time
          if idle_duration >= idle_threshold:
              Generate HaulerIdleStart/End pair

   f) INTERMEDIATE NODE EVENTS:
      for inter_node_id in intermediate_nodes:
          Generate HaulerNodeLeave (previous)
          Generate HaulerNodeArrive (intermediate)

   g) NODE TRANSITION:
      if matched_node.id != last_node_id:
          seg_length = calculate segment distance
          if seg_length < min_node_distance:
              Skip (too close)
          Generate HaulerNodeLeave (previous)
          Generate HaulerNodeArrive (current)

   h) LOAD DETECTION (Payload 0→100 transition):
      if prev_payload < 50 and stable_payload >= 50:
          load_duration = load_end_time - load_start_time
          Generate HaulerLoadStart
          For each bucket in loader_cycles_per_load:
              Generate 8 loader cycle events
          Generate HaulerLoadEnd

   i) Update tracking:
      last_node_id = matched_node.id
      last_event_time = time
      last_speed = speed
```

#### Payload Smoothing Logic

```python
# Prevents oscillation when payload hovers around threshold
stable_payload = 0 if raw_payload < 50 else 100

# Only change state when crossing threshold
if (is_currently_loaded) != (is_raw_loaded):
    stable_payload = 0 if raw_payload < 50 else 100
else:
    # Keep previous state (stick to 0 or 100)
    stable_payload = previous_stable_payload
```

#### Load Event Generation

```python
# When payload transitions from 0→100:
load_start_time = last_empty_time or last_event_time
load_end_time = current_event_time
load_duration_sec = (load_end_time - load_start_time).total_seconds()

loader_cycles_per_load = 2  # 2 buckets per load
bucket_duration_sec = load_duration_sec / 2

For bucket 0..1:
    t_bucket_start = load_start_time + bucket_duration_sec * bucket
    t_bucket_end = load_start_time + bucket_duration_sec * (bucket + 1)

    Generate 8 events spread evenly:
        [DIG_START, DIG_END, SWING_START, SWING_END,
         LOAD_START, LOAD_END, RETURN_START, RETURN_END]
```

### 4.5 SimulationGenerator

**File**: `scripts/simulation_generator.py`

**Purpose**: Database extraction, preprocessing, and file generation

#### Main Pipeline Functions

| Function | Purpose |
|----------|---------|
| `fetch_telemetry_data()` | Queries database in batches |
| `create_roads_from_trajectories()` | Douglas-Peucker path simplification |
| `detect_road_network()` | Grid-based density detection |
| `detect_zones()` | Load/dump zone detection |
| `convert_reader_zones_to_model()` | Converts Reader.py zones to model format |

#### Douglas-Peucker Algorithm

```python
def douglas_peucker(points, epsilon):
    # Recursive algorithm:
    # 1. Find point furthest from line (start → end)
    # 2. If distance > epsilon:
    #    - Recursively simplify left segment
    #    - Recursively simplify right segment
    # 3. Else: return just [start, end]
```

---

## Event Types

### 5.1 Hauler Events (11 types)

| Event Type | Description | Trigger |
|------------|-------------|---------|
| `HaulerInit` | Hauler initialization | First telemetry point |
| `HaulerNodeArrive` | Arrives at node | GPS enters node boundary |
| `HaulerNodeLeave` | Departs from node | GPS leaves node |
| `HaulerEnterZone` | Enters zone boundary | Geofence detection |
| `HaulerEnterRoute` | Enters route from zone | Zone exit |
| `HaulerLoadStart` | Begins loading | Payload 0→100 transition |
| `HaulerLoadEnd` | Finishes loading | Load complete |
| `HaulerDumpStart` | Begins dumping | At dump zone |
| `HaulerDumpEnd` | Finishes dumping | Payload 100→0 |
| `HaulerIdleStart` | Begins idling | Speed 0 for > threshold |
| `HaulerIdleEnd` | Ends idling | Speed > 0 |

### 5.2 Loader Cycle Events (8 types per bucket)

| Event Type | Sequence | Description |
|------------|----------|-------------|
| `LoaderCycleDigStart` | 1 | Dig sub-event start |
| `LoaderCycleDigEnd` | 2 | Dig sub-event end |
| `LoaderCycleSwingStart` | 3 | Swing sub-event start |
| `LoaderCycleSwingEnd` | 4 | Swing sub-event end |
| `LoaderCycleLoadStart` | 5 | Load sub-event start |
| `LoaderCycleLoadEnd` | 6 | Load sub-event end |
| `LoaderCycleReturnStart` | 7 | Return sub-event start |
| `LoaderCycleReturnEnd` | 8 | Return sub-event end |

### 5.3 Typical Cycle Event Sequence

```
1. HaulerInit (t=0)
   └─ Initial position and state

2. HaulerNodeArrive (recurring)
   └─ Node-to-node progression

3. HaulerNodeLeave (recurring)
   └─ Departure from node

4. HaulerLoadStart
   └─ At load zone, payload 0→100

5. LoaderCycleDig[Start|End] × 2 buckets
   └─ 8 events per bucket

6. HaulerLoadEnd
   └─ Payload = 100%

7. HaulerIdleStart/End (optional)
   └─ Queue delays if duration ≥ 30s

8. HaulerDumpStart/End
   └─ At dump zone, payload 100→0

9. Return to node travel
   └─ Complete cycle
```

---

## Business Logic Rules

### 6.1 Payload Threshold (50%)

**Purpose**: Distinguishes loaded vs. empty state

| Payload | Classification |
|---------|----------------|
| < 50% | TRAVEL_UNLOADED, QUEUING, LOADING |
| ≥ 50% | TRAVEL_LOADED, DUMPING |

**Implementation**: Tolerates sensor noise via smoothing algorithm.

### 6.2 Cycle Counting Logic

```python
# Increment cycle_count when transitioning empty → loaded
if last_payload < 50 and current_payload >= 50:
    cycle_count += 1

# Reset cycle_distance after dump completes
if last_payload >= 50 and current_payload < 50:
    cycle_distance = 0
```

### 6.3 State Inference Hierarchy

| Priority | Method | Example |
|----------|--------|---------|
| 1 (Primary) | Segment type mapping | SPOTTING_AT_SOURCE → SPOTTING |
| 2 (Secondary) | Speed-based | ≤1 km/h: stopped, >1 km/h: moving |
| 3 (Tertiary) | Location-based | In load zone → LOADING |

### 6.4 Idle Detection Algorithm

```
Idle Start Condition:
  speed[t-1] > 1 AND speed[t] ≤ 1

Idle End Condition:
  speed[t-1] ≤ 1 AND speed[t] > 1

Generate Idle Events If:
  idle_duration ≥ idle_threshold (default 30 seconds)
```

### 6.5 Minimum Node Distance

**Parameter**: `min_node_distance = 15m`

**Purpose**: Prevents redundant node events from noisy GPS

```python
if seg_length < min_node_distance:
    skip this node
    update last_node_id
    continue  # Don't generate arrival/leave events
```

### 6.6 Road-Constrained Navigation

**Why Required**:
- Prevents haulers from "jumping" across terrain
- Enforces realistic mining site topology
- Generates intermediate node events for road traversal

**Fallback Chain**:
1. Try same-road navigation
2. Try endpoint switch (at road boundary)
3. Try multi-road BFS path
4. Give up (return None)

### 6.7 Load Duration Calculation

```python
# Key Variables:
last_empty_time      # Last time payload was below 50%
load_start_time      # Transitions when payload hits 100%
load_duration = load_end_time - load_start_time

# Bucket Distribution:
loader_cycles_per_load = 2  # Site-specific (typically 1-3)
bucket_duration = load_duration / loader_cycles_per_load

# Each bucket generates 8 events spread evenly
events_per_bucket = 8
event_interval = bucket_duration / 8
```

### 6.8 Coordinate Conversions

| Format | Unit | Usage |
|--------|------|-------|
| Database | millimeters (mm) | Raw storage |
| Internal | meters (m) | All calculations |

```python
# Conversion formula:
x_meters = x_mm / 1000.0
```

---

## Hauler Movement Rules

This section defines the strict movement constraints that haulers must follow when traversing the road network. These rules ensure accurate simulation of real-world mining truck behavior.

### 7.1 Core Movement Constraints

| Constraint | Description |
|------------|-------------|
| Sequential Node Traversal | The hauler **must move sequentially following the node order of a single road** |
| No Node Skipping | Within the same road, the hauler **must traverse every node one by one** - no nodes may be skipped |
| Road Transition Points | Road transitions are **only allowed at the last node of the current road**, then continue from the first node of the next road |

#### Movement Flow Diagram

```
Road A: [N1] → [N2] → [N3] → [N4] (last node)
                                    ↓
                            Road Transition Allowed
                                    ↓
Road B: [N5] (first node) → [N6] → [N7] → [N8]
```

#### Invalid Movement Examples

```
❌ INVALID: Skip nodes within same road
   Road A: [N1] → [N3] (skipped N2)

❌ INVALID: Road switch before reaching last node
   Road A: [N1] → [N2] → Road B: [N5] (N3, N4 not traversed)

✓ VALID: Complete traversal then switch
   Road A: [N1] → [N2] → [N3] → [N4] → Road B: [N5] → [N6]
```

### 7.2 Road Switching Conditions

#### Condition 1: GPS-Inferred Node Mismatch

A road switch is **mandatory** when:
1. The hauler is currently traveling on a road
2. Has **not yet reached the last node** of the current road
3. But the **next node inferred from GPS data does not belong to the current road**

```
Scenario:
  Current Road: A (nodes: [N1, N2, N3, N4])
  Hauler Position: At N2
  GPS-Inferred Next Node: N7 (belongs to Road B)

  Action: Road switch is MANDATORY
```

#### Condition 2: Road Selection Algorithm

When switching roads, the system must find **a road or sequence of roads** such that:

| Requirement | Description |
|-------------|-------------|
| Node Existence | All nodes traversed by the hauler must exist on those roads |
| Sequential Order | The node order must match exactly and sequentially |
| Multi-Road Switch | May require switching more than one road consecutively |
| Constraint Satisfaction | Continue switching until all Core Movement Constraints are satisfied |

#### Road Selection Flow

```
1. Identify GPS-inferred target node
2. Find all roads containing the target node
3. For each candidate road:
   a. Check if recent traversed nodes exist on this road
   b. Verify node order matches sequentially
   c. Validate direction consistency
4. If no single road satisfies:
   a. Search for multi-road path (BFS)
   b. Ensure path maintains sequential ordering
5. Select valid road(s) and update state
```

### 7.3 State Tracking Requirements

The system **must maintain a complete history** of road traversal:

#### Required State Information

| State Field | Purpose |
|-------------|---------|
| `road_history` | List of all `road_id`s the hauler has traveled on |
| `current_road_id` | The road currently being traversed |
| `visited_nodes` | Complete sequence of nodes traversed |
| `current_node_index` | Position within current road's node list |
| `direction` | Travel direction (forward=1, backward=-1) |

#### State Management Rules

```python
# State tracking rules:
1. Road history must NOT be overwritten
2. Only append new roads or update status (active/finished)
3. Node history must be preserved for validation
4. Direction changes must be explicitly tracked
```

#### State Update Operations

| Operation | Action | Example |
|-----------|--------|---------|
| Append Road | Add new road to history | `road_history.append(new_road_id)` |
| Update Status | Mark road as finished | `road_status[road_id] = "finished"` |
| Record Node | Add node to visited list | `visited_nodes.append(node_id)` |
| Update Position | Set current index | `current_node_index = target_index` |

### 7.4 Priority and Trade-offs

| Priority | Level | Description |
|----------|-------|-------------|
| **Accuracy** | **HIGHEST** | Correct node sequencing and road following is paramount |
| Completeness | High | All nodes must be visited, no skipping allowed |
| Validation | High | Continuous verification of movement constraints |
| Performance | Secondary | Speed optimizations must NOT compromise accuracy |

#### Non-Negotiable Rules

```
1. ❌ NEVER skip nodes for performance optimization
2. ❌ NEVER relax ordering rules to improve speed
3. ❌ NEVER overwrite road history
4. ✓ ALWAYS validate node sequence before accepting
5. ✓ ALWAYS generate intermediate node events
6. ✓ ALWAYS maintain complete traversal history
```

### 7.5 Implementation in RoadNavigator

The `road_navigator.py` module implements these rules through the following methods:

| Method | Responsibility |
|--------|----------------|
| `navigate_to_gps()` | Main entry point, enforces all movement rules |
| `_handle_same_road_navigation()` | Handles sequential traversal within a road |
| `_handle_road_switch_navigation()` | Validates and executes road switches |
| `_find_valid_road_for_switch()` | Finds roads satisfying all constraints |
| `_verify_switch_validity()` | Validates recent nodes match road sequence |
| `_get_intermediate_nodes()` | Extracts all nodes between current and target |
| `_find_multi_road_path()` | BFS search for multi-road paths |

#### Validation Chain

```
navigate_to_gps(target_node)
    │
    ├─→ Is target on current road?
    │       │
    │       ├─→ YES: _handle_same_road_navigation()
    │       │         └─→ Generate intermediate node events
    │       │
    │       └─→ NO: _handle_road_switch_navigation()
    │                 │
    │                 ├─→ _find_valid_road_for_switch()
    │                 │         └─→ Validate node sequence
    │                 │
    │                 └─→ _verify_switch_validity()
    │                           └─→ Check 3-node window
    │
    └─→ Return NavigationResult with intermediate_nodes
```

---

## State Machines

### 8.1 Hauler Lifecycle State Machine

```
                    ┌─────────────┐
                    │ HaulerInit  │
                    └──────┬──────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
    [Empty]            [Loaded]           [Queuing]
    TRAVEL_             TRAVEL_
    UNLOADED            LOADED
        │                  │
        │                  ├─→ DumpStart
        │                  │      │
        │                  │      ├─→ DumpEnd
        │                  │      │      ▼
        │                  │      └─→ [Empty]
        │                  │
        └─────────────┬────┘
                      │
                   LoadStart
                      │
                    [Loading] ────┬────→ LoaderCycle (8 phases)
                      │           │
                      │           └────→ LoaderCycle (bucket 2)
                      │
                   LoadEnd
                      │
                      ▼
                   [Loaded] ─────→ Travel ─────→ Dump
```

### 8.2 Road Navigation State Machine

```
┌──────────────────────────────────────────────────────────────┐
│ ROAD NAVIGATION STATES                                        │
└──────────────────────────────────────────────────────────────┘

[UNINITIALIZED]
    ↓ navigate_to_gps() [first call]
    ├─→ _handle_initial_navigation()
    │       Find road containing target
    │       Initialize state.current_road_id
    │       state.direction = 1
    │       Add to state.visited_nodes

[SAME_ROAD]
    ↓ navigate_to_gps() [target on current road]
    ├─→ _handle_same_road_navigation()
    │       Determine direction (target_index vs current_index)
    │       Extract intermediate nodes
    │       Update state.current_node_index
    │       Maintain direction (no reversal mid-road)

[ROAD_SWITCH]
    ↓ navigate_to_gps() [target NOT on current road]
    ├─→ _handle_road_switch_navigation()
    │       _find_valid_road_for_switch()
    │           • Validate road contains recent nodes
    │           • Check direction consistency (3-node window)
    │       Update state.current_road_id
    │       Update state.road_history

[FALLBACK_SWITCH]
    ↓ If no valid road found
    ├─→ _handle_road_switch_fallback()
    │       Is current node at endpoint?
    │           ├─→ Yes: Search for connecting road
    │           └─→ No: Try multi-road BFS

[MULTIROAD_PATH]
    ↓ If no direct endpoint connection
    └─→ _find_multi_road_path()
            BFS through road network (max 1000 iterations)
            Return first path found
```

---

## Configuration Parameters

### 9.1 Key Tunable Parameters

| Parameter | Default | Unit | Purpose |
|-----------|---------|------|---------|
| `grid_size` (NodeMatcher) | 50 | m | Spatial index cell size |
| `grid_size` (zone detection) | 10 | m | Zone clustering grid |
| `min_density` | 3 | points | Points per cell to mark as road |
| `simplify_epsilon` | 10 | m | Douglas-Peucker threshold |
| `min_segment_distance` | 15 | m | Minimum node spacing |
| `min_node_distance` | 15 | m | Skip nodes closer than this |
| `max_search_distance` | 50 | m | Max GPS-to-node distance |
| `idle_threshold_seconds` | 30 | s | Min duration for idle event |
| `loader_cycles_per_load` | 2 | buckets | Buckets per hauler load |
| `PAYLOAD_THRESHOLD` | 50 | % | Empty/loaded boundary |
| `DEFAULT_PAYLOAD_CAPACITY` | 227 | tonnes | Cat 797F capacity |

### 9.2 Database Batching

```python
batch_size = 100        # Handles per batch
limit = 100,000         # Records per site
sample_interval = 5     # Every 5th record
```

---

## Performance Characteristics

### 10.1 Spatial Indexing (NodeMatcher)

| Operation | Complexity |
|-----------|------------|
| Grid cell access | O(1) |
| Candidate search (3×3) | O(9) |
| Effective lookup | O(1) |
| Fallback (full search) | O(n) |

### 10.2 Event Generation

| Operation | Complexity |
|-----------|------------|
| Message processing | O(n) where n = messages |
| State lookups | O(1) |
| Node matching | O(1) grid-based |
| Road navigation | O(1) state machine |

### 10.3 Memory Usage

| Component | Estimate |
|-----------|----------|
| 100k telemetry points | ~8-10 MB |
| Node spatial index | ~1-2 MB |
| Road network | ~500 KB |

---

## Data Quality & Validation

### 11.1 Message Validation

```python
def _extract_message_data(msg):
    # Check for valid coordinates
    if pathEasting is None or pathNorthing is None:
        return None

    # Validate payload range [0, 100]
    if payload < 0 or payload > 100:
        payload = 0

    # Handle missing timestamps
    if event_time is None:
        event_time = gps_to_utc(segment_id, elapsed_time)

    # Detect reversed travel (negative speed)
    orientation = "reverse" if speed < 0 else "forward"
    speed = abs(speed)
```

### 11.2 Node Matching Validation

```python
# Distance threshold
if distance_to_node > max_search_distance:
    return None  # No match

# Spatial grid fallback
if no_candidates_in_3x3_grid:
    candidates = full_node_list  # O(n) fallback
```

### 11.3 Road Continuity Validation

```python
# Sequence validation in road
def _sequence_in_road(sequence, road_nodes, forward=True):
    # Ensure nodes appear consecutively in road_nodes
    # Check both forward and backward directions
    # Return False if gap detected

# Direction consistency check
# Verify indices are monotonic in travel direction
```

---

## Key Formulas & Calculations

### Time Conversions

```python
# GPS Epoch: January 6, 1980
GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)
LEAP_SECONDS = timedelta(seconds=18)

# GPS Time → UTC
utc = GPS_EPOCH + timedelta(seconds=(segment_id + elapsed_sec)) - LEAP_SECONDS

# Event time in minutes
time_minutes = (event_time - simulation_start_time).total_seconds() / 60.0
```

### Distance Calculations

```python
# 2D Euclidean
d = sqrt((x2-x1)**2 + (y2-y1)**2)

# 3D Euclidean
d = sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)

# Grade calculation
grade = (z2-z1) / sqrt((x2-x1)**2 + (y2-y1)**2)
```

### Payload Conversions

```python
# Percentage to tonnes
tonnes = (percent / 100) * 227000  # kg
tonnes = percent * 2.27            # simplified

# Threshold
threshold_tonnes = 50% * 227 = 113.5 tonnes
```

### Node Grid Indexing

```python
# Grid cell key
key = (int(x / 50), int(y / 50))  # 50m cells

# 3×3 Neighborhood
neighbors = [
    (gx-1, gy-1), (gx-1, gy), (gx-1, gy+1),
    (gx,   gy-1), (gx,   gy), (gx,   gy+1),
    (gx+1, gy-1), (gx+1, gy), (gx+1, gy+1)
]
```

---

## Known Constraints & Limitations

1. **Single-Machine Processing**: RoadNavigator instance per hauler (not shared)
2. **Memory Limitation**: All telemetry loaded into memory (100k point limit default)
3. **Road Topology**: Assumes acyclic or simple road graphs (BFS limit prevents infinite loops)
4. **GPS Noise**: Payload smoothing helps but may lag true transitions
5. **Time Resolution**: GPS timestamps with millisecond precision converted to seconds
6. **Zone Detection**: Requires ≥20 stopped points per grid cell (configurable)
7. **No Offline Routing**: Uses sequential GPS-to-node matching, not pre-computed routes
8. **Loader Cycles**: Assumes uniform distribution across bucket duration

---

## Output File Structure

### Events Ledger Output

```json
{
    "status": true,
    "data": {
        "version": "20250818",
        "events": [
            {
                "eid": 1,
                "time": 0.0,
                "etype": "HaulerInit",
                "hauler": {...},
                "node": {...},
                ...
            }
        ],
        "summary": {
            "total_events": 10000,
            "total_haulers": 50,
            "simulation_duration_minutes": 480,
            "event_type_counts": {...}
        }
    }
}
```

---

*Document Version: 1.0*
*Last Updated: 2026-02-05*
*Compatible with Events Version: 20250818*
