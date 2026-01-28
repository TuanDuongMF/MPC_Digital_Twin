# Events Structure Specification

## Document Purpose

This document provides a comprehensive specification of the simulation events structure. 
It is intended to facilitate the conversion of real-world autonomous hauler GPS/telemetry data 
into the simulation events format.

---

## 1. Event Architecture Overview

### 1.1 Class Hierarchy

```
BaseEvent (abstract)
├── RecordedEvent (log_level="record") → Included in output JSON
│   ├── Hauler Events (HaulerNodeArrive, HaulerLoadEnd, etc.)
│   ├── Infrastructure Events (TrolleyInit, ChargerInit, etc.)
│   └── 100+ event classes
└── DebugEvent (log_level="debug") → Only when debugging enabled
    └── HaulerNodeRequest, HaulerNodeAcquired, HaulerNodeRelease
```

### 1.2 Event Storage Formats

#### Runtime Format (Python Object)
```python
HaulerNodeArrive(
    eid=12345,
    time=45.678,
    etype="HaulerNodeArrive",
    log_level="record",
    hauler=Hauler(...),
    node=Node(...),
    battery_hauler=Battery(...)
)
```

#### Output Format (Dictionary)
```python
{
    "eid": 12345,                    # Event ID (int)
    "time": 45.678,                  # Event time in minutes (float)
    "etype": "HaulerNodeArrive",     # Event type name (str)
    "log_level": "record",           # "record" or "debug" (str)
    
    "hauler": {                      # Hauler struct (dict with 80+ fields)
        "id": 1,
        "name": "Hauler_T01",
        "speed": 35.5,
        "payload": 185.5,
        # ... all hauler fields
    },
    
    "node": {                        # Node struct (dict)
        "id": 4523,
        "name": "R_Seg_123_N2",
        "isTrolley": False
    },
    
    "trolley": None,                 # Trolley struct or None if not applicable
    "charger": None,                 # Charger struct or None if not applicable
    "loader": None,                  # Loader struct or None if not applicable
    "delay": None,                   # Delay struct or None if not applicable
    "ess": None,                     # ESS struct or None if not applicable
    "zone": None,                    # Zone struct or None if not applicable
    "battery_hauler": None,          # Battery struct or None if not applicable
    "crusher": None                  # Crusher struct or None if not applicable
}
```

> **Note**: Only include structs that are relevant to the event type. 
> Omit or set to `None` for structs not used by the event.

---

## 2. Base Event Fields

| Field | Type | Description | Required |
|-------|------|-------------|----------|
| `eid` | int | Unique event identifier, auto-generated, sequential | Yes |
| `time` | float | Event timestamp in minutes from simulation start | Yes |
| `etype` | str | Event type name (class name) | Yes |
| `log_level` | str | "record" for normal events, "debug" for debug events | Yes |

---

## 3. Struct Specifications

### 3.1 Hauler Struct (80+ fields)

The Hauler struct contains comprehensive information about a hauler's state at event time.

#### Identity Fields
| Field | Type | Description | GPS Data Source |
|-------|------|-------------|-----------------|
| `id` | int | Internal hauler ID | Machine ID |
| `uid` | int | Unique identifier | Machine ID |
| `name` | str | User-defined hauler name | Machine name/ID |
| `model_id` | int | Model specification ID | Machine model |
| `circuit_id` | int | Material movement plan circuit ID | Route assignment |

#### Position & Location Fields
| Field | Type | Description | GPS Data Source |
|-------|------|-------------|-----------------|
| `orientation` | str | "forward" or "reverse" | GPS heading delta |
| `location` | int | Current location type (see Location Dictionary) | Geofence detection |
| `location_id` | int | ID of current location zone (-1 for routes) | Geofence ID |
| `destination` | int | Destination zone type | Dispatch system |
| `destination_id` | int | Destination zone ID | Dispatch system |
| `next_stop` | int | Next stop zone type | Dispatch system |
| `next_stop_id` | int | Next stop zone ID | Dispatch system |
| `origin` | int | Previous zone type | Previous geofence |
| `origin_id` | int | Previous zone ID | Previous geofence ID |

#### Location Dictionary
```python
location_dict = {
    "route": 0,      # On haul road
    "load": 1,       # At load zone
    "dump": 2,       # At dump zone
    "charge": 3,     # At charge zone
    "fuel": 4,       # At fuel zone
    "service": 5     # At service zone
}
```

#### Motion Fields
| Field | Type | Unit | Description | GPS Data Source |
|-------|------|------|-------------|-----------------|
| `speed` | float | kph | Instantaneous speed | GPS velocity |
| `segmentspeed` | float | kph | Average segment speed | Calculated |
| `seglength` | float | m | Current segment length | Road network |
| `physicalgrade` | float | % | Physical road grade | DTM/Road network |
| `totalgrade` | float | % | Effective grade (includes rolling resistance) | Calculated |
| `speed_limit` | float | kph | Current speed limit | Road network |
| `speed_limit_source` | int | - | Speed limit source (see dict) | Road network |

#### Speed Limit Source Dictionary
```python
speed_limit_dict = {
    "Design": 0,      # Road design speed
    "Machine": 1,     # Machine capability
    "Grade": 2,       # Grade-limited
    "TKPH": 3,        # Tire-limited
    "Trolley": 4,     # Trolley-limited
    "Energy": 5       # Energy-limited
}
```

#### Distance & Time Fields
| Field | Type | Unit | Description | GPS Data Source |
|-------|------|------|-------------|-----------------|
| `distance` | float | m | Total distance traveled | GPS odometer |
| `route_distance` | float | m | Distance since leaving last zone | Calculated |
| `cycle_distance` | float | m | Distance since cycle start (at load) | Calculated |
| `route_time` | float | min | Time since leaving last zone | Calculated |
| `cycle_time` | float | min | Time since cycle start | Calculated |
| `hauler_delta_time` | float | min | Time since previous event | Calculated |
| `travel_time_node_to_node` | float | min | Travel time between nodes | Calculated |
| `wait_time_node_to_node` | float | min | Wait time at node | Calculated |
| `smu` | float | min | Service meter units (total operating time) | Machine telemetry |
| `time_in_state` | float | min | Time in current state | Calculated |

#### Cycle & Count Fields
| Field | Type | Description | GPS Data Source |
|-------|------|-------------|-----------------|
| `cycle_count` | int | Increments at each load start | Load events |
| `route_count` | int | Increments at each zone departure | Zone exit events |
| `current_material_plan` | int | Material movement plan line ID | Dispatch system |

#### Energy - Fuel Fields
| Field | Type | Unit | Description | GPS Data Source |
|-------|------|------|-------------|-----------------|
| `fuel_level` | float | 0-1 | Current fuel level (fraction) | Fuel sensor |
| `fuel_rate` | float | L/hr | Total fuel consumption rate | Engine telemetry |
| `fuel_propel_rate` | float | L/hr | Fuel rate for propulsion | Engine telemetry |
| `fuel_idle_rate` | float | L/hr | Fuel rate for idling | Engine telemetry |
| `fuel_fill_rate` | float | L/hr | Fuel fill rate (when fueling) | Fuel system |

#### Energy - Battery/Electric Fields
| Field | Type | Unit | Description | GPS Data Source |
|-------|------|------|-------------|-----------------|
| `soc` | float | 0-1 | State of charge (internal) | BMS |
| `charge_level` | float | 0-1 | State of charge (output) | BMS |
| `energy_state` | int | - | Energy state code | BMS/System |
| `energy_next_stop` | float | % | Predicted energy at next stop | Calculated |
| `eta_next_stop` | float | min | ETA to next stop | Calculated |

#### Power - Battery Fields
| Field | Type | Unit | Description | GPS Data Source |
|-------|------|------|-------------|-----------------|
| `total_kw_battery` | float | kW | Total battery power (+ discharge, - charge) | BMS |
| `brake_charge_kw_battery` | float | kW | Power from braking | Drive system |
| `idle_disch_kw_battery` | float | kW | Idle discharge power | BMS |
| `propel_disch_kw_battery` | float | kW | Propulsion discharge power | Drive system |
| `retarding_charge_kw_battery` | float | kW | Retarding charge power | Drive system |
| `regen_waste_kw_battery` | float | kW | Wasted regen power | Drive system |
| `regen_kw_wheels` | float | kW | Regen power at wheels | Drive system |
| `propel_kw_wheels` | float | kW | Propel power at wheels | Drive system |

#### Power - Charger Fields
| Field | Type | Unit | Description | GPS Data Source |
|-------|------|------|-------------|-----------------|
| `charger_charge_kw_source` | float | kW | Real charging power at grid | Charger system |
| `charger_charge_kvar_source` | float | kVAR | Reactive charging power | Charger system |
| `charger_charge_kw_charger` | float | kW | Charging power at charger output | Charger system |
| `charger_charge_kw_battery` | float | kW | Charging power at battery | BMS |

#### Power - Trolley Fields
| Field | Type | Unit | Description | GPS Data Source |
|-------|------|------|-------------|-----------------|
| `trolley_total_kw_source` | float | kW | Total trolley power (grid) | Trolley system |
| `trolley_total_kvar_source` | float | kVAR | Total reactive trolley power | Trolley system |
| `trolley_propel_kw_source` | float | kW | Propel power from trolley (grid) | Trolley system |
| `trolley_propel_kvar_source` | float | kVAR | Propel reactive power | Trolley system |
| `trolley_charge_kw_source` | float | kW | Battery charge from trolley (grid) | Trolley system |
| `trolley_charge_kvar_source` | float | kVAR | Charge reactive power | Trolley system |
| `trolley_total_kw_trolley` | float | kW | Total power at trolley output | Trolley system |
| `trolley_propel_kw_trolley` | float | kW | Propel power at trolley output | Trolley system |
| `trolley_charge_kw_trolley` | float | kW | Charge power at trolley output | Trolley system |
| `trolley_propel_kw_hauler` | float | kW | Propel power at hauler input | Drive system |
| `trolley_charge_kw_hauler` | float | kW | Charge power at hauler input | Drive system |
| `trolley_total_kw_battery` | float | kW | Total trolley power at battery | BMS |
| `trolley_propel_kw_battery` | float | kW | Propel power at battery | BMS |
| `trolley_charge_kw_battery` | float | kW | Charge power at battery | BMS |
| `trolley_propel_fuel` | float | L/hr | Fuel saved by trolley | Calculated |
| `trolley_demand_kw_trolley` | float | kW | Additional power requested | Drive system |

#### TKPH Fields
| Field | Type | Unit | Description | GPS Data Source |
|-------|------|------|-------------|-----------------|
| `tkph_front` | float | TKPH | Front wheel TKPH | Tire monitoring |
| `tkph_rear` | float | TKPH | Rear wheel TKPH | Tire monitoring |
| `tkph_trail` | float | TKPH | Trail wheel TKPH | Tire monitoring |
| `tkph_activated` | bool | - | TKPH limit active | Tire monitoring |

#### State Fields
| Field | Type | Description | GPS Data Source |
|-------|------|-------------|-----------------|
| `hauler_state` | int | Current hauler state (see dictionary) | State machine |
| `payload` | float | Current payload in tonnes | Payload sensor |
| `battery_name` | str | Name of installed battery | System config |

#### Hauler State Dictionary
```python
hauler_state_dict = {
    0: "Travel Loaded",
    1: "Travel Loaded Trolley",
    2: "Travel Unloaded",
    3: "Travel Unloaded Trolley",
    4: "Loading",
    5: "Dumping",
    6: "Charging",
    7: "Fueling",
    8: "Queuing",           # Queue at load/dump/charge/route
    9: "Delay",             # Scheduled delay (power on/off)
    10: "Stall",            # Energy stall
    11: "Finished",         # Schedule complete
    12: "Spotting",         # Maneuvering at zone
    13: "Trolley Queue",    # Waiting for trolley power
    14: "Crusher Wait",     # Waiting at crusher
    15: "Charging",         # Alternative charging state
    16: "Delay and Charge", # Combined delay/charge
    17: "Delay and Swap",   # Combined delay/swap
    18: "Passing Bay Delay" # Waiting at passing bay
}
```

---

### 3.2 Node Struct

| Field | Type | Description | GPS Data Source |
|-------|------|-------------|-----------------|
| `id` | int | Node unique identifier | Road network |
| `name` | str | Node name (generated) | Road network |
| `isTrolley` | bool | Is this a trolley node | Road network |

---

### 3.3 Trolley Struct

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `id` | int | - | Trolley/DET ID |
| `name` | str | - | User-defined name |
| `total_kw_source` | float | kW | Total real power at grid |
| `total_kvar_source` | float | kVAR | Total reactive power |
| `total_kw_trolley` | float | kW | Total power at trolley output |
| `propel_kw_source` | float | kW | Propel power at grid |
| `propel_kvar_source` | float | kVAR | Propel reactive power |
| `propel_kw_trolley` | float | kW | Propel power at trolley |
| `propel_kw_hauler` | float | kW | Propel power at hauler |
| `charge_kw_source` | float | kW | Charge power at grid |
| `charge_kvar_source` | float | kVAR | Charge reactive power |
| `charge_kw_trolley` | float | kW | Charge power at trolley |
| `charge_kw_hauler` | float | kW | Charge power at hauler |
| `num_trucks` | int | - | Trucks on active trolley |
| `num_trucks_shutdown_det` | int | - | Trucks on inactive trolley |
| `num_mounted_trucks` | int | - | Total mounted trucks |
| `total_kw_rms_trolley` | float | kW | RMS power (rolling average) |
| `total_kw_power_limit` | float | kW | Current power limit |
| `trolley_state` | int | - | Trolley state code |

---

### 3.4 Charger Struct

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `zone_name` | str | - | Charge zone name |
| `zone_id` | int | - | Charge zone ID |
| `spot_id` | int | - | Spot ID within zone |
| `node_id` | int | - | Node ID of charger |
| `charger_delta_time` | float | min | Time since last charger event |
| `zone_charge_kw_source` | float | kW | Total zone real power |
| `zone_charge_kvar_source` | float | kVAR | Total zone reactive power |
| `zone_num_trucks` | int | - | Trucks in zone |
| `charger_state` | int | - | Charger state code |
| `charge_kw_source` | float | kW | Individual charger power |
| `charge_kvar_source` | float | kVAR | Individual reactive power |

#### Charger State Dictionary
```python
charger_state_dict = {
    0: "Idle",
    1: "Delay",
    2: "Connect",
    3: "Charge",
    4: "Disconnect"
}
```

---

### 3.5 Battery Struct

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `name` | str | - | Battery name |
| `soc` | float | 0-1 | State of charge |
| `battery_loc` | str | - | Current location |
| `on_truck` | bool | - | Is battery on hauler |
| `battery_state` | int | - | Battery state code |
| `is_swappable` | bool | - | Is battery swappable |
| `charger_charge_kw_source` | float | kW | Charging power (grid) |
| `charger_charge_kvar_source` | float | kVAR | Reactive charging power |
| `charger_charge_kw_charger` | float | kW | Charging power (charger) |
| `charger_charge_kw_battery` | float | kW | Charging power (battery) |

#### Battery State Dictionary
```python
battery_state_dict = {
    0: "On Truck - Idle",
    1: "On Truck - Charging",
    2: "On Truck - Discharging",
    3: "Off Truck - Idle",
    4: "Off Truck - Charging",
    5: "Off Truck - Cooling",
    6: "Off Truck - Storage"
}
```

---

### 3.6 ESS (Energy Storage System) Struct

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `id` | int | - | ESS/Power node ID |
| `ess_kw_battery` | float | kW | ESS battery power |
| `ess_kw_source` | float | kW | ESS real power at grid |
| `ess_kw_reserved_source` | float | kW | Reserved real power |
| `ess_kvar_source` | float | kVAR | ESS reactive power |
| `ess_kvar_reserved_source` | float | kVAR | Reserved reactive power |
| `soc` | float | 0-1 | ESS state of charge |
| `supply_kw_source` | float | kW | Line supply real power |
| `supply_kw_reserved_source` | float | kW | Reserved line power |
| `supply_kvar_source` | float | kVAR | Line reactive power |
| `supply_kvar_reserved_source` | float | kVAR | Reserved reactive power |

---

### 3.7 Delay Struct

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `name` | str | - | Delay name |
| `action` | str | - | Delay action ("pullover", "service zone") |
| `start_time` | float | min | Delay start time |
| `duration` | float | min | Delay duration |

---

### 3.8 Loader Struct

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `name` | str | - | Loader name |
| `hauler_name` | str | - | Hauler being loaded |
| `time_duration` | float | min | Event duration |
| `power` | float | kW | Loader power consumption |
| `indv_payload` | float | tonnes | Individual bucket payload |

---

### 3.9 Crusher Struct

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `name` | str | - | Crusher name |
| `hopper_fill` | float | 0-1 | Hopper fill ratio |
| `processed_material` | float | tonnes | Total processed material |
| `power` | float | kW | Crusher power consumption |

---

### 3.10 Zone Struct

| Field | Type | Description |
|-------|------|-------------|
| `name` | str | Zone name |
| `zone_type` | str | Zone type ("load", "dump", "charge", etc.) |

---

### 3.11 PassingBay Struct

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `action` | str | - | "passing" or "waiting" |
| `duration` | float | min | Expected wait duration |

---

## 4. Event Types Catalog

### 4.1 Hauler Movement Events

| Event Type | Description | Required Structs |
|------------|-------------|------------------|
| `HaulerInit` | Hauler initialization at t=0 | hauler, node, battery_hauler |
| `HaulerNodeArrive` | Hauler arrives at node | hauler, node, battery_hauler |
| `HaulerNodeLeave` | Hauler departs from node | hauler, node, trolley, battery_hauler |
| `HaulerEnterZone` | Hauler enters zone boundary | hauler |
| `HaulerEnterRoute` | Hauler enters route from zone | (special fields) |
| `HaulerEnergyStall` | Hauler runs out of energy | hauler, node, battery_hauler |
| `HaulerScheduleComplete` | Hauler completes all tasks | hauler, node, battery_hauler |

### 4.2 Material Handling Events

| Event Type | Description | Required Structs |
|------------|-------------|------------------|
| `HaulerLoadStart` | Hauler begins loading | hauler, battery_hauler |
| `HaulerLoadEnd` | Hauler finishes loading | hauler, battery_hauler |
| `HaulerDumpStart` | Hauler begins dumping | hauler, battery_hauler |
| `HaulerDumpEnd` | Hauler finishes dumping | hauler, battery_hauler |

### 4.3 Idle/Wait Events

| Event Type | Description | Required Structs |
|------------|-------------|------------------|
| `HaulerIdleStart` | Hauler begins idling (traffic) | hauler, node, battery_hauler |
| `HaulerIdleEnd` | Hauler finishes idling | hauler, node, battery_hauler |
| `HaulerLoadIdleStart` | Waiting for loader | hauler, battery_hauler |
| `HaulerLoadIdleEnd` | Loader ready | hauler, battery_hauler |
| `HaulerCrusherWaitStart` | Waiting at crusher | hauler, crusher, battery_hauler |
| `HaulerCrusherWaitEnd` | Crusher ready | hauler, crusher, battery_hauler |

### 4.4 Charging Events

| Event Type | Description | Required Structs |
|------------|-------------|------------------|
| `ChargerInit` | Charger initialization | charger, hauler, battery_hauler |
| `Chargeconnectstart` | Begin charger connection | charger, hauler, ess, battery_hauler |
| `Chargeconnectfinish` | Connection complete | charger, hauler, ess, battery_hauler |
| `HaulerActiveChargingStart` | Charging interval start | charger, hauler, ess, battery_hauler |
| `HaulerActiveChargingEnd` | Charging interval end | charger, hauler, ess, battery_hauler |
| `Chargedelaystart` | Charge delay (power limit) | charger, hauler, ess, battery_hauler |
| `Chargedelayend` | Charge delay end | charger, hauler, ess, battery_hauler |
| `Chargedisconnectstart` | Begin disconnection | charger, hauler, ess, battery_hauler |
| `Chargedisconnectfinish` | Disconnection complete | charger, hauler, battery_hauler |

### 4.5 Fueling Events

| Event Type | Description | Required Structs |
|------------|-------------|------------------|
| `FuelPumpInit` | Fuel pump initialization | fuelpump |
| `HaulerFuelingConnectStart` | Begin fuel connection | hauler, fuelpump |
| `HaulerFuelingConnectEnd` | Connection complete | hauler, fuelpump |
| `HaulerFuelingStart` | Fueling begins | hauler, fuelpump |
| `HaulerFuelingEnd` | Fueling complete | hauler, fuelpump |
| `HaulerFuelingDisconnectStart` | Begin disconnection | hauler, fuelpump |
| `HaulerFuelingDisconnectEnd` | Disconnection complete | hauler, fuelpump |

### 4.6 Trolley Events

| Event Type | Description | Required Structs |
|------------|-------------|------------------|
| `TrolleyInit` | Trolley initialization | trolley |
| `HaulerMountOnTrolley` | Mechanical mount | trolley, hauler, node, battery_hauler |
| `HaulerConnectToTrolley` | Electrical connection | trolley, hauler, node, ess, battery_hauler |
| `HaulerTravelOnTrolleyStart` | Begin trolley travel | trolley, hauler, node, ess, battery_hauler |
| `HaulerTravelOnTrolleyEnd` | End trolley travel | trolley, hauler, node, ess, battery_hauler |
| `HaulerDisconnectFromTrolley` | Electrical disconnect | trolley, hauler, node, ess, battery_hauler |
| `HaulerDismountFromTrolley` | Mechanical dismount | trolley, hauler, node, battery_hauler |
| `HaulerTrolleyIdleStart` | Wait for trolley power | hauler, node, trolley, ess, battery_hauler |
| `HaulerTrolleyIdleEnd` | Trolley power available | hauler, node, trolley, ess, battery_hauler |
| `TrolleyShutdownStart` | Trolley delay starts | zone, delay, trolley |
| `TrolleyShutdownEnd` | Trolley delay ends | zone, delay, trolley |

### 4.7 Delay Events

| Event Type | Description | Required Structs |
|------------|-------------|------------------|
| `HaulerDelayTrigger` | Delay notification | hauler, delay, node, battery_hauler |
| `HaulerDelayStart` | Delay begins | hauler, delay, node, battery_hauler |
| `HaulerDelayEnd` | Delay ends | hauler, delay, node, battery_hauler |
| `HaulerDelayExtend` | Delay extended | hauler, delay, node, battery_hauler |
| `HaulerDerateStart` | Speed derate begins | hauler, delay, node, battery_hauler |
| `HaulerDerateEnd` | Speed derate ends | hauler, delay, node, battery_hauler |

### 4.8 Battery Swap Events

| Event Type | Description | Required Structs |
|------------|-------------|------------------|
| `BatteryHaulerInit` | Battery initialization | battery_hauler, capacity_kwh |
| `BatteryHaulerInit_swap` | Swappable battery init | battery_hauler, capacity_kwh |
| `BatteryHaulerNeutralSwapRemovalStart` | Battery removal start | hauler, charger, battery_hauler |
| `BatteryHaulerNeutralSwapRemovalEnd` | Battery removal end | hauler, charger, battery_hauler |
| `HaulerBatteryQueueStart` | Queue for battery | hauler, charger |
| `HaulerBatteryQueueEnd` | Battery available | hauler, charger |
| `BatteryHaulerNeutralSwapInstallStart` | Installation start | hauler, charger, battery_hauler |
| `BatteryHaulerNeutralSwapInstallEnd` | Installation end | hauler, charger, battery_hauler |
| `BatteryHaulerActiveChargingStart` | External battery charge start | charger, battery_hauler, ess |
| `BatteryHaulerActiveChargingEnd` | External battery charge end | charger, battery_hauler, ess |

### 4.9 Loader Events

| Event Type | Description | Required Structs |
|------------|-------------|------------------|
| `LoaderIdleStart` | Loader idle (no hauler) | loader, ess |
| `LoaderIdleEnd` | Hauler arrived | loader, ess |
| `LoaderCycleDigStart` | Dig sub-event start | loader, ess |
| `LoaderCycleDigEnd` | Dig sub-event end | hauler, loader, ess |
| `LoaderCycleSwingStart` | Swing sub-event start | loader, ess |
| `LoaderCycleSwingEnd` | Swing sub-event end | hauler, loader, ess |
| `LoaderCycleLoadStart` | Load sub-event start | loader, ess |
| `LoaderCycleLoadEnd` | Load sub-event end | hauler, loader, ess |
| `LoaderCycleReturnStart` | Return sub-event start | loader, ess |
| `LoaderCycleReturnEnd` | Return sub-event end | hauler, loader, ess |
| `LoaderShutdownStart` | Loader delay start | loader, delay |
| `LoaderShutdownEnd` | Loader delay end | loader, delay |

### 4.10 Crusher Events

| Event Type | Description | Required Structs |
|------------|-------------|------------------|
| `CrusherIdleStart` | Crusher idle | crusher, ess |
| `CrusherIdleEnd` | Crusher active | crusher, ess |
| `CrusherDumpStart` | Receiving dump | crusher, ess |
| `CrusherDumpEnd` | Dump complete | crusher, ess |
| `CrusherBusy` | Hopper full | crusher, ess |
| `CrusherFree` | Hopper has space | crusher, ess |
| `CrusherBottomOut` | Hopper empty | crusher, ess |
| `CrusherShutdownStart` | Crusher delay start | crusher, delay |
| `CrusherShutdownEnd` | Crusher delay end | crusher, delay |

### 4.11 ESS Events

| Event Type | Description | Required Structs |
|------------|-------------|------------------|
| `EssInit` | ESS initialization | ess |
| `ESSChargeStart` | ESS begins charging | ess |
| `ESSChargeEnd` | ESS stops charging | ess |
| `ESSFull` | ESS fully charged | ess |

### 4.12 Passing Bay Events

| Event Type | Description | Required Structs |
|------------|-------------|------------------|
| `HaulerPassingBayYieldStart` | Begin yield at bay | hauler, passing_bay, node, battery_hauler |
| `HaulerPassingBayYieldEnd` | End yield at bay | hauler, passing_bay, node, battery_hauler |

### 4.13 Zone Shutdown Events

| Event Type | Description | Required Structs |
|------------|-------------|------------------|
| `ZoneShutdownTrigger` | Shutdown triggered | zone, delay |
| `ZoneShutdownStart` | Shutdown begins | zone, delay |
| `ZoneShutdownEnd` | Shutdown ends | zone, delay |
| `ChargerShutdownStart` | Charger delay start | charger, delay |
| `ChargerShutdownEnd` | Charger delay end | charger, delay |

---

## 5. GPS Data to Events Mapping Guide

### 5.1 Required Input Data

| Data Category | Required Fields | Update Frequency |
|---------------|-----------------|------------------|
| **Position** | Latitude, Longitude, Altitude, Timestamp | 1-10 Hz |
| **Motion** | Speed, Heading | 1-10 Hz |
| **Machine State** | Engine status, Operating mode | On change |
| **Payload** | Weight, Load status | On change |
| **Energy** | Fuel level OR SOC | 1 Hz |
| **Power** | Battery power, Motor power | 1 Hz |
| **Geofences** | Zone entry/exit | On change |

### 5.2 Event Generation Logic

#### HaulerNodeArrive
```python
Trigger: GPS position enters node boundary (within tolerance)
Required:
  - GPS timestamp → time (convert to minutes)
  - Machine ID → hauler.name, hauler.id
  - GPS velocity → hauler.speed
  - Payload sensor → hauler.payload
  - Fuel/SOC sensor → hauler.fuel_level, hauler.charge_level
  - Node ID (from geofence) → node.id, node.name
```

#### HaulerLoadEnd
```python
Trigger: Payload sensor detects load complete
Required:
  - GPS timestamp → time
  - Machine ID → hauler.name
  - Final payload → hauler.payload
  - Fuel/SOC at completion → hauler.fuel_level, hauler.charge_level
  - Load zone geofence → hauler.location = 1, hauler.location_id
```

#### HaulerIdleStart
```python
Trigger: Speed = 0 AND not in zone AND not scheduled delay
Required:
  - GPS timestamp → time
  - Machine ID → hauler.name
  - Last node ID → node.id
  - Current state inference → hauler.hauler_state = 8
```

### 5.3 State Inference Rules

| GPS/Telemetry Condition | Inferred State |
|------------------------|----------------|
| Speed > 0, Payload > 0, Location = route | Travel Loaded (0) |
| Speed > 0, Payload = 0, Location = route | Travel Unloaded (2) |
| Speed = 0, Location = load zone, Loading active | Loading (4) |
| Speed = 0, Location = dump zone, Dumping active | Dumping (5) |
| Speed = 0, Location = charge zone, Charging active | Charging (6) |
| Speed = 0, Location = route, No scheduled delay | Queuing (8) |
| Speed = 0, Scheduled delay active | Delay (9) |
| SOC = 0 OR Fuel = 0 | Stall (10) |

### 5.4 Missing Data Handling

For fields not available from GPS telemetry, use these approaches:

| Missing Field | Recommended Approach |
|--------------|---------------------|
| `totalgrade` | Calculate from road network + rolling resistance |
| `seglength` | Get from road network definition |
| Power fields | Use machine power models or set to 0 |
| TKPH fields | Use tire monitoring or set to 0, tkph_activated=False |
| `trolley_*` fields | Set to 0 if no trolley system |
| `charger_*` fields | Set to 0 if not charging |
| `cycle_count`, `route_count` | Track incrementally from events |

---

## 6. Output File Structure

### 6.1 Complete Output JSON
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
        "log_level": "record",
        "hauler": { ... },
        "node": { ... }
      },
      {
        "eid": 2,
        "time": 1.234,
        "etype": "HaulerNodeLeave",
        "log_level": "record",
        "hauler": { ... },
        "node": { ... },
        "trolley": null
      },
      ...
    ],
    "summary": { ... },
    "TUM_data": { ... },
    "optimize_data": { ... },
    "route_table": { ... },
    "dashboard_data": { ... },
    "engine_errors_data": [ ... ]
  }
}
```

### 6.2 Events-Only Format (for GPS conversion)
```json
{
  "events": [
    {
      "eid": 1,
      "time": 0.0,
      "etype": "HaulerInit",
      "log_level": "record",
      "hauler": {
        "id": 1,
        "name": "Hauler_T01",
        "speed": 0.0,
        "payload": 0.0,
        ...
      },
      "node": {
        "id": 100,
        "name": "LoadZone_1_Spot_1",
        "isTrolley": false
      }
    },
    ...
  ]
}
```

### 6.3 Version History

| Version | Date | Key Changes |
|---------|------|-------------|
| 20250818 | Latest | Added end_speed, end_speed_source |
| 20250707 | Jul 2025 | Added trolley RMS power, power limit |
| 20250623 | Jun 2025 | Major restructure, added route metrics |
| 20250514 | May 2025 | Added trolley hauler power fields |
| 20240807 | Aug 2024 | Added circuit_id, speed_limit fields |
| 20240304 | Mar 2024 | Added reactive power (kVAR) fields |

---

## 7. Validation Checklist

Before using converted GPS data as events:

- [ ] All timestamps are in minutes from simulation start
- [ ] All event IDs (eid) are unique and sequential
- [ ] All required structs for each event type are populated
- [ ] Location types match location dictionary values (0-5)
- [ ] Hauler states match hauler_state_dict values (0-18)
- [ ] All power values have correct signs (+ discharge, - charge for battery)
- [ ] Events are sorted by eid (ascending)
- [ ] All struct field names match specification exactly (case-sensitive)
- [ ] Null/None used for optional structs not applicable to event type

---

## 8. Example Events

### 8.1 HaulerNodeArrive Example (Dictionary Format)
```python
{
    # Base fields
    "eid": 15234,
    "time": 127.456,                      # minutes from simulation start
    "etype": "HaulerNodeArrive",
    "log_level": "record",
    
    # Hauler struct - all fields
    "hauler": {
        # Identity
        "id": 1,
        "uid": 1001,
        "name": "Hauler_T01",
        "model_id": 5,
        "circuit_id": 1,
        
        # Position & Location
        "orientation": "forward",
        "location": 0,                    # 0 = route
        "location_id": -1,                # -1 for routes
        "destination": 2,                 # 2 = dump
        "destination_id": 1,
        "next_stop": 2,
        "next_stop_id": 1,
        "origin": 1,                      # 1 = load
        "origin_id": 0,
        
        # Motion
        "speed": 35.5,                    # kph
        "segmentspeed": 34.2,             # kph
        "seglength": 245.5,               # meters
        "physicalgrade": 5.2,             # %
        "totalgrade": 6.8,                # %
        "speed_limit": 40.0,              # kph
        "speed_limit_source": 0,          # 0 = Design
        
        # Distance & Time
        "distance": 15234.5,              # meters total
        "route_distance": 1234.5,         # meters since zone exit
        "cycle_distance": 2456.7,         # meters since load
        "route_time": 12.5,               # minutes
        "cycle_time": 25.3,               # minutes
        "hauler_delta_time": 0.5,         # minutes since last event
        "travel_time_node_to_node": 0.5,  # minutes
        "wait_time_node_to_node": 0.0,    # minutes
        "smu": 1234.5,                    # service meter units
        "time_in_state": 0.5,             # minutes
        
        # Cycle & Count
        "cycle_count": 12,
        "route_count": 45,
        "current_material_plan": 1,
        
        # Energy - Fuel (for diesel trucks)
        "fuel_level": 0.85,               # 0-1 fraction
        "fuel_rate": 45.2,                # L/hr
        "fuel_propel_rate": 40.1,         # L/hr
        "fuel_idle_rate": 5.1,            # L/hr
        "fuel_fill_rate": 0.0,            # L/hr
        
        # Energy - Battery (for electric trucks)
        "soc": 0.0,                       # 0-1 (internal)
        "charge_level": 0.0,              # 0-1 (output)
        "energy_state": 2,
        "energy_next_stop": 0.75,         # predicted SOC at next stop
        "eta_next_stop": 5.2,             # minutes
        
        # Power - Battery
        "total_kw_battery": 0.0,          # kW (+ discharge, - charge)
        "brake_charge_kw_battery": 0.0,
        "idle_disch_kw_battery": 0.0,
        "propel_disch_kw_battery": 0.0,
        "retarding_charge_kw_battery": 0.0,
        "regen_waste_kw_battery": 0.0,
        "regen_kw_wheels": 0.0,
        "propel_kw_wheels": 0.0,
        
        # Power - Charger (when charging)
        "charger_charge_kw_source": 0.0,
        "charger_charge_kvar_source": 0.0,
        "charger_charge_kw_charger": 0.0,
        "charger_charge_kw_battery": 0.0,
        
        # Power - Trolley (when on trolley)
        "trolley_total_kw_source": 0.0,
        "trolley_total_kvar_source": 0.0,
        "trolley_propel_kw_source": 0.0,
        "trolley_propel_kvar_source": 0.0,
        "trolley_charge_kw_source": 0.0,
        "trolley_charge_kvar_source": 0.0,
        "trolley_total_kw_trolley": 0.0,
        "trolley_propel_kw_trolley": 0.0,
        "trolley_charge_kw_trolley": 0.0,
        "trolley_propel_kw_hauler": 0.0,
        "trolley_charge_kw_hauler": 0.0,
        "trolley_total_kw_battery": 0.0,
        "trolley_propel_kw_battery": 0.0,
        "trolley_charge_kw_battery": 0.0,
        "trolley_propel_fuel": 0.0,
        "trolley_demand_kw_trolley": 0.0,
        
        # TKPH
        "tkph_front": 0.0,
        "tkph_rear": 0.0,
        "tkph_trail": 0.0,
        "tkph_activated": False,
        
        # State
        "hauler_state": 0,                # 0 = Travel Loaded
        "payload": 185.5,                 # tonnes
        "battery_name": ""
    },
    
    # Node struct
    "node": {
        "id": 4523,
        "name": "R_Seg_123_N2",
        "isTrolley": False
    },
    
    # Optional structs - None when not applicable
    "trolley": None,
    "charger": None,
    "loader": None,
    "delay": None,
    "ess": None,
    "zone": None,
    "battery_hauler": None,
    "crusher": None
}
```

### 8.2 HaulerLoadEnd Example
```python
{
    "eid": 15240,
    "time": 130.125,
    "etype": "HaulerLoadEnd",
    "log_level": "record",
    
    "hauler": {
        "id": 1,
        "name": "Hauler_T01",
        "location": 1,                    # 1 = load zone
        "location_id": 0,                 # load zone ID
        "hauler_state": 4,                # 4 = Loading
        "payload": 185.5,                 # final payload after loading
        "fuel_level": 0.84,
        "charge_level": 0.0,
        "speed": 0.0,                     # stationary during load
        "cycle_count": 13,                # incremented at load start
        # ... other hauler fields
    },
    
    "battery_hauler": None                # or Battery struct if swappable
}
```

### 8.3 Chargeconnectstart Example (Electric Truck)
```python
{
    "eid": 15300,
    "time": 145.678,
    "etype": "Chargeconnectstart",
    "log_level": "record",
    
    "hauler": {
        "id": 2,
        "name": "Hauler_E01",
        "location": 3,                    # 3 = charge zone
        "location_id": 0,
        "hauler_state": 12,               # 12 = Spotting
        "charge_level": 0.25,             # SOC before charging
        "payload": 0.0,                   # empty truck charging
        "speed": 0.0,
        # ... other hauler fields
    },
    
    "charger": {
        "zone_name": "Charger_Zone_1",
        "zone_id": 0,
        "spot_id": 1,
        "node_id": 5001,
        "charger_delta_time": 0.0,
        "zone_charge_kw_source": 500.0,   # zone total power
        "zone_charge_kvar_source": 50.0,
        "zone_num_trucks": 2,
        "charger_state": 2,               # 2 = Connect
        "charge_kw_source": 0.0,          # not charging yet
        "charge_kvar_source": 0.0
    },
    
    "ess": {
        "id": 1,
        "ess_kw_battery": 0.0,
        "ess_kw_source": 500.0,
        "ess_kvar_source": 50.0,
        "soc": 0.8,
        "supply_kw_source": 1000.0,
        # ... other ESS fields
    },
    
    "battery_hauler": {
        "name": "Battery_E01_Main",
        "soc": 0.25,
        "battery_loc": "Charger_Zone_1",
        "on_truck": True,
        "battery_state": 1,               # 1 = On Truck - Charging
        "is_swappable": False,
        "charger_charge_kw_source": 0.0,
        "charger_charge_kvar_source": 0.0,
        "charger_charge_kw_charger": 0.0,
        "charger_charge_kw_battery": 0.0
    }
}
```

### 8.4 HaulerIdleStart Example (Traffic Queue)
```python
{
    "eid": 15350,
    "time": 150.234,
    "etype": "HaulerIdleStart",
    "log_level": "record",
    
    "hauler": {
        "id": 1,
        "name": "Hauler_T01",
        "location": 0,                    # 0 = route (not in zone)
        "location_id": -1,
        "hauler_state": 8,                # 8 = Queuing
        "speed": 0.0,                     # stopped
        "payload": 185.5,
        # ... other hauler fields
    },
    
    "node": {
        "id": 4530,
        "name": "R_Seg_125_N1",
        "isTrolley": False
    },
    
    "battery_hauler": None
}
```

---

*Document Version: 1.1*
*Last Updated: 2026-01-19*
*Format: Dictionary (original format)*
*Compatible with Events Version: 20250818*
