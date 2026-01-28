# DES Inputs Specification

**Version:** `CAT_2.0.2`

---

## 1. Top-Level Structure

```json
{
  "version": "CAT_2.0.2",
  "machine_specs": {
    "hauler_specs": {},
    "loader_specs": {}
  },
  "material_properties": {},
  "map_id": 1,
  "map_translate": {},
  "default_powernode_priorities": {},
  "settings": {},
  "zone_defaults": {},
  "economic_settings": {},
  "nodes": [],
  "roads": [],
  "trolleys": [],
  "load_zones": [],
  "dump_zones": [],
  "crushers": [],
  "fuel_zones": [],
  "charge_zones": [],
  "service_zones": [],
  "routes": [],
  "loaders": [],
  "haulers": [],
  "batteries": [],
  "esses": {},
  "electrical_distributions": [],
  "haulers_assignment": [],
  "operations": {},
  "override_parameters": {},
  "intersections": []
}
```

---

## 2. Network Components

### 2.1 Nodes

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | int | Yes | Unique node identifier |
| `name` | str | No | Node name |
| `coords` | list[float] | Yes | [x, y, z] coordinates (length=3) |
| `speed_limit` | float | Yes | Speed limit at node (kph) |
| `rolling_resistance` | float | Yes | Rolling resistance (%) |
| `banking` | float | No | Road banking angle (degrees) |
| `curvature` | float | No | Curvature radius (m), empty if straight |
| `lane_width` | float | No | Lane width (m) |
| `traction` | float | No | Traction coefficient |
| `waypoint` | bool | No | Is waypoint flag |

**Example:**

```json
{
  "id": 1,
  "name": "Node_1",
  "coords": [1234.56, 789.01, 100.5],
  "speed_limit": 40.0,
  "rolling_resistance": 2.5,
  "banking": 0,
  "curvature": "",
  "lane_width": 14,
  "traction": 0.6,
  "waypoint": false
}
```

### 2.2 Roads

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | int | Yes | Unique road identifier |
| `name` | str | Yes | Road name |
| `nodes` | list[int] | Yes | Ordered list of node IDs (min 2) |
| `ways_num` | int | No | Number of ways (1=one-way, 2=two-way) |
| `lanes_num` | int | No | Number of lanes per way |
| `speed_limit` | float | Yes | Default speed limit (kph) |
| `rolling_resistance` | float | Yes | Default rolling resistance (%) |
| `is_generated` | bool | Yes | True if auto-generated (zones) |
| `lane_width` | float | No | Lane width (m) |
| `banking` | float | No | Road banking (degrees) |
| `traction_coefficient` | float | No | Traction coefficient |

**Example:**

```json
{
  "id": 1,
  "name": "Haul_Road_1",
  "nodes": [1, 2, 3, 4, 5],
  "ways_num": 2,
  "lanes_num": 1,
  "speed_limit": 40.0,
  "rolling_resistance": 2.5,
  "is_generated": false,
  "lane_width": 14,
  "traction_coefficient": 0.6
}
```

### 2.3 Routes

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | int | Yes | Unique route identifier |
| `name` | str | Yes | Route name |
| `haul` | list[int] | Yes | Road IDs for haul direction (loaded) |
| `return` | list[int] | Yes | Road IDs for return direction (empty) |
| `load_zone` | int | No | Load zone ID |
| `dump_zone` | int | No | Dump zone ID |
| `production` | bool | No | Is production route |
| `used_by_current_MMP` | bool | No | Used by current material plan |

**Example:**

```json
{
  "id": 1,
  "name": "Route_LZ1_DZ1",
  "haul": [1, 2, 3],
  "return": [4, 5, 6],
  "load_zone": 1,
  "dump_zone": 1,
  "production": true
}
```

### 2.4 Intersections

| Field | Type | Description |
|-------|------|-------------|
| `id` | int | Intersection identifier |
| `name` | str | Intersection name |
| `containing` | list[int] | Node IDs within intersection |
| `entrances` | list[dict] | Entrance definitions |
| `exits` | list[dict] | Exit definitions |

---

## 3. Infrastructure Components

### 3.1 Trolleys (DET)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | int | Yes | Unique trolley identifier |
| `name` | str | Yes | Trolley name |
| `type` | str | Yes | Trolley type |
| `status` | bool | Yes | Active status |
| `nodes` | list[int] | Yes | Node IDs on trolley path |
| `ess_id` | int | No | Power node ID |
| `powernode_priority` | int | No | Power priority (0-10) |
| `connect_speed` | float | Yes | Connection speed (kph) |
| `maximum_speed` | float | Yes | Max speed on trolley (kph) |
| `hauler_speed_on_trolley` | float | Yes | Hauler speed limit on trolley |
| `line_rail_efficiency` | float | Yes | Line rail efficiency (0-1) |
| `substation_efficiency` | float | Yes | Substation efficiency (0-1) |
| `substation_output_power_limit` | float | Yes | Max power output (kW) |
| `power_module_rms_limit` | float | Yes | RMS power limit (kW) |
| `power_factor` | float | No | Power factor (0-1) |
| `power_factor_lagging` | bool | No | Power factor lagging flag |
| `rejection_rate` | float | No | Trolley rejection rate (0-1) |
| `max_propel_preferred` | bool | No | Prefer max propel power |
| `queue_at_entry` | int | No | Queue strategy (0-2) |
| `speed_reduction` | int | No | Speed reduction strategy (0-5) |
| `power_prioritization` | int | No | Power prioritization (0-11) |
| `stop_propel` | bool | No | Stop propel when power limited |
| `extra_buffer_demand_strategy` | float | No | Extra buffer (0-1) |
| `adaptive_speed_reduction_soc_target` | float | No | SOC target for adaptive speed |
| `det_rms_power` | dict | No | RMS power settings |
| `additional_data_for_animation` | dict | No | Animation metadata |

**queue_at_entry Values:**

| Value | Description |
|-------|-------------|
| 0 | No queue |
| 1 | Queue for propel power |
| 2 | Queue for propel and charge power |

**speed_reduction Values:**

| Value | Description |
|-------|-------------|
| 0 | Off/Default |
| 1 | Std propel on total |
| 2 | Std propel on propel |
| 3 | Demand |
| 4 | Demand optimistic |
| 5 | Adaptive |

**power_prioritization Values:**

| Value | Description |
|-------|-------------|
| 0 | Off |
| 1 | FIFS propel |
| 2 | FIFS all |
| 3 | LIFS propel |
| 4 | LIFS all |
| 5 | Evenly distributed |
| 6 | SOC based propel |
| 7 | SOC based all |
| 8-11 | LHDR energy based options |

**Example:**

```json
{
  "id": 1,
  "name": "Trolley_Main",
  "type": "electric",
  "status": true,
  "nodes": [10, 11, 12, 13, 14],
  "ess_id": 1,
  "powernode_priority": 2,
  "connect_speed": 5.0,
  "maximum_speed": 25.0,
  "hauler_speed_on_trolley": 25.0,
  "line_rail_efficiency": 0.95,
  "substation_efficiency": 0.98,
  "substation_output_power_limit": 9000,
  "power_module_rms_limit": 9000,
  "power_factor": 0.95,
  "rejection_rate": 0.02,
  "queue_at_entry": 0,
  "speed_reduction": 0,
  "power_prioritization": 0,
  "det_rms_power": {
    "mode": "off",
    "max_power": 9000,
    "rms_moving_time_window": 30
  }
}
```

### 3.2 Load Zones

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | int | Yes | Unique zone identifier |
| `name` | str | Yes | Zone name |
| `material` | list[str] | Yes | Material names available |
| `terminal_zone` | bool | No | Is terminal zone |
| `spots` | list | Yes | List of spot objects |
| `spots[].id` | int | Yes | Spot identifier |
| `spots[].ess_id` | int | No | Power node ID |
| `spots[].roads` | list[list[int]] | Yes | Road ID sequences [[enter, load, exit], ...] |
| `spots[].uturn_roads` | list | No | U-turn road IDs |
| `spots[].reverse_roads` | list | No | Reverse road IDs |

**Example:**

```json
{
  "id": 1,
  "name": "Load_Zone_1",
  "material": ["Ore_1700"],
  "terminal_zone": false,
  "spots": [
    {
      "id": 1,
      "ess_id": null,
      "roads": [[101, 102, 103]]
    }
  ]
}
```

### 3.3 Dump Zones

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | int | Yes | Unique zone identifier |
| `name` | str | Yes | Zone name |
| `terminal_zone` | bool | No | Is terminal zone |
| `spots` | list | Yes | List of spot objects |
| `spots[].id` | int | Yes | Spot identifier |
| `spots[].roads` | list[list[int]] | Yes | Road ID sequences |

### 3.4 Charge Zones

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | int | Yes | Unique zone identifier |
| `name` | str | Yes | Zone name |
| `output_power` | float | Yes | Charger output power (kW) |
| `efficiency` | float | Yes | Charger efficiency (0-1) |
| `cable_efficiency` | float | Yes | Cable efficiency (0-1) |
| `ramup_time` | float | Yes | Ramp-up time (min) |
| `connect_time` | float | Yes | Connection time (min) |
| `disconnect_time` | float | Yes | Disconnection time (min) |
| `power_factor` | float | No | Power factor (0-1) |
| `power_factor_lagging` | bool | No | Power factor lagging |
| `terminal_zone` | bool | No | Is terminal zone |
| `battery_swap` | bool | No | Battery swap enabled |
| `num_charger` | int | No | Number of chargers (swap) |
| `additional_battery` | int | No | Additional batteries (swap) |
| `cooldown_time_min` | float | No | Cooldown time (min) |
| `chargers` | list | No | Charger details for swap |
| `spots` | list | Yes | List of spot objects |
| `spots[].id` | int | Yes | Spot identifier |
| `spots[].ess_id` | int | No | Power node ID |
| `spots[].powernode_priority` | int | No | Power priority |
| `spots[].roads` | list[list[int]] | Yes | Road ID sequences [[enter, exit], ...] |
| `additional_data_for_animation` | dict | No | Animation metadata |

**Example:**

```json
{
  "id": 1,
  "name": "Charger_Zone_1",
  "output_power": 1500,
  "efficiency": 0.95,
  "cable_efficiency": 0.98,
  "ramup_time": 0.5,
  "connect_time": 1.0,
  "disconnect_time": 0.5,
  "power_factor": 0.95,
  "terminal_zone": false,
  "spots": [
    {
      "id": 1,
      "ess_id": 1,
      "powernode_priority": 1,
      "roads": [[201, 202]]
    }
  ]
}
```

### 3.5 Fuel Zones

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | int | Yes | Unique zone identifier |
| `name` | str | Yes | Zone name |
| `fuel_rate` | float | No | Fuel fill rate (L/min) |
| `connect_time` | float | Yes | Connection time (min) |
| `disconnect_time` | float | Yes | Disconnection time (min) |
| `terminal_zone` | bool | No | Is terminal zone |
| `spots` | list | Yes | List of spot objects |

### 3.6 Service Zones

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | int | Yes | Unique zone identifier |
| `name` | str | No | Zone name |
| `terminal_zone` | bool | No | Is terminal zone |
| `is_show_service` | bool | No | Show in service display |
| `spots` | list | Yes | List of spot objects |
| `spots[].id` | int | Yes | Spot identifier |
| `spots[].roads` | list[list[int]] | Yes | Road ID sequences |
| `additional_data_for_animation` | dict | No | Animation metadata |

### 3.7 Crushers

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | int | Yes | Crusher identifier |
| `name` | str | Yes | Crusher name |
| `hopper_size` | float | Yes | Hopper capacity (tonnes) |
| `process_rate` | float | Yes | Process rate (tonnes/hr) |
| `dump_zone_id` | int | Yes | Associated dump zone ID |
| `spot_ids` | list[int] | Yes | Associated spot IDs |
| `dump_time` | float | No | Dump time (min) |
| `ess_id` | int | No | Power node ID |
| `powernode_priority` | int | No | Power priority |
| `power_factor` | float | No | Power factor |
| `idle_power_kW` | float | No | Idle power consumption (kW) |
| `active_power_kW` | float | No | Active power consumption (kW) |

### 3.8 ESS (Power Nodes)

ESS objects are stored as a dictionary with string keys.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | int | Yes | Unique ESS identifier |
| `name` | str | Yes | ESS name |
| `unlimited_power` | bool | No | Unlimited power flag |
| `line_power` | float/str | No | Line power limit (kW) or "unlimited" |
| `include_in_economic_calculations` | bool | No | Include in TCO |
| `ess_attached` | bool | No | Has battery storage |
| `out_power_max` | float | No | Max output power (kW) |
| `capacity` | float | No | Battery capacity (kWh) |
| `out_eff` | float | No | Output efficiency (0-1) |
| `in_eff` | float | No | Input efficiency (0-1) |
| `soc_init` | float | No | Initial SOC (0-1) |
| `SOC_Min` | float | No | Minimum SOC (0-1) |
| `SOC_Max` | float | No | Maximum SOC (0-1) |
| `C_Rate` | float | No | C-Rate charging |
| `power_queuing_prioritization` | dict | No | Power queuing settings |
| `additional_data_for_animation` | dict | No | Position for animation |

**Example:**

```json
{
  "1": {
    "id": 1,
    "name": "Power_Node_1",
    "unlimited_power": false,
    "line_power": 15000,
    "include_in_economic_calculations": true,
    "ess_attached": true,
    "out_power_max": 5000,
    "capacity": 10000,
    "out_eff": 0.95,
    "in_eff": 0.95,
    "soc_init": 0.8,
    "SOC_Min": 0.1,
    "SOC_Max": 0.9,
    "C_Rate": 0.5
  }
}
```

---

## 4. Machine Components

### 4.1 Haulers

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | int | Yes | Unique hauler identifier |
| `name` | str | Yes | Hauler name (e.g., "Hauler_1_1") |
| `group` | str | No | Hauler group name |
| `type` | str | Yes | "diesel" or "electric" |
| `model_id` | int | Yes | Machine model ID |
| `machine_name` | str | No | Machine config name |
| `hauler_group_id` | int | No | Group ID for dispatching |
| `initial_position` | int | Yes | 1=ON_ROUTE, 2=SERVICE_ZONE |
| `initial_conditions` | dict | Yes | Initial state |
| `initial_conditions.route_id` | int | Yes | Initial route ID |
| `initial_conditions.road_id` | int | Yes | Initial road ID |
| `initial_conditions.node_id` | int | Yes | Initial node ID |
| `initial_fuel_level_pct` | float | No | Initial fuel level (0-1) for diesel |
| `initial_charge_level_pct` | float | No | Initial SOC (0-1) for electric |
| `battery_state_of_health` | float | No | Battery SOH (0-1) |
| `EndOfLifeSOH` | float | No | End of life SOH threshold |
| `AvgAnnualAmbientTemp` | float | No | Average ambient temperature (°C) |
| `CoolingActivationTemperature` | float | No | Cooling activation temp |
| `RefridgerationActivationTemperature` | float | No | Refrigeration activation temp |

**Example:**

```json
{
  "id": 1,
  "name": "Hauler_Fleet1_1",
  "group": "Fleet1",
  "type": "electric",
  "model_id": 1,
  "machine_name": "793BEM_1440",
  "hauler_group_id": 1,
  "initial_position": 1,
  "initial_conditions": {
    "route_id": 1,
    "road_id": 1,
    "node_id": 5
  },
  "initial_charge_level_pct": 0.95,
  "battery_state_of_health": 0.9,
  "EndOfLifeSOH": 84.7,
  "AvgAnnualAmbientTemp": 25
}
```

### 4.2 Loaders

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | int | Yes | Unique loader identifier |
| `name` | str | Yes | Loader name |
| `model_id` | int | Yes | Machine model ID |
| `used_for` | str | No | "Truck Loading" or other |
| `machine_name` | str | No | Machine config name |
| `initial_conditions.load_zone_id` | int | Yes | Load zone ID |
| `initial_conditions.spot_id` | int | Yes | Spot ID within zone |
| `fill_factor_pct` | float | No | Fill factor percentage (0-1.5) |
| `powernode_priority` | int | No | Power node priority |
| `initial_charge_fuel_levels_pct` | float | No | Initial fuel/charge (0-1) |

**Example:**

```json
{
  "id": 1,
  "name": "Loader_1 (1)",
  "model_id": 101,
  "used_for": "Truck Loading",
  "machine_name": "994K",
  "initial_conditions": {
    "load_zone_id": 1,
    "spot_id": 1
  },
  "fill_factor_pct": 1.0,
  "powernode_priority": 0
}
```

### 4.3 Batteries

| Field | Type | Description |
|-------|------|-------------|
| `name` | str | Battery name |
| `type` | str | Battery chemistry type |
| `battery_size` | float | Battery capacity (kWh) |
| `location` | str | Current location |
| `machine_name` | str | Associated machine name |
| `battery_state_of_health` | float | SOH (0-1) |

### 4.4 Haulers Assignment

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | int | Yes | Assignment identifier |
| `hauler_id` | int | Yes | Hauler ID |
| `load_zone` | list[int] | Yes | Assigned load zone IDs |
| `dump_zone` | list[int] | Yes | Assigned dump zone IDs |
| `fuel_zone_assigned` | list[int] | Yes | Assigned fuel zone IDs |
| `charge_zone_assigned` | list[int] | Yes | Assigned charge zone IDs |
| `service_zone_assigned` | list[int] | Yes | Assigned service zone IDs |

---

## 5. Machine Specs

### 5.1 Hauler Specs

```json
{
  "hauler_specs": {
    "1": {
      "machine": {
        "machine": {
          "MachineType": 6,
          "EmptyWeight": 165000,
          "PayloadIndex": 227000,
          "HaulerMax": 392000,
          "Machine Fuel Tank Capacity (L)": 3785,
          "battery_size": 1440,
          "battery_type": "LFP",
          "GHGFreeConfig": "BEV",
          "TransType": "Electric"
        },
        "rimpull": [...],
        "retarding": [...],
        "trolley_header": [...]
      }
    }
  }
}
```

### 5.2 Loader Specs

```json
{
  "loader_specs": {
    "101": {
      "loader": {
        "LoaderName": "994K",
        "MachineType": 11,
        "IsHauler": 0,
        "BucketCapacity": 23.0,
        "type": "diesel"
      },
      "is_hybrid": false
    }
  }
}
```

---

## 6. Settings

### 6.1 Global Settings

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `sim_time` | float | Yes | Simulation duration (min) |
| `random_seed` | int | Yes | Random seed |
| `fpc_engine` | str | No | FPC engine name |
| `intersection_system` | str | No | "none", "safe", "simple", "advanced" |
| `bump_prevention` | str | No | "off", "fpc_based", "physics_based" |
| `road_network_logic` | bool | No | Enable road network logic |
| `road_traction_coefficient` | float | No | Default traction |
| `reassignment_threshold_min` | float | No | Reassignment threshold (min) |
| `passing_bay_logic` | bool | No | Enable passing bay |
| `passing_bay_waiting_time` | float | No | Waiting time (min) |
| `max_SOC_logic_for_DET` | bool | No | Max SOC for DET |
| `max_SOC_logic_for_SET` | bool | No | Max SOC for SET |
| `lane_width` | float | Yes | Lane width (m) |
| `banking` | float | No | Default banking |
| `gap_between_lanes` | float | No | Gap between lanes (m) |
| `distance_between_lanes` | float | Yes | Total distance between lanes |
| `driving_side` | int | No | 0=LEFT, 1=RIGHT |
| `fuel_when_to_fill_lvl` | float | No | Fuel trigger level (0-1) |
| `variation_model` | int | No | -1=off, 0=on |
| `target_payload` | float | No | Target payload factor |
| `payload_precision` | float | No | Payload precision |
| `loader_time_variation` | float | No | Loader time variation (0-1) |
| `loader_payload_variation` | float | No | Loader payload variation (0-1) |
| `truck_time_variation` | float | No | Truck time variation (0-1) |
| `truck_payload_variation` | float | No | Truck payload variation |
| `operational_delays` | str | No | Delay type |
| `min_foll_dist` | float | Yes | Min following distance (m) |
| `verbose` | bool | Yes | Verbose logging |
| `reduce_speed_logic` | bool | Yes | Enable speed reduction |
| `braking` | bool | Yes | Enable braking model |
| `objective` | str | Yes | Simulation objective |
| `log_level` | str | Yes | "record" or "debug" |
| `spd_lim` | float | Yes | Speed limit override |
| `charger_thresholds` | list[float] | Yes | Charger thresholds |
| `charger_connect_times` | list[float] | Yes | Connect times |
| `initial_fuel_level_pct` | float | Yes | Default fuel level |
| `initial_charge_level_pct` | float | Yes | Default charge level |
| `attach_to_trolley_speed` | float | Yes | Trolley attach speed (kph) |
| `battery_state_of_health` | float | No | Default SOH |
| `battery_min_pct` | float | No | Min battery % |
| `battery_max_pct` | float | No | Max battery % |
| `battery_charge_pct` | float | No | Target charge % |
| `battery_trolley_pct` | float | No | Trolley charge target |
| `material_density` | float | No | Default material density |
| `calculate_BLE` | bool | Yes | Calculate battery life |
| `new_prerun` | bool | No | New prerun algorithm |
| `power_loss_model` | bool | No | Enable power loss model |
| `intersection_dispatching` | bool | No | Intersection dispatching |
| `ambient_temperature` | float | No | Ambient temperature (°C) |

### 6.2 Fueling/Charging Dispatching

```json
{
  "fueling_charging_dispatching": {
    "prerun_soc_buffer": 0.05,
    "prerun_soc_buffer_trolley": 0.25,
    "soe_penalty": 1,
    "charging_strategy": "rule_based",
    "fueling_charging_dispatching_policy": "ccv_bem",
    "secondary_check_for_CZ_dispatch": true
  }
}
```

**charging_strategy Values:**

| Value | Description |
|-------|-------------|
| `rule_based` | Rule-based dispatching |
| `machine_learning` | ML-based dispatching |

**fueling_charging_dispatching_policy Values:**

| Value | Description |
|-------|-------------|
| `ccv_bem` | CCV BEM policy |
| `ccv_betm` | CCV BETM policy |

### 6.3 Speed Limits

```json
{
  "speed_limits": {
    "curvature_based": true,
    "corner_speed_limit": false,
    "driver_speed_behavior": false,
    "curvature_speed_application": "on",
    "cfh_version": "v1",
    "grade_limits": {
      "status": false,
      "loaded": {
        "defined_grades": [-20, -5, 0, 5, 15],
        "defined_speed_limits": [10, 20, 40, 30, 15]
      },
      "empty": {
        "defined_grades": [-30, -5, 0, 5, 15],
        "defined_speed_limits": [15, 30, 60, 15, 15]
      },
      "interpolate": false
    }
  }
}
```

### 6.4 Battery Life Settings

```json
{
  "battery_life_settings": {
    "version": 0,
    "days_per_week": 7,
    "SchedulePeriod": 0,
    "HoursPerInterval": 5000,
    "DayToNightTempSwing": 0,
    "SummerToWinterTempSwing": 0,
    "AmbientToUnderhoodTempDelta": 5,
    "CurrentLimit": 1,
    "TimeStepCycle": 5,
    "TimeStepCalendar": 60,
    "DiagnosticDay": 60,
    "SoCoptimisation": 0,
    "SoCtargetHardlimit": 1,
    "MaxYearsOverride": 20,
    "CalendarAgeing": 1,
    "VoltageCorrection": 0,
    "IncludeModuleMass": 1
  }
}
```

### 6.5 Tires/TKPH Settings

```json
{
  "tires": {
    "calculate_TKPH": false,
    "TKPH_RollingWindow": 60,
    "TKPH_ambient_temperature": 34,
    "front_tire_TKPH_limit": 1394,
    "rear_tire_TKPH_limit": 1394,
    "TKPH_speed_adjustment": false,
    "front_tire_TKPH_deactivate_limit": 10000,
    "rear_tire_TKPH_deactivate_limit": 10000,
    "TKPH_limiting_speed": 25
  }
}
```

### 6.6 Intersections Settings

```json
{
  "intersections": {
    "intersect_logic": true,
    "intersect_length": 10
  }
}
```

---

## 7. Economic Settings

| Field | Type | Description |
|-------|------|-------------|
| `calculate_TCO` | bool | Calculate TCO flag |
| `discount_rate` | float | Discount rate (0-1) |
| `fuel` | float | Fuel cost ($/L) |
| `def` | float | DEF cost ($/L) |
| `electricity` | list[dict] | Electricity costs [{cost, annual_weighting}] |
| `demand_charge` | list[dict] | Demand charges [{cost, annual_weighting}] |
| `carbon_tax` | float | Carbon tax ($/tonne CO2) |
| `fuel_carbon_emissions` | float | Fuel emissions (kg CO2/L) |
| `electricity_carbon_emissions` | float | Grid emissions (kg CO2/kWh) |
| `ESS` | float | ESS cost ($/kWh) |
| `charger` | float | Charger cost ($/kW) |
| `substation` | float | Substation cost ($/kVA) |
| `trolley_site_infrastructure` | float | Trolley infrastructure cost ($) |
| `fuel_infrastructure` | float | Fuel infrastructure cost ($) |
| `autonomy` | float | Autonomy cost ($) |
| `aets` | float | AETS cost ($) |
| `utilization_of_calendar_hours` | float | Calendar utilization (0-1) |
| `mechanical_availability` | float | Mechanical availability (0-1) |
| `ess_maintenance` | float | ESS maintenance rate |
| `fuel_infrastructure_maintenance` | float | Fuel maintenance rate |
| `trolley_infrastructure_maintenance` | float | Trolley maintenance rate |
| `charging_infrastructure_maintenance` | float | Charger maintenance rate |
| `autonomy_site_maintenance` | float | Autonomy maintenance rate |
| `autonomy_fleet_operator` | float | Fleet operator cost ($) |
| `autonomy_fleet_operator_size` | int | Fleet size for operator |
| `partial_truck_round` | float | Partial truck rounding |
| `adjust_fleet_for_availability` | bool | Adjust fleet for availability |
| `use_shift_scheduler` | bool | Use shift scheduler |
| `Working_Days_per_Week` | int | Working days |
| `Shifts_per_day` | int | Shifts per day |
| `Hours_per_Shift` | float | Hours per shift |
| `user_tco_config_name_lookup` | dict | Per-machine TCO configs |

---

## 8. Operations

### 8.1 Material Schedules

```json
{
  "material_schedules": {
    "selected_material": 1,
    "all_material_schedule": [
      {
        "id": 1,
        "name": "Default_MMP",
        "hauler_assignment": {
          "scheduling_method": "production_target_based"
        },
        "mixed_fleet_based_initial_assignment": false,
        "data": [
          {
            "load_zone": "Load_Zone_1",
            "dump_zone": "Dump_Zone_1",
            "route": "",
            "material": "Ore",
            "density": 1700,
            "quantity": 10000,
            "plan_group_id": 1,
            "assigned_machine_type": "Hauler"
          }
        ],
        "plan_group": [
          {
            "id": 1,
            "hauler_group_id": [
              {"id": 1, "num_of_hauler": 5}
            ]
          }
        ]
      }
    ]
  }
}
```

**scheduling_method Values:**

| Value | Description |
|-------|-------------|
| `grouped_assignment` | Direct hauler-route assignment |
| `minestar_dispatching` | MineStar dispatching |
| `truck_schedule_based` | Schedule-based |
| `default_production_target_based` | Simple production target |
| `production_target_based` | Advanced production target |

### 8.2 Operational Delays

```json
{
  "operational_delays": {
    "haulers": [
      {
        "id": 1,
        "name": "Shift_Change",
        "action": "pullover",
        "start_time": 480,
        "duration": 30,
        "applied_to": ["Hauler_Fleet1_1", "Hauler_Fleet1_2"],
        "repeat": {
          "interval": 720,
          "count": 2
        },
        "smu": false,
        "dependent": false,
        "keyon": false,
        "mtbs_interval": true
      }
    ],
    "trolleys": [],
    "load_zones": [],
    "dump_zones": [],
    "charge_zones": [],
    "crushers": []
  }
}
```

**action Values:**

| Value | Description |
|-------|-------------|
| `pullover` | Pull over and stop |
| `service zone` | Go to service zone |
| `derate_speed` | Reduce speed |

---

## 9. Material Properties

```json
{
  "material_properties": {
    "Ore_1700": {
      "id": 1,
      "material": "Ore",
      "density": 1700
    },
    "Waste_1800": {
      "id": 2,
      "material": "Waste",
      "density": 1800
    }
  }
}
```

---

## 10. Override Parameters

```json
{
  "override_parameters": {
    "simulation": {},
    "exploration": {
      "param_name": "value"
    }
  }
}
```

---

## 11. Default Power Node Priorities

```json
{
  "default_powernode_priorities": {
    "loaders": 0,
    "trolleys": 2,
    "chargers": 1,
    "crushers": 0
  }
}
```
