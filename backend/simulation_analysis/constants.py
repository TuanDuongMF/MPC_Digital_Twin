"""
Constants and enumerations for simulation events conversion.

Based on events_structure_specification.md
"""

from enum import IntEnum
from typing import Dict


class HaulerState(IntEnum):
    """Hauler state codes as defined in events specification."""
    TRAVEL_LOADED = 0
    TRAVEL_LOADED_TROLLEY = 1
    TRAVEL_UNLOADED = 2
    TRAVEL_UNLOADED_TROLLEY = 3
    LOADING = 4
    DUMPING = 5
    CHARGING = 6
    FUELING = 7
    QUEUING = 8
    DELAY = 9
    STALL = 10
    FINISHED = 11
    SPOTTING = 12
    TROLLEY_QUEUE = 13
    CRUSHER_WAIT = 14
    CHARGING_ALT = 15
    DELAY_AND_CHARGE = 16
    DELAY_AND_SWAP = 17
    PASSING_BAY_DELAY = 18


class LocationType(IntEnum):
    """Location type codes."""
    ROUTE = 0
    LOAD = 1
    DUMP = 2
    CHARGE = 3
    FUEL = 4
    SERVICE = 5


class SpeedLimitSource(IntEnum):
    """Speed limit source codes."""
    DESIGN = 0
    MACHINE = 1
    GRADE = 2
    TKPH = 3
    TROLLEY = 4
    ENERGY = 5


# Mapping from AMT segment types to hauler states
SegmentTypeMapping: Dict[str, HaulerState] = {
    "SPOTTING_AT_SOURCE": HaulerState.SPOTTING,
    "SPOTTING_AT_SINK": HaulerState.SPOTTING,
    "TRAVELLING_EMPTY": HaulerState.TRAVEL_UNLOADED,
    "TRAVELLING_FULL": HaulerState.TRAVEL_LOADED,
    "Spotting.At.Source": HaulerState.SPOTTING,
    "Spotting.At.Sink": HaulerState.SPOTTING,
    "Travelling.Empty": HaulerState.TRAVEL_UNLOADED,
    "Travelling.Full": HaulerState.TRAVEL_LOADED,
}

# Mapping from AMT segment types to location types
SegmentToLocation: Dict[str, LocationType] = {
    "SPOTTING_AT_SOURCE": LocationType.LOAD,
    "SPOTTING_AT_SINK": LocationType.DUMP,
    "TRAVELLING_EMPTY": LocationType.ROUTE,
    "TRAVELLING_FULL": LocationType.ROUTE,
    "Spotting.At.Source": LocationType.LOAD,
    "Spotting.At.Sink": LocationType.DUMP,
    "Travelling.Empty": LocationType.ROUTE,
    "Travelling.Full": LocationType.ROUTE,
}


# Default hauler struct template with all required fields
DEFAULT_HAULER_STRUCT = {
    # Identity
    "id": 0,
    "uid": 0,
    "name": "",
    "model_id": 1,
    "circuit_id": 1,
    
    # Position & Location
    "orientation": "forward",
    "location": LocationType.ROUTE,
    "location_id": -1,
    "destination": LocationType.DUMP,
    "destination_id": -1,
    "next_stop": LocationType.DUMP,
    "next_stop_id": -1,
    "origin": LocationType.LOAD,
    "origin_id": -1,
    
    # Motion
    "speed": 0.0,
    "segmentspeed": 0.0,
    "seglength": 0.0,
    "physicalgrade": 0.0,
    "totalgrade": 0.0,
    "speed_limit": 40.0,
    "speed_limit_source": SpeedLimitSource.DESIGN,
    
    # Distance & Time
    "distance": 0.0,
    "route_distance": 0.0,
    "cycle_distance": 0.0,
    "route_time": 0.0,
    "cycle_time": 0.0,
    "hauler_delta_time": 0.0,
    "travel_time_node_to_node": 0.0,
    "wait_time_node_to_node": 0.0,
    "smu": 0.0,
    "time_in_state": 0.0,
    
    # Cycle & Count
    "cycle_count": 0,
    "route_count": 0,
    "current_material_plan": 1,
    
    # Energy - Fuel (diesel)
    "fuel_level": 1.0,
    "fuel_rate": 0.0,
    "fuel_propel_rate": 0.0,
    "fuel_idle_rate": 0.0,
    "fuel_fill_rate": 0.0,
    
    # Energy - Battery (electric)
    "soc": 0.0,
    "charge_level": 0.0,
    "energy_state": 0,
    "energy_next_stop": 0.0,
    "eta_next_stop": 0.0,
    
    # Power - Battery
    "total_kw_battery": 0.0,
    "brake_charge_kw_battery": 0.0,
    "idle_disch_kw_battery": 0.0,
    "propel_disch_kw_battery": 0.0,
    "retarding_charge_kw_battery": 0.0,
    "regen_waste_kw_battery": 0.0,
    "regen_kw_wheels": 0.0,
    "propel_kw_wheels": 0.0,
    
    # Power - Charger
    "charger_charge_kw_source": 0.0,
    "charger_charge_kvar_source": 0.0,
    "charger_charge_kw_charger": 0.0,
    "charger_charge_kw_battery": 0.0,
    
    # Power - Trolley
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
    "hauler_state": HaulerState.TRAVEL_UNLOADED,
    "payload": 0.0,
    "battery_name": "",
}


# Default node struct template
DEFAULT_NODE_STRUCT = {
    "id": 0,
    "name": "",
    "isTrolley": False,
}


# Event types used in conversion
class EventType:
    """Event type names."""
    HAULER_INIT = "HaulerInit"
    HAULER_NODE_ARRIVE = "HaulerNodeArrive"
    HAULER_NODE_LEAVE = "HaulerNodeLeave"
    HAULER_ENTER_ZONE = "HaulerEnterZone"
    HAULER_ENTER_ROUTE = "HaulerEnterRoute"
    HAULER_LOAD_START = "HaulerLoadStart"
    HAULER_LOAD_END = "HaulerLoadEnd"
    HAULER_DUMP_START = "HaulerDumpStart"
    HAULER_DUMP_END = "HaulerDumpEnd"
    HAULER_IDLE_START = "HaulerIdleStart"
    HAULER_IDLE_END = "HaulerIdleEnd"


# Payload threshold for empty/loaded classification (percentage)
PAYLOAD_THRESHOLD = 50

# Typical payload capacity in kg (for conversion from percentage)
DEFAULT_PAYLOAD_CAPACITY = 227000.0  # kg (Cat 797F, 227 tonnes)
