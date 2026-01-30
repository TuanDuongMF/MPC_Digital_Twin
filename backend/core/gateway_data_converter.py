"""
Gateway Data Converter

Converts JSON data from gateway parser (GWMReader) to AMTCycleProdInfoMessage objects.

Uses logic from parse_gateway_messages.py to process parser output and convert to database format.

ACTUAL JSON STRUCTURE FROM GWMReader:
{
    "CycleProdInfo": {
        "208188426": [  // machineId as key
            [208188426, 1440799449, '2025-09-01T22:03:51.17', 95.185, 95.728, 8026.64, 12320.37, 276.73, 755.52, 12.2, 12, 12, 11.4, 9, 9.3, 0, -34, 7, 27, 5, 0, 58, 31, 0],
            // ... 24 elements per record
        ]
    }
}

VERIFIED COLUMN MAPPING (based on parse_gateway_messages.py):
JSON Array Index | Database Column | Description | Unit/Notes
-----------------|-----------------|-------------|------------
[0]              | machineId | Machine Unique Id (208188426) | int
[1]              | segmentId | GPS timestamp (1440799449) | int (seconds since GPS epoch)
[2]              | Time | ISO timestamp string ('2025-09-01T22:03:51.17') | datetime string
[3]              | expectedElapsedTime | Expected elapsed (95.185) | float (already in seconds)
[4]              | actualElapsedTime | Actual elapsed (95.728) | float (already in seconds)
[5]              | pathEasting | X coordinate (8026.64) | float (already in meters)
[6]              | pathNorthing | Y coordinate (12320.37) | float (already in meters)
[7]              | pathElevation | Z coordinate (276.73) | float (already in meters)
[8]              | plannedDistance | Distance (755.52) | float (already in meters)
[9]              | expectedSpeed | Expected speed (12.2) | float (m/s, convert to km/h: *3.6)
[10]             | actualSpeed | Actual speed (12) | float (m/s, convert to km/h: *3.6)
[11]             | expectedDesiredSpeed | Expected desired (12) | float (m/s, convert to km/h: *3.6)
[12]             | actualDesiredSpeed | Actual desired (11.4) | float (m/s, convert to km/h: *3.6)
[13]             | leftWidth | Left width (9) | float (already in meters)
[14]             | rightWidth | Right width (9.3) | float (already in meters)
[15]             | pathBank | Road banking (0) | float (degrees)
[16]             | pathHeading | Direction (-34) | float (degrees)
[17]             | payloadPercent | Payload % (7) | int (0-200, special encoding)
[18]             | expectedSpeedSource | Expected source (27) | int
[19]             | expectedASLR | Expected ASLR (5) | int
[20]             | expectedRegModEnum | Expected reg mod (0) | int
[21]             | actualSpeedSource | Actual source (58) | int
[22]             | actualASLR | Actual ASLR (31) | int
[23]             | actualRegModEnum | Actual reg mod (0) | int

Note: Logic from parse_gateway_messages.py processes 24-element arrays and converts them to dict format
with database column names. Then we convert these dicts to tuple format for AMTCycleProdInfoMessage.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timezone, timedelta
from .Reader import AMTCycleProdInfoReader
from .Cycle import Cycle
from .Zone import Zone
from .constants import gps_epoch, leap_seconds


# Helper functions from parse_gateway_messages.py
def safe_int(val) -> Optional[int]:
    """Safely convert to int."""
    if val is None:
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


def safe_payload(val) -> Optional[int]:
    """Handle payload percent special encoding."""
    if val is None:
        return None
    try:
        v = int(val)
        return v if v <= 200 else v - 255
    except (ValueError, TypeError):
        return None


def extract_cp_records(parser_output: Dict[str, Any]) -> Tuple[Dict[str, List], bool]:
    """Extract CycleProdInfo records from parser output."""
    if 'CycleProdInfo2' in parser_output and parser_output['CycleProdInfo2']:
        return parser_output['CycleProdInfo2'], True
    if 'CycleProdInfo' in parser_output and parser_output['CycleProdInfo']:
        return parser_output['CycleProdInfo'], False
    return {}, False


def parse_message_to_dict(data: List) -> Dict[str, Any]:
    """
    Parse a raw message tuple into a dictionary with DB column names.
    Based on parse_gateway_messages.py logic.

    Raw data format (24 fields from parser):
        0:  machineId
        1:  segmentId/cycleId
        2:  start_time
        3:  expectedElapsedTime
        4:  actualElapsedTime
        5:  pathEasting
        6:  pathNorthing
        7:  pathElevation
        8:  plannedDistance
        9:  expectedSpeed
        10: actualSpeed
        11: expectedDesiredSpeed
        12: actualDesiredSpeed
        13: leftWidth
        14: rightWidth
        15: pathBank
        16: pathHeading
        17: payloadPercent
        18: expectedSpeedSource
        19: expectedASLR
        20: expectedRegModEnum
        21: actualSpeedSource
        22: actualASLR
        23: actualRegModEnum
    """
    return {
        "expectedElapsedTime": safe_int(data[3]),
        "actualElapsedTime": safe_int(data[4]),
        "pathEasting": safe_int(data[5]),
        "pathNorthing": safe_int(data[6]),
        "pathElevation": safe_int(data[7]),
        "plannedDistance": safe_int(data[8]),
        "expectedSpeed": safe_int(data[9]),
        "actualSpeed": safe_int(data[10]),
        "expectedDesiredSpeed": safe_int(data[11]),
        "actualDesiredSpeed": safe_int(data[12]),
        "leftWidth": safe_int(data[13]),
        "rightWidth": safe_int(data[14]),
        "pathBank": safe_int(data[15]),
        "pathHeading": safe_int(data[16]),
        "payloadPercent": safe_payload(data[17]),
        "expectedSpeedSource": safe_int(data[18]),
        "expectedASLR": safe_int(data[19]),
        "expectedRegModEnum": safe_int(data[20]),
        "actualSpeedSource": safe_int(data[21]),
        "actualASLR": safe_int(data[22]),
        "actualRegModEnum": safe_int(data[23]),
    }


def process_parser_output(parser_output: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Process parser output and return list of JSON objects with DB column names.
    Based on parse_gateway_messages.py logic - keeps original logic from source file.

    Args:
        parser_output: Raw output from the gateway parser

    Returns:
        List of dictionaries with keys matching database columns (without machineId, segmentId, Time)
    """
    records_by_ip, is_cp2 = extract_cp_records(parser_output)

    if not records_by_ip:
        return []

    all_records = []

    for ip_address, messages in records_by_ip.items():
        for msg in messages:
            if len(msg) < 24:
                continue

            record = parse_message_to_dict(msg)
            all_records.append(record)

    return all_records


def convert_gateway_json_to_objects(
    json_data: Dict[str, Any],
    machine_info: Optional[Dict[str, Any]] = None,
    site_name: Optional[str] = None
) -> Tuple[Optional[List[Cycle]], Optional[List[Zone]], Dict[str, Any]]:
    """
    Convert JSON data from gateway parser to Cycle and Zone objects.
    
    Args:
        json_data: JSON data from gateway parser (GWMReader output)
        machine_info: Machine information dict with "Name" and "TypeName"
        site_name: Site name (optional, for logging)
    
    Returns:
        Tuple of (cycles, zones, metadata):
        - cycles: List of Cycle objects or None
        - zones: List of Zone objects or None  
        - metadata: Dictionary with conversion metadata (stats, errors, etc.)
    """
    metadata = {
        "total_messages": 0,
        "converted_messages": 0,
        "errors": [],
        "warnings": []
    }
    
    if not json_data:
        metadata["errors"].append("Empty JSON data")
        return None, None, metadata
    
    # Use logic from parse_gateway_messages.py to process parser output
    # This returns list of dicts with database column names (identical to parse_gateway_messages.py)
    db_records = process_parser_output(json_data)
    
    if not db_records:
        metadata["errors"].append("No message data found in JSON")
        return None, None, metadata
    
    metadata["total_messages"] = len(db_records)
    
    # Extract raw messages from parser output to get machineId, segmentId, Time
    # (required for tuple format but not present in db_records)
    records_by_ip, _ = extract_cp_records(json_data)
    raw_messages = []
    if records_by_ip:
        for ip_address, messages in records_by_ip.items():
            for msg in messages:
                if len(msg) >= 24:
                    raw_messages.append(msg)
    
    # Convert dict records + raw messages to tuple format for AMTCycleProdInfoMessage
    tuples_data = _convert_db_records_to_tuples(db_records, raw_messages, metadata)
    
    if not tuples_data:
        metadata["errors"].append("Failed to convert messages to tuple format")
        return None, None, metadata
    
    # Default machine info if not provided
    if not machine_info:
        machine_info = {
            "Name": site_name or "Unknown",
            "TypeName": "Unknown"
        }
        metadata["warnings"].append("Using default machine info")
    
    # Use Reader to parse and create objects
    try:
        # Validate tuple format before parsing
        if tuples_data:
            sample_tuple = tuples_data[0]
            if len(sample_tuple) != 25:
                metadata["errors"].append(
                    f"Invalid tuple length: {len(sample_tuple)}, expected 25. "
                    f"This indicates field mapping may be incorrect. "
                    f"Check JSON structure and adjust mapping in gateway_data_converter.py"
                )
                return None, None, metadata
        
        cycles, zones = AMTCycleProdInfoReader.parse_cp1_data(tuples_data, machine_info)
        metadata["converted_messages"] = len(tuples_data)
        metadata["total_cycles"] = len(cycles) if cycles else 0
        metadata["total_zones"] = len(zones) if zones else 0
        
        # Add warning if no cycles/zones created (might indicate mapping issue)
        if metadata["converted_messages"] > 0 and metadata["total_cycles"] == 0:
            metadata["warnings"].append(
                "No cycles created from messages. This may indicate incorrect field mapping. "
                "Please verify JSON structure matches expected format."
            )
        
        return cycles, zones, metadata
    except Exception as e:
        error_msg = f"Error parsing data: {str(e)}"
        metadata["errors"].append(error_msg)
        # Add helpful context for common errors
        if "tuple index out of range" in str(e).lower() or "index" in str(e).lower():
            metadata["errors"].append(
                "Tuple index error suggests field order mismatch. "
                "Check JSON structure and verify tuple order matches AMTCycleProdInfoMessage constructor."
            )
        return None, None, metadata


def _extract_messages_from_json(json_data: Dict[str, Any], metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract message data from JSON structure.
    Handles different possible JSON formats from gateway parser.
    
    Actual structure from GWMReader:
    {
        "CycleProdInfo": {
            "208188426": [
                [208188426, 1440799449, '2025-09-01T22:03:51.17', 95.185, 95.728, ...],  // 24 elements
                ...
            ]
        }
    }
    """
    messages = []
    
    # Track JSON structure for error handling (not exposed in response)
    metadata["_json_structure"] = {
        "type": type(json_data).__name__,
        "keys": list(json_data.keys()) if isinstance(json_data, dict) else None
    }
    
    # Handle actual GWMReader structure: CycleProdInfo -> {machineId: [[...], ...]}
    if isinstance(json_data, dict) and "CycleProdInfo" in json_data:
        cycle_prod_info = json_data["CycleProdInfo"]
        if isinstance(cycle_prod_info, dict):
            # Flatten all arrays from all machine IDs
            for machine_id, records in cycle_prod_info.items():
                if isinstance(records, list):
                    # Each record is already an array of 24 elements
                    # Convert to dict format for easier processing
                    for record in records:
                        if isinstance(record, list) and len(record) >= 24:
                            # Convert array to dict with field names
                            msg_dict = {
                                "machineId": record[0] if len(record) > 0 else 0,
                                "segmentId": record[1] if len(record) > 1 else 0,
                                "Time": record[2] if len(record) > 2 else None,
                                "expectedElapsedTime": record[3] if len(record) > 3 else 0.0,
                                "actualElapsedTime": record[4] if len(record) > 4 else 0.0,
                                "pathEasting": record[5] if len(record) > 5 else 0.0,
                                "pathNorthing": record[6] if len(record) > 6 else 0.0,
                                "pathElevation": record[7] if len(record) > 7 else 0.0,
                                "plannedDistance": record[8] if len(record) > 8 else 0.0,
                                "expectedSpeed": record[9] if len(record) > 9 else 0.0,
                                "actualSpeed": record[10] if len(record) > 10 else 0.0,
                                "expectedDesiredSpeed": record[11] if len(record) > 11 else 0.0,
                                "actualDesiredSpeed": record[12] if len(record) > 12 else 0.0,
                                "leftWidth": record[13] if len(record) > 13 else 0.0,
                                "rightWidth": record[14] if len(record) > 14 else 0.0,
                                "pathBank": record[15] if len(record) > 15 else 0.0,
                                "pathHeading": record[16] if len(record) > 16 else 0.0,
                                "payloadPercent": record[17] if len(record) > 17 else 0,
                                "expectedSpeedSource": record[18] if len(record) > 18 else 0,
                                "expectedASLR": record[19] if len(record) > 19 else 0,
                                "expectedRegModEnum": record[20] if len(record) > 20 else 0,
                                "actualSpeedSource": record[21] if len(record) > 21 else 0,
                                "actualASLR": record[22] if len(record) > 22 else 0,
                                "actualRegModEnum": record[23] if len(record) > 23 else 0,
                            }
                            messages.append(msg_dict)
            
            metadata["_json_structure"]["found_in_key"] = "CycleProdInfo"
            metadata["_json_structure"]["structure_type"] = "CycleProdInfo -> {machineId: [[...24 elements...], ...]}"
            return messages
    
    # Fallback to old logic for other possible structures
    if isinstance(json_data, list):
        # Direct list of messages
        messages = json_data
    elif isinstance(json_data, dict):
        # Try common keys
        if "messages" in json_data:
            messages = json_data["messages"]
        elif "data" in json_data:
            if isinstance(json_data["data"], list):
                messages = json_data["data"]
            elif isinstance(json_data["data"], dict) and "messages" in json_data["data"]:
                messages = json_data["data"]["messages"]
        elif "records" in json_data:
            messages = json_data["records"]
        elif "telemetry" in json_data:
            messages = json_data["telemetry"]
        else:
            # Try to find any list value that might contain messages
            for key, value in json_data.items():
                if isinstance(value, list) and len(value) > 0:
                    # Check if first item looks like a message (has common fields)
                    if isinstance(value[0], dict):
                        first_item = value[0]
                        if any(field in first_item for field in ["segmentId", "pathEasting", "actualSpeed", "machineId", "segment_id", "path_easting"]):
                            messages = value
                            metadata["_json_structure"]["found_in_key"] = key
                            break
    
    metadata["total_messages"] = len(messages)
    
    return messages


def _convert_db_records_to_tuples(
    records: List[Dict[str, Any]], 
    raw_messages: List[List],
    metadata: Dict[str, Any]
) -> List[tuple]:
    """
    Convert database record dicts (from parse_gateway_messages.py) to tuple format.
    Uses raw_messages to get machineId, segmentId, Time (not present in records).
    """
    tuples = []
    errors = []
    
    # Ensure records and raw_messages counts match
    if len(records) != len(raw_messages):
        metadata["errors"].append(
            f"Mismatch: {len(records)} records but {len(raw_messages)} raw messages"
        )
        return []
    
    for idx, (record, raw_msg) in enumerate(zip(records, raw_messages)):
        try:
            tuple_data = _db_record_to_tuple(record, raw_msg)
            if tuple_data:
                tuples.append(tuple_data)
            else:
                errors.append(f"Record {idx}: Failed to convert")
        except Exception as e:
            errors.append(f"Record {idx}: {str(e)}")
    
    if errors:
        metadata["warnings"].extend(errors[:10])
    
    return tuples


def _convert_json_messages_to_tuples(
    messages: List[Dict[str, Any]], 
    metadata: Dict[str, Any]
) -> List[tuple]:
    """
    Convert JSON message dictionaries to tuple format expected by AMTCycleProdInfoMessage.
    
    Based on AMTCycleProdInfoMessage constructor, there are two formats:
    
    1. Format with 27 fields (from to_array() - deserialized):
       [0] start_time (timestamp)
       [1] segmentId
       [2] expectedElapsedTime
       [3] expectedTimeGPS
       [4] expectedTime (timestamp)
       [5] actualElapsedTime
       [6] actualTimeGPS
       [7] actualTime (timestamp)
       [8] pathEasting
       [9] pathNorthing
       [10] pathElevation
       [11] plannedDistance
       [12] expectedSpeed
       [13] actualSpeed
       [14] expectedDesiredSpeed
       [15] actualDesiredSpeed
       [16] leftWidth
       [17] rightWidth
       [18] pathBank
       [19] pathHeading
       [20] payloadPercent
       [21] expectedSpeedSource
       [22] expectedASLR
       [23] expectedRegModEnum
       [24] actualSpeedSource
       [25] actualASLR
       [26] actualRegModEnum
    
    2. Format from database (not 27 fields):
       [0] machineId
       [1] segmentId
       [2] start_time (datetime or timestamp)
       [3] expectedElapsedTime
       [4] actualElapsedTime
       [5] pathEasting
       [6] pathNorthing
       [7] pathElevation
       [8] plannedDistance
       [9] expectedSpeed
       [10] actualSpeed
       [11] expectedDesiredSpeed
       [12] actualDesiredSpeed
       [13] leftWidth
       [14] rightWidth
       [15] pathBank
       [16] pathHeading
       [17] payloadPercent
       [18] expectedSpeedSource
       [19] expectedASLR
       [20] expectedRegModEnum
       [21] actualSpeedSource
       [22] actualASLR
       [23] actualRegModEnum
       [24] cycleDistance (optional)
    """
    tuples = []
    errors = []
    
    for idx, msg in enumerate(messages):
        try:
            # Convert JSON message to tuple
            # Handle different field name variations
            tuple_data = _json_message_to_tuple(msg)
            if tuple_data:
                tuples.append(tuple_data)
            else:
                errors.append(f"Message {idx}: Failed to convert")
        except Exception as e:
            errors.append(f"Message {idx}: {str(e)}")
    
    if errors:
        metadata["warnings"].extend(errors[:10])  # Limit to first 10 errors
    
    return tuples


def _db_record_to_tuple(record: Dict[str, Any], raw_msg: List) -> Optional[tuple]:
    """
    Convert a database record dict (from parse_gateway_messages.py) to tuple format.
    Uses raw_msg to get machineId, segmentId, Time (not present in record dict).
    
    Returns tuple in format expected by AMTCycleProdInfoMessage (non-27-field format from database):
    [0] machineId
    [1] segmentId
    [2] start_time (datetime or timestamp)
    [3] expectedElapsedTime
    [4] actualElapsedTime
    ...
    [24] cycleDistance (optional, default 0.0)
    """
    try:
        # Get machineId, segmentId, Time from raw message (not present in record dict)
        # Raw message format: [machineId, segmentId, Time, ...]
        machine_id = raw_msg[0] if len(raw_msg) > 0 else 0
        segment_id = raw_msg[1] if len(raw_msg) > 1 else 0
        time_str = raw_msg[2] if len(raw_msg) > 2 else None
        
        # Parse start_time
        if time_str:
            if isinstance(time_str, str):
                try:
                    start_time = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                    if start_time.tzinfo is None:
                        start_time = start_time.replace(tzinfo=timezone.utc)
                    start_time_str = start_time.isoformat()
                except:
                    # Calculate from segmentId if parse fails
                    start_time_gps = segment_id
                    start_time = gps_epoch + timedelta(seconds=start_time_gps) - leap_seconds
                    start_time_str = start_time.isoformat()
            else:
                start_time_gps = segment_id
                start_time = gps_epoch + timedelta(seconds=start_time_gps) - leap_seconds
                start_time_str = start_time.isoformat()
        else:
            # Calculate from segmentId
            start_time_gps = segment_id
            start_time = gps_epoch + timedelta(seconds=start_time_gps) - leap_seconds
            start_time_str = start_time.isoformat()
        
        # Speed values from GWMReader are in m/s, convert to km/h (* 3.6)
        tuple_data = (
            int(machine_id),                                       # 0: machineId
            int(segment_id),                                       # 1: segmentId
            start_time_str,                                        # 2: start_time (ISO string)
            float(record.get("expectedElapsedTime", 0.0)),      # 3: expectedElapsedTime
            float(record.get("actualElapsedTime", 0.0)),          # 4: actualElapsedTime
            float(record.get("pathEasting", 0.0)),               # 5: pathEasting
            float(record.get("pathNorthing", 0.0)),              # 6: pathNorthing
            float(record.get("pathElevation", 0.0)),             # 7: pathElevation
            float(record.get("plannedDistance", 0.0)),            # 8: plannedDistance
            float(record.get("expectedSpeed", 0.0)) * 3.6,       # 9: expectedSpeed (m/s -> km/h)
            float(record.get("actualSpeed", 0.0)) * 3.6,          # 10: actualSpeed (m/s -> km/h)
            float(record.get("expectedDesiredSpeed", 0.0)) * 3.6, # 11: expectedDesiredSpeed (m/s -> km/h)
            float(record.get("actualDesiredSpeed", 0.0)) * 3.6,   # 12: actualDesiredSpeed (m/s -> km/h)
            float(record.get("leftWidth", 0.0)),                  # 13: leftWidth
            float(record.get("rightWidth", 0.0)),                 # 14: rightWidth
            float(record.get("pathBank", 0.0)),                   # 15: pathBank
            float(record.get("pathHeading", 0.0)),                # 16: pathHeading
            int(record.get("payloadPercent", 0)),                 # 17: payloadPercent
            int(record.get("expectedSpeedSource", 0)),            # 18: expectedSpeedSource
            int(record.get("expectedASLR", 0)),                   # 19: expectedASLR
            int(record.get("expectedRegModEnum", 0)),            # 20: expectedRegModEnum
            int(record.get("actualSpeedSource", 0)),              # 21: actualSpeedSource
            int(record.get("actualASLR", 0)),                     # 22: actualASLR
            int(record.get("actualRegModEnum", 0)),              # 23: actualRegModEnum
            float(record.get("cycleDistance", 0.0)),               # 24: cycleDistance
        )
        
        if len(tuple_data) != 25:
            raise ValueError(f"Invalid tuple length: {len(tuple_data)}, expected 25")
        
        return tuple_data
    except Exception as e:
        return None


def _json_message_to_tuple(msg: Dict[str, Any]) -> Optional[tuple]:
    """
    Convert a single JSON message dict to tuple format.
    
    Returns tuple in format expected by AMTCycleProdInfoMessage (non-27-field format from database):
    [0] machineId
    [1] segmentId
    [2] start_time (datetime or timestamp)
    [3] expectedElapsedTime
    [4] actualElapsedTime
    [5] pathEasting
    [6] pathNorthing
    [7] pathElevation
    [8] plannedDistance
    [9] expectedSpeed
    [10] actualSpeed
    [11] expectedDesiredSpeed
    [12] actualDesiredSpeed
    [13] leftWidth
    [14] rightWidth
    [15] pathBank
    [16] pathHeading
    [17] payloadPercent
    [18] expectedSpeedSource
    [19] expectedASLR
    [20] expectedRegModEnum
    [21] actualSpeedSource
    [22] actualASLR
    [23] actualRegModEnum
    [24] cycleDistance (optional, default 0.0)
    """
    try:
        # Helper function to get value with fallback
        def get_value(key_variations, default=0.0):
            for key in key_variations:
                if key in msg:
                    val = msg[key]
                    # Handle None values
                    if val is None:
                        return default
                    return val
            return default
        
        # Field mapping is now verified based on actual JSON structure from GWMReader
        
        # Extract machineId (required for database format)
        machine_id = get_value([
            "machineId", "machine_id", "Machine Unique Id", "MachineId", 
            "machine_unique_id", "MachineUniqueId", "machineUniqueId"
        ], 0)
        
        # Extract segmentId (GPS timestamp)
        segment_id = get_value([
            "segmentId", "segment_id", "SegmentId", "gps_time", "gpsTime",
            "segment_id", "GPS_Time", "GPS_TIME"
        ], 0)
        
        # Extract segmentId (GPS timestamp)
        segment_id = get_value(["segmentId", "segment_id", "SegmentId", "gps_time", "gpsTime"], 0)
        
        # Extract elapsed times (already in seconds from GWMReader JSON)
        expected_elapsed = get_value([
            "expectedElapsedTime", "expected_elapsed_time", "ExpectedElapsedTime",
            "expectedElapsed", "expected_elapsed"
        ], 0.0)
        actual_elapsed = get_value([
            "actualElapsedTime", "actual_elapsed_time", "ActualElapsedTime",
            "actualElapsed", "actual_elapsed"
        ], 0.0)
        
        # Values are already in seconds (not milliseconds) from GWMReader
        expected_elapsed = float(expected_elapsed)
        actual_elapsed = float(actual_elapsed)
        
        # Get start_time
        start_time_val = get_value([
            "start_time", "startTime", "StartTime", "Time", "time",
            "ArrivalTIme", "ArrivalTime", "arrival_time"
        ], None)
        
        if start_time_val:
            if isinstance(start_time_val, str):
                try:
                    start_time = datetime.fromisoformat(start_time_val.replace('Z', '+00:00'))
                    if start_time.tzinfo is None:
                        start_time = start_time.replace(tzinfo=timezone.utc)
                except:
                    # Calculate from segmentId if parse fails
                    start_time_gps = segment_id
                    start_time = gps_epoch + timedelta(seconds=start_time_gps) - leap_seconds
            elif isinstance(start_time_val, (int, float)):
                # Could be timestamp or GPS time
                if start_time_val > 1e10:  # Likely timestamp
                    start_time = datetime.fromtimestamp(start_time_val, tz=timezone.utc)
                else:  # Likely GPS time
                    start_time = gps_epoch + timedelta(seconds=start_time_val) - leap_seconds
            else:
                start_time_gps = segment_id
                start_time = gps_epoch + timedelta(seconds=start_time_gps) - leap_seconds
        else:
            # Calculate from segmentId
            start_time_gps = segment_id
            start_time = gps_epoch + timedelta(seconds=start_time_gps) - leap_seconds
        
        # Extract coordinates (already in meters from GWMReader JSON)
        path_easting = get_value([
            "pathEasting", "path_easting", "easting", "x", "X", "PathEasting"
        ], 0.0)
        path_northing = get_value([
            "pathNorthing", "path_northing", "northing", "y", "Y", "PathNorthing"
        ], 0.0)
        path_elevation = get_value([
            "pathElevation", "path_elevation", "elevation", "z", "Z", "PathElevation"
        ], 0.0)
        
        # Values are already in meters (not millimeters) from GWMReader
        path_easting = float(path_easting)
        path_northing = float(path_northing)
        path_elevation = float(path_elevation)
        
        # Extract planned distance (already in meters)
        planned_distance = get_value([
            "plannedDistance", "planned_distance", "PlannedDistance"
        ], 0.0)
        planned_distance = float(planned_distance)
        
        # Speed values from GWMReader are in m/s, convert to km/h
        expected_speed = get_value([
            "expectedSpeed", "expected_speed", "ExpectedSpeed"
        ], 0.0)
        expected_speed = float(expected_speed) * 3.6
        actual_speed = get_value([
            "actualSpeed", "actual_speed", "ActualSpeed"
        ], 0.0)
        actual_speed = float(actual_speed) * 3.6
        expected_desired_speed = get_value([
            "expectedDesiredSpeed", "expected_desired_speed", "ExpectedDesiredSpeed"
        ], 0.0)
        expected_desired_speed = float(expected_desired_speed) * 3.6
        actual_desired_speed = get_value([
            "actualDesiredSpeed", "actual_desired_speed", "ActualDesiredSpeed"
        ], 0.0)
        actual_desired_speed = float(actual_desired_speed) * 3.6
        
        # Widths are already in meters from GWMReader JSON
        left_width = get_value([
            "leftWidth", "left_width", "LeftWidth"
        ], 0.0)
        right_width = get_value([
            "rightWidth", "right_width", "RightWidth"
        ], 0.0)
        left_width = float(left_width)
        right_width = float(right_width)
        
        path_bank = get_value([
            "pathBank", "path_bank", "PathBank", "bank"
        ], 0.0)
        path_heading = get_value([
            "pathHeading", "path_heading", "PathHeading", "heading"
        ], 0.0)
        
        payload_percent = get_value([
            "payloadPercent", "payload_percent", "PayloadPercent", "payload"
        ], 0)
        # Handle payload > 200 (database uses 255 for unknown)
        if payload_percent > 200:
            payload_percent = payload_percent - 255
        
        expected_speed_source = get_value([
            "expectedSpeedSource", "expected_speed_source", "ExpectedSpeedSource"
        ], 0)
        expected_aslr = get_value([
            "expectedASLR", "expected_aslr", "ExpectedASLR"
        ], 0)
        expected_reg_mod_enum = get_value([
            "expectedRegModEnum", "expected_reg_mod_enum", "ExpectedRegModEnum"
        ], 0)
        actual_speed_source = get_value([
            "actualSpeedSource", "actual_speed_source", "ActualSpeedSource"
        ], 0)
        actual_aslr = get_value([
            "actualASLR", "actual_aslr", "ActualASLR"
        ], 0)
        actual_reg_mod_enum = get_value([
            "actualRegModEnum", "actual_reg_mod_enum", "ActualRegModEnum"
        ], 0)
        
        cycle_distance = get_value([
            "cycleDistance", "cycle_distance", "CycleDistance"
        ], 0.0)
        
        # Create tuple in format expected by AMTCycleProdInfoMessage (non-27-field format from database)
        # CRITICAL: Order must match AMTCycleProdInfoMessage.__init__ (else branch, lines 56-87)
        # Format: [machineId, segmentId, start_time, expectedElapsedTime, actualElapsedTime, ...]
        # 
        # Expected order (from AMTCycleProdInfoMessage constructor):
        # [0] machineId -> data[0]
        # [1] segmentId -> data[1]  
        # [2] start_time -> data[2]
        # [3] expectedElapsedTime -> data[3]
        # [4] actualElapsedTime -> data[4]
        # [5] pathEasting -> data[5]
        # [6] pathNorthing -> data[6]
        # [7] pathElevation -> data[7]
        # [8] plannedDistance -> data[8]
        # [9] expectedSpeed -> data[9]
        # [10] actualSpeed -> data[10]
        # [11] expectedDesiredSpeed -> data[11]
        # [12] actualDesiredSpeed -> data[12]
        # [13] leftWidth -> data[13]
        # [14] rightWidth -> data[14]
        # [15] pathBank -> data[15]
        # [16] pathHeading -> data[16]
        # [17] payloadPercent -> data[17]
        # [18] expectedSpeedSource -> data[18]
        # [19] expectedASLR -> data[19]
        # [20] expectedRegModEnum -> data[20]
        # [21] actualSpeedSource -> data[21]
        # [22] actualASLR -> data[22]
        # [23] actualRegModEnum -> data[23]
        # [24] cycleDistance -> data[24] (optional)
        
        # Convert start_time to string (ISO format) for AMTCycleProdInfoMessage constructor
        # Constructor expects string or timestamp, not datetime object
        if isinstance(start_time, datetime):
            start_time_str = start_time.isoformat()
        elif isinstance(start_time, str):
            start_time_str = start_time
        else:
            # Fallback: convert to ISO string from timestamp
            start_time_str = datetime.fromtimestamp(float(start_time), tz=timezone.utc).isoformat()
        
        tuple_data = (
            int(machine_id),                                       # 0: machineId
            int(segment_id),                                       # 1: segmentId
            start_time_str,                                        # 2: start_time (ISO string)
            float(expected_elapsed),                              # 3: expectedElapsedTime
            float(actual_elapsed),                                 # 4: actualElapsedTime
            float(path_easting),                                   # 5: pathEasting
            float(path_northing),                                  # 6: pathNorthing
            float(path_elevation),                                 # 7: pathElevation
            float(planned_distance),                               # 8: plannedDistance
            float(expected_speed),                                 # 9: expectedSpeed
            float(actual_speed),                                   # 10: actualSpeed
            float(expected_desired_speed),                         # 11: expectedDesiredSpeed
            float(actual_desired_speed),                           # 12: actualDesiredSpeed
            float(left_width),                                     # 13: leftWidth
            float(right_width),                                    # 14: rightWidth
            float(path_bank),                                      # 15: pathBank
            float(path_heading),                                   # 16: pathHeading
            int(payload_percent),                                  # 17: payloadPercent
            int(expected_speed_source),                            # 18: expectedSpeedSource
            int(expected_aslr),                                    # 19: expectedASLR
            int(expected_reg_mod_enum),                            # 20: expectedRegModEnum
            int(actual_speed_source),                              # 21: actualSpeedSource
            int(actual_aslr),                                      # 22: actualASLR
            int(actual_reg_mod_enum),                             # 23: actualRegModEnum
            float(cycle_distance),                                 # 24: cycleDistance
        )
        
        # Validate tuple length (should be 25 for non-27-field format)
        # Note: JSON from GWMReader has 24 elements, but we add cycleDistance to make 25
        if len(tuple_data) != 25:
            raise ValueError(f"Invalid tuple length: {len(tuple_data)}, expected 25")
        
        return tuple_data
    except Exception as e:
        return None


def get_objects_summary(cycles: Optional[List[Cycle]], zones: Optional[List[Zone]]) -> Dict[str, Any]:
    """
    Get summary information about converted objects.
    
    Args:
        cycles: List of Cycle objects
        zones: List of Zone objects
    
    Returns:
        Dictionary with summary information
    """
    summary = {
        "cycles_count": 0,
        "zones_count": 0,
        "total_messages": 0,
        "cycles_info": [],
        "zones_info": []
    }
    
    if cycles:
        summary["cycles_count"] = len(cycles)
        for cycle in cycles:
            summary["total_messages"] += len(cycle.messages) if cycle.messages else 0
            summary["cycles_info"].append({
                "cycleId": cycle.cycleId,
                "machineId": cycle.machineId,
                "segments_count": len(cycle.segments) if cycle.segments else 0,
                "messages_count": len(cycle.messages) if cycle.messages else 0,
                "loss": cycle.loss if hasattr(cycle, 'loss') else None,
                "efficiency": cycle.efficiency if hasattr(cycle, 'efficiency') else None
            })
    
    if zones:
        summary["zones_count"] = len(zones)
        for zone in zones:
            summary["zones_info"].append({
                "zoneType": zone.zoneType.name if hasattr(zone.zoneType, 'name') else str(zone.zoneType),
                "points_count": len(zone.points) if zone.points else 0,
                "centroid": zone.centroid if hasattr(zone, 'centroid') else None
            })
    
    return summary


def extract_zones_from_import(
    parser_output: Dict[str, Any],
) -> Tuple[List, List]:
    """
    Extract Cycles and Zones from imported data using Reader.py standard algorithms.

    Uses parse_cp1_data() per machine to classify segments (Spotting, Travelling)
    and DBSCAN clustering (createLoadDumpAreas) for zone detection.

    Args:
        parser_output: Raw parser output from GWMReader

    Returns:
        Tuple of (all_cycles, all_zones) — Cycle and Zone objects from Reader.py
    """
    records_by_machine, is_cp2 = extract_cp_records(parser_output)

    if not records_by_machine:
        return [], []

    all_cycles = []
    all_zones = []

    for machine_key, raw_messages in records_by_machine.items():
        if not raw_messages:
            continue

        # Build 25-element tuples for this machine
        machine_tuples = []
        for raw_msg in raw_messages:
            if len(raw_msg) < 24:
                continue
            record = parse_message_to_dict(raw_msg)
            t = _db_record_to_tuple(record, raw_msg)
            if t is not None:
                machine_tuples.append(t)

        if not machine_tuples:
            continue

        # Determine machine_id from the first tuple
        machine_id = machine_tuples[0][0]
        machine_info = {
            "Name": f"Machine_{machine_id}",
            "TypeName": "Unknown",
        }

        # Sort tuples by segmentId then actualElapsedTime for correct segment grouping
        machine_tuples.sort(key=lambda x: (x[1], x[4]))

        parse_fn = AMTCycleProdInfoReader.parse_cp2_data if is_cp2 else AMTCycleProdInfoReader.parse_cp1_data
        result = parse_fn(machine_tuples, machine_info)

        if result and result[0]:
            all_cycles.extend(result[0])
        if result and result[1]:
            all_zones.extend(result[1])

    return all_cycles, all_zones


def convert_imported_records_to_telemetry(
    parser_output: Dict[str, Any],
    records: List[Dict[str, Any]],
    sample_interval: int = 5  # Deprecated: no longer used, kept for backward compatibility
) -> List[Tuple]:
    """
    Convert imported records (dicts) to telemetry tuple format for process_site.

    Args:
        parser_output: Raw parser output containing machineId, segmentId in arrays
        records: List of processed records (dicts) from process_parser_output
        sample_interval: Deprecated - no longer used. Interval is now calculated from
                        actualElapsedTime in the record.

    Returns:
        List of tuples in format:
        (machine_id, segment_id, cycle_id, interval, pathEasting, pathNorthing,
         pathElevation, expectedSpeed, actualSpeed, pathBank, pathHeading,
         leftWidth, rightWidth, payloadPercent)
    """
    # Extract raw messages to get machineId, segmentId
    records_by_ip, _ = extract_cp_records(parser_output)
    
    # Create mapping from record index to machineId, segmentId
    raw_messages = []
    for ip_address, messages in records_by_ip.items():
        for msg in messages:
            if len(msg) >= 24:
                raw_messages.append({
                    "machineId": int(msg[0]) if msg[0] is not None else 0,
                    "segmentId": int(msg[1]) if msg[1] is not None else 0,
                })
    
    # Ensure records and raw_messages have same length
    if len(records) != len(raw_messages):
        # If mismatch, try to match by creating minimal raw_messages
        if len(raw_messages) < len(records):
            # Extend with default values
            while len(raw_messages) < len(records):
                raw_messages.append({"machineId": 0, "segmentId": 0})
        else:
            # Truncate to match
            raw_messages = raw_messages[:len(records)]
    
    # Convert to telemetry tuples
    telemetry_tuples = []

    for i, record in enumerate(records):
        raw_msg = raw_messages[i] if i < len(raw_messages) else {"machineId": 0, "segmentId": 0}
        
        machine_id = raw_msg.get("machineId", 0)
        segment_id = raw_msg.get("segmentId", 0)
        
        # Use segmentId as cycle_id (common pattern in telemetry data)
        cycle_id = segment_id

        # Use actualElapsedTime from parser output (in seconds) and convert to milliseconds
        # This ensures correct time calculation in event generation which expects ms
        actual_elapsed_sec = float(record.get("actualElapsedTime", 0.0))
        interval = int(actual_elapsed_sec * 1000)  # Convert seconds to milliseconds
        
        # Extract fields from record dict
        path_easting = float(record.get("pathEasting", 0.0))
        path_northing = float(record.get("pathNorthing", 0.0))
        path_elevation = float(record.get("pathElevation", 0.0))
        # Speed values from GWMReader are in m/s, convert to km/h
        expected_speed = float(record.get("expectedSpeed", 0.0)) * 3.6
        actual_speed = float(record.get("actualSpeed", 0.0)) * 3.6
        path_bank = float(record.get("pathBank", 0.0))
        path_heading = float(record.get("pathHeading", 0.0))
        left_width = float(record.get("leftWidth", 0.0))
        right_width = float(record.get("rightWidth", 0.0))
        payload_percent = int(record.get("payloadPercent", 0))
        
        # Handle payload > 200
        if payload_percent > 200:
            payload_percent = payload_percent - 255
        
        # Create tuple in format expected by process_site
        telemetry_tuple = (
            machine_id,        # 0: machine_id
            segment_id,        # 1: segment_id
            cycle_id,          # 2: cycle_id
            interval,          # 3: interval
            path_easting,      # 4: pathEasting
            path_northing,     # 5: pathNorthing
            path_elevation,    # 6: pathElevation
            expected_speed,    # 7: expectedSpeed
            actual_speed,      # 8: actualSpeed
            path_bank,         # 9: pathBank
            path_heading,      # 10: pathHeading
            left_width,        # 11: leftWidth
            right_width,       # 12: rightWidth
            payload_percent,   # 13: payloadPercent
        )
        
        telemetry_tuples.append(telemetry_tuple)
    
    # Sort by machine_id, cycle_id, segment_id, interval
    telemetry_tuples.sort(key=lambda x: (x[0], x[2], x[1], x[3]))
    
    return telemetry_tuples
