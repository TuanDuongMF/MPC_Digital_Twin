"""Stage 6: Fleet Configuration - replicates Fleet_Configuration from C++.

Processes machine CSVs, creates machines.json, assigns machines to zones
in model.json. Adds haulers, loaders, processors, FuelMachine, Layover
arrays with full C++ field set.
"""

import csv
import json
import os

from .geometry import distance_2d


# Machine name -> machine_id mapping (from machineDataSet.json)
DEFAULT_MACHINE_MAP = {
    "793F": 1,
    "794AC": 2,
    "798": 3,
    "6060FS": 4,
}


def _read_csv_rows(filepath):
    """Read CSV file into list of dicts."""
    rows = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except FileNotFoundError:
        pass
    return rows


def _find_nearest_node(x, y, z, nodes, max_dist=100):
    """Find nearest node within max_dist, return node id or 0.

    Replicates distance() function from C++ Fleet_Configuration.
    """
    best_id = 0
    best_dist = float("inf")
    for node in nodes:
        coords = node.get("coords", [0, 0, 0])
        # coords are [y, x, z]
        nx, ny = coords[1], coords[0]
        dist = distance_2d(x, y, nx, ny)
        if dist < best_dist and dist < max_dist:
            best_dist = dist
            best_id = node["id"]
    return best_id


def _find_road_for_node(node_id, roads):
    """Find road containing a node, return road id or 0."""
    for road in roads:
        if node_id in road.get("nodes", []):
            return road["id"]
    return 0


def _get_machine_id(class_name):
    """Map machine class name to machine_id.

    C++: "793" -> 1, "794" -> 2, "798" -> 3.
    """
    class_upper = str(class_name).upper().replace(" ", "").replace("-", "")
    if "793" in class_upper:
        return 1
    if "794" in class_upper:
        return 2
    if "798" in class_upper:
        return 3
    return 1  # default


def _is_truthy(val):
    """Check if value represents True."""
    return str(val).strip().lower() in ("1", "true", "yes")


def _safe_int(val):
    """Convert to int safely (C++ stoi equivalent).

    Handles Python bool strings "True"/"False" from pyodbc.
    """
    if val is None or val == "" or val == "None":
        return 0
    s = str(val).strip().lower()
    if s in ("true", "yes"):
        return 1
    if s in ("false", "no"):
        return 0
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return 0


def _safe_float(val):
    """Convert to float safely (C++ stof equivalent)."""
    if val is None or val == "" or val == "None":
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def _read_destinations(output_path):
    """Read Destinations.csv into list of dicts."""
    return _read_csv_rows(os.path.join(output_path, "Destinations.csv"))


def _find_destination_name(dest_oid, destinations):
    """Match LOCATION_OID to get destination NAME from Destinations.csv.

    Replicates C++ destination matching logic.
    """
    if not dest_oid:
        return ""
    try:
        dest_val = int(dest_oid)
    except (ValueError, TypeError):
        dest_val = dest_oid

    for dest in destinations:
        loc_oid = str(dest.get("LOCATION_OID", "")).strip()
        try:
            loc_val = int(loc_oid) if loc_oid else 0
        except ValueError:
            loc_val = loc_oid
        if loc_val == dest_val:
            return dest.get("NAME", "")
    return ""


def _build_machine_entry(row, cls_info, pos, comms):
    """Build the full machine data dict from CSV columns.

    Contains all fields that C++ copies into hauler/loader/processor objects.
    """
    # C++ type conversions: strtoll for OIDs, stoi for int fields, stof for floats
    moid_str = str(row.get("MACHINE_OID", "")).strip()
    try:
        moid_int = int(moid_str) if moid_str else 0
    except ValueError:
        moid_int = 0

    eid_str = str(row.get("ENDPOINTID", "")).strip()
    try:
        eid_int = int(eid_str) if eid_str else 0
    except ValueError:
        eid_int = 0

    return {
        "MachineName": str(row.get("NAME", "")).strip(),
        "MachineOID": moid_int,  # C++ strtoll
        "SerialNumber": str(row.get("SERIALNUMBER", "")).strip(),
        "MachineClassName": cls_info.get("name", ""),
        "MachineClassType": cls_info.get("ecf_class_id", ""),
        "ClassOID": str(row.get("CLASS", "")).strip(),
        "MachineClassOID": str(row.get("CLASS", "")).strip(),
        "Type": str(row.get("ECF_CLASS_ID", "")).strip(),
        "IsActive": _safe_int(row.get("IS_ACTIVE", "0")),  # C++ stoi
        "IgnoreForAssignment": _safe_int(row.get("IGNOREFORASSIGNMENT", "0")),  # C++ stoi
        "Productivity": _safe_float(row.get("PRODUCTIVITY", "")),  # C++ stof
        "FuelCapacity": _safe_float(row.get("FUELCAPACITY", "")),  # C++ stof
        "TKPHRate": _safe_float(row.get("TKPHRATE", "")),  # C++ stof
        "MaxPayload": _safe_float(row.get("MAXPAYLOAD", "")),  # C++ stof
        "NominalPayload": _safe_float(row.get("NOMINALPAYLOAD", "")),  # C++ stof
        "DefaultMaterial": row.get("DEFAULTMATERIAL", ""),
        "Material": row.get("MATERIAL", ""),
        "EndpointID": eid_int,  # C++ strtoll
        "ExternalRef": row.get("EXTERNALREF", ""),
        "ExternalDesc": row.get("EXTERNALDESC", ""),
        "IsCrusher": _safe_int(str(row.get("IS_CRUSHER", "0")).strip()),  # C++ stoi
        "LoadPlanOID": str(row.get("LOAD_PLAN_OID", "")).strip(),
        "ID": _safe_int(row.get("ID", "")),  # C++ stoi
        "x": pos.get("x", 0),
        "y": pos.get("y", 0),
        "z": pos.get("z", 0),
        "ip_address": comms.get("ip", ""),
        "port": comms.get("port", ""),
        "destination_oid": pos.get("destination_oid"),  # C++ default nullptr
        "waypoint_oid": pos.get("waypoint_oid"),  # C++ default nullptr
    }


def _remove_zones(model):
    """Remove empty zones from model.json.

    Replicates remove_zones() from C++:
    - load_zones: remove if machineOID is empty string
    - dump_zones: remove if machineOID is empty string or not in processors
    - chargers: remove if machineOID is empty string
    """
    # Collect processor machineOIDs
    processor_oids = set()
    for proc in model.get("processors", []):
        moid = proc.get("MachineOID", "")
        if moid:
            processor_oids.add(moid)
            try:
                processor_oids.add(int(moid))
            except (ValueError, TypeError):
                pass

    # Remove empty load_zones
    model["load_zones"] = [
        lz for lz in model.get("load_zones", [])
        if lz.get("machineOID") != ""
    ]

    # Remove empty dump_zones (empty string OR machineOID not in processors)
    filtered_dz = []
    for dz in model.get("dump_zones", []):
        moid = dz.get("machineOID", "")
        if isinstance(moid, str) and moid == "":
            continue
        if isinstance(moid, int) and moid not in processor_oids:
            continue
        filtered_dz.append(dz)
    model["dump_zones"] = filtered_dz

    # Remove empty chargers
    model["chargers"] = [
        ch for ch in model.get("chargers", [])
        if ch.get("machineOID") != ""
    ]


def configure_fleet(output_path, template_path):
    """Process machine data, create machines.json, update model.json.

    Replicates Fleet_Configuration::assignMachine() from C++.
    """
    print("[Stage 6] Configuring fleet...")

    # Load machine dataset mapping
    machine_dataset_path = os.path.join(template_path, "machineDataSet.json")
    machine_map = dict(DEFAULT_MACHINE_MAP)
    if os.path.exists(machine_dataset_path):
        with open(machine_dataset_path, "r", encoding="utf-8") as f:
            ds = json.load(f)
        for entry in ds.get("machine_map", []):
            mid = entry.get("machineId") or entry.get("mahcineId")
            if mid is not None:
                machine_map[entry["machineName"]] = mid

    # Read CSVs
    machines_csv = _read_csv_rows(os.path.join(output_path, "Machine.csv"))
    machine_class_csv = _read_csv_rows(os.path.join(output_path, "Machine_Class.csv"))
    machine_pos_csv = _read_csv_rows(os.path.join(output_path, "Machine_POS.csv"))
    machine_comms_csv = _read_csv_rows(os.path.join(output_path, "MachineComms.csv"))
    destinations = _read_destinations(output_path)

    # Build lookup maps
    class_map = {}
    for row in machine_class_csv:
        oid = str(row.get("MACHINECLASS_OID", "")).strip()
        if oid:
            class_map[oid] = {
                "name": row.get("NAME", ""),
                "ecf_class_id": row.get("ECF_CLASS_ID", ""),
            }

    pos_map = {}
    for row in machine_pos_csv:
        moid = str(row.get("MACHINE_OID", "")).strip()
        if moid:
            # C++ strtoll for destinationOID/waypointOID, default nullptr
            dest_str = str(row.get("DESTINATION_OID", "")).strip()
            wp_str = str(row.get("WAYPOINT_OID", "")).strip()
            try:
                dest_val = int(dest_str) if dest_str else None
            except ValueError:
                dest_val = None
            try:
                wp_val = int(wp_str) if wp_str else None
            except ValueError:
                wp_val = None
            # C++ only sets if both row[1] and row[2] are non-empty
            if dest_str and wp_str:
                d_oid = dest_val
                w_oid = wp_val
            else:
                d_oid = None
                w_oid = None
            pos_map[moid] = {
                "x": float(row.get("X", 0) or 0),
                "y": float(row.get("Y", 0) or 0),
                "z": float(row.get("Z", 0) or 0),
                "destination_oid": d_oid,
                "waypoint_oid": w_oid,
            }

    comms_map = {}
    for row in machine_comms_csv:
        oid = str(row.get("OID", "")).strip()
        if oid:
            comms_map[oid] = {
                "ip": row.get("ip", ""),
                "port": row.get("port", ""),
            }

    # Read model.json
    model_path = os.path.join(output_path, "model.json")
    with open(model_path, "r", encoding="utf-8") as f:
        model = json.load(f)

    nodes = model.get("nodes", [])
    roads = model.get("roads", [])

    # Find first service station with non-null destinationOID for default service_zone
    default_service_zone_id = 0
    for ss in model.get("service_stations", []):
        dest = ss.get("destinationOID", "")
        if dest and dest != "":
            default_service_zone_id = ss.get("id", 0)
            break

    # Process each machine
    haulers_out = []
    loaders_out = []
    processors_out = []
    fuel_machines_out = []
    layover_out = []
    all_machines = []

    hauler_count = 0
    loader_count = 0
    processor_count = 0

    for row in machines_csv:
        is_active = row.get("IS_ACTIVE", "0")
        ignore = row.get("IGNOREFORASSIGNMENT", "0")
        if not _is_truthy(is_active) or _is_truthy(ignore):
            continue

        machine_oid = str(row.get("MACHINE_OID", "")).strip()
        class_oid = str(row.get("CLASS", "")).strip()
        load_plan_oid = str(row.get("LOAD_PLAN_OID", "")).strip()

        cls_info = class_map.get(class_oid, {})
        class_name = cls_info.get("name", "")
        pos = pos_map.get(machine_oid, {"x": 0, "y": 0, "z": 0})
        comms = comms_map.get(machine_oid, {})

        entry = _build_machine_entry(row, cls_info, pos, comms)
        all_machines.append(entry)

        # C++ uses row[1] ("Type") which maps to ECF_CLASS_ID in Machine.csv
        # e.g. "XAEntity.MachineClass.MobileClass.TruckClass"
        # Machine.csv has no TYPE column header; ECF_CLASS_ID is the actual field
        ecf_class_id = str(row.get("ECF_CLASS_ID", "")).strip()
        # Store as Type in entry (matches C++ machine_json["Type"] = row[1])
        entry["Type"] = ecf_class_id

        # ── Classify machine type (C++: machineType.find("Truck") etc.) ──
        is_truck = "Truck" in ecf_class_id
        is_loader_type = "Load" in ecf_class_id
        is_processor = "Process" in ecf_class_id
        is_fuel = "Fuel" in ecf_class_id
        is_layover = "Layover" in ecf_class_id

        if is_truck:
            hauler_count += 1
            mid = _get_machine_id(class_name)

            # C++: spawn_in_road = false, so road_id/node_id always nullptr
            # service_zone_id/service_zone_spot_id set conditionally
            init_cond = {
                "route_id": None,
                "road_id": None,
                "node_id": None,
                "service_zone_id": default_service_zone_id if default_service_zone_id else None,
                "service_zone_spot_id": 1 if default_service_zone_id else None,
            }

            hauler = {
                "id": hauler_count,
                "MachineOID": entry["MachineOID"],
                "group_id": hauler_count,
                "key": "haulers",
                "name": entry["MachineName"],
                "lane": 1,
                "Type": entry["Type"],
                "IsActive": entry["IsActive"],
                "SerialNumber": entry["SerialNumber"],
                "ip_address": entry["ip_address"],
                "ID": entry["ID"],
                "MachineClassOID": entry["MachineClassOID"],
                "IgnoreForAssignment": entry["IgnoreForAssignment"],
                "Productivity": entry["Productivity"],
                "FuelCapacity": entry["FuelCapacity"],
                "TKPHRate": entry["TKPHRate"],
                "MaxPayload": entry["MaxPayload"],
                "NominalPayload": entry["NominalPayload"],
                "DefaultMaterial": entry["DefaultMaterial"],
                "Material": entry["Material"],
                "EndpointID": entry["EndpointID"],
                "ExternalRef": entry["ExternalRef"],
                "ExternalDesc": entry["ExternalDesc"],
                "IsCrusher": entry["IsCrusher"],
                "MachineClassType": entry["MachineClassType"],
                "MachineClassName": entry["MachineClassName"],
                "machine_id": mid,
                "number_of_haulers": 1,
                "initial_position": 2,
                "initial_conditions": init_cond,
                "model_scale": 1,
                "geometry_name": "_default",
                "is_local_machine": None,
                "fuel": 0.9,
                "fuel_tank": "750 gal",
                "is_deactivate": False,
            }
            haulers_out.append(hauler)

        elif is_loader_type and load_plan_oid:
            loader_count += 1
            inload = False

            # C++ loop 1: match planOID, assign machineOID, set load_zone_id
            # C++ does NOT break - iterates all zones (can overwrite)
            lz_id = 0
            for lz in model.get("load_zones", []):
                plan_oid_val = lz.get("planOID", 0)
                try:
                    lp_int = int(load_plan_oid) if load_plan_oid else 0
                except ValueError:
                    lp_int = 0

                if plan_oid_val == lp_int:
                    inload = True
                    lz["machineOID"] = entry["MachineOID"]
                    lz_id = lz.get("id", 0)

            # C++ loop 2: set waypoint/destination on matching zones AND loader
            loader_wp_oid = None
            loader_dest_oid = None
            for lz in model.get("load_zones", []):
                if lz.get("machineOID") == entry["MachineOID"]:
                    wp_oid = pos.get("waypoint_oid", "")
                    dest_oid = pos.get("destination_oid", "")
                    lz["waypointOID"] = wp_oid
                    lz["destinationOID"] = dest_oid
                    lz["destinationName"] = _find_destination_name(
                        dest_oid, destinations
                    )
                    # C++ also sets on temp_json (loader object)
                    loader_wp_oid = wp_oid
                    loader_dest_oid = dest_oid

            # C++ builds loader object always, but only appends if inload
            loader = {
                "id": loader_count,
                "MachineOID": entry["MachineOID"],
                "name": entry["MachineName"],
                "key": "loaders",
                "machine_id": 4,
                "configured": "6060 FS (ID: 1496)",
                "fill_factor_pct": 1,
                "initial_charge_fuel_levels_pct": None,
                "is_deactivate": False,
                "initial_conditions": {
                    "load_zone_id": lz_id,
                    "assigned_load_spots": [1],
                },
                "MachineClassType": entry["MachineClassType"],
                "MachineClassName": entry["MachineClassName"],
                "Type": entry["Type"],
                "IsActive": entry["IsActive"],
                "SerialNumber": entry["SerialNumber"],
                "ip_address": entry["ip_address"],
                "ID": entry["ID"],
                "MachineClassOID": entry["MachineClassOID"],
                "IgnoreForAssignment": entry["IgnoreForAssignment"],
                "Productivity": entry["Productivity"],
                "FuelCapacity": entry["FuelCapacity"],
                "TKPHRate": entry["TKPHRate"],
                "MaxPayload": entry["MaxPayload"],
                "NominalPayload": entry["NominalPayload"],
                "DefaultMaterial": entry["DefaultMaterial"],
                "Material": entry["Material"],
                "EndpointID": entry["EndpointID"],
                "ExternalRef": entry["ExternalRef"],
                "ExternalDesc": entry["ExternalDesc"],
                "IsCrusher": entry["IsCrusher"],
            }
            # C++ sets waypointOID/destinationOID on loader in loop 2
            if loader_wp_oid is not None:
                loader["waypointOID"] = loader_wp_oid
                loader["destinationOID"] = loader_dest_oid
            if inload:
                loaders_out.append(loader)

        elif is_processor:
            processor_count += 1

            # C++ iterates ALL dump_zones (no break)
            # Sets waypointOID/destinationOID on processor only if match found
            dump_zone_id = None
            proc_wp_oid = None
            proc_dest_oid = None
            for dz in model.get("dump_zones", []):
                dz_moid = dz.get("machineOID", "")
                try:
                    dz_moid_int = int(dz_moid) if dz_moid else 0
                    m_oid_int = int(machine_oid) if machine_oid else 0
                except (ValueError, TypeError):
                    dz_moid_int = dz_moid
                    m_oid_int = machine_oid

                if dz_moid_int == m_oid_int and dz_moid:
                    if entry["IsActive"] == 1 and entry["IgnoreForAssignment"] == 0:
                        wp_oid = pos.get("waypoint_oid", "")
                        dest_oid = pos.get("destination_oid", "")
                        dz["waypointOID"] = wp_oid
                        dz["destinationOID"] = dest_oid
                        dz["destinationName"] = _find_destination_name(
                            dest_oid, destinations
                        )
                        dump_zone_id = dz.get("id")
                        proc_wp_oid = wp_oid
                        proc_dest_oid = dest_oid
                    else:
                        dz["machineOID"] = ""

            processor = {
                "id": processor_count,
                "dump_zoneID": dump_zone_id,
                "MachineOID": entry["MachineOID"],
                "name": entry["MachineName"],
                "key": "processors",
                "SerialNumber": entry["SerialNumber"],
                "ip_address": entry["ip_address"],
                "ID": entry["ID"],
                "MachineClassOID": entry["MachineClassOID"],
                "IgnoreForAssignment": entry["IgnoreForAssignment"],
                "Productivity": entry["Productivity"],
                "FuelCapacity": entry["FuelCapacity"],
                "TKPHRate": entry["TKPHRate"],
                "MaxPayload": entry["MaxPayload"],
                "NominalPayload": entry["NominalPayload"],
                "DefaultMaterial": entry["DefaultMaterial"],
                "Material": entry["Material"],
                "EndpointID": entry["EndpointID"],
                "ExternalRef": entry["ExternalRef"],
                "ExternalDesc": entry["ExternalDesc"],
                "IsCrusher": entry["IsCrusher"],
                "MachineClassType": entry["MachineClassType"],
                "MachineClassName": entry["MachineClassName"],
            }
            # C++ only sets waypointOID/destinationOID if a matching dump_zone found
            if proc_wp_oid is not None:
                processor["waypointOID"] = proc_wp_oid
                processor["destinationOID"] = proc_dest_oid
            processors_out.append(processor)

        elif is_fuel:
            # Match to charger zone (C++ no break — iterates all chargers)
            fuel_wp_oid = None
            fuel_dest_oid = None
            for ch in model.get("chargers", []):
                ch_moid = ch.get("machineOID", "")
                try:
                    ch_moid_int = int(ch_moid) if ch_moid else 0
                    m_oid_int = int(machine_oid) if machine_oid else 0
                except (ValueError, TypeError):
                    ch_moid_int = ch_moid
                    m_oid_int = machine_oid

                if ch_moid_int == m_oid_int and ch_moid:
                    wp_oid = pos.get("waypoint_oid", "")
                    dest_oid = pos.get("destination_oid", "")
                    ch["waypointOID"] = wp_oid
                    ch["destinationOID"] = dest_oid
                    ch["machineOID"] = entry["MachineOID"]
                    ch["destinationName"] = _find_destination_name(
                        dest_oid, destinations
                    )
                    fuel_wp_oid = wp_oid
                    fuel_dest_oid = dest_oid

            fuel_machine = {
                "id": entry["MachineOID"],
                "MachineOID": entry["MachineOID"],
                "name": entry["MachineName"],
                "key": "Fuel Bay",
                "machine_id": "",
                "MachineClassType": entry["MachineClassType"],
                "MachineClassName": entry["MachineClassName"],
                "Type": entry["Type"],
                "IsActive": entry["IsActive"],
                "SerialNumber": entry["SerialNumber"],
                "ip_address": entry["ip_address"],
                "ID": entry["ID"],
                "MachineClassOID": entry["MachineClassOID"],
                "IgnoreForAssignment": entry["IgnoreForAssignment"],
                "Productivity": entry["Productivity"],
                "FuelCapacity": entry["FuelCapacity"],
                "TKPHRate": entry["TKPHRate"],
                "MaxPayload": entry["MaxPayload"],
                "NominalPayload": entry["NominalPayload"],
                "DefaultMaterial": entry["DefaultMaterial"],
                "Material": entry["Material"],
                "EndpointID": entry["EndpointID"],
                "ExternalRef": entry["ExternalRef"],
                "ExternalDesc": entry["ExternalDesc"],
                "IsCrusher": entry["IsCrusher"],
            }
            # C++ sets waypointOID/destinationOID and initial_conditions inside match loop
            if fuel_wp_oid is not None:
                fuel_machine["waypointOID"] = fuel_wp_oid
                fuel_machine["destinationOID"] = fuel_dest_oid
                # fuel_zone_id comes from matched charger
                for ch in model.get("chargers", []):
                    if ch.get("machineOID") == entry["MachineOID"]:
                        fuel_machine["initial_conditions"] = {
                            "fuel_zone_id": ch.get("id", 0),
                        }
                        break
            fuel_machines_out.append(fuel_machine)

        elif is_layover:
            # Match to service_station zone (C++ no break — iterates all)
            lay_wp_oid = None
            lay_dest_oid = None
            for ss in model.get("service_stations", []):
                ss_moid = ss.get("machineOID", "")
                try:
                    ss_moid_int = int(ss_moid) if ss_moid else 0
                    m_oid_int = int(machine_oid) if machine_oid else 0
                except (ValueError, TypeError):
                    ss_moid_int = ss_moid
                    m_oid_int = machine_oid

                if ss_moid_int == m_oid_int and ss_moid:
                    wp_oid = pos.get("waypoint_oid", "")
                    dest_oid = pos.get("destination_oid", "")
                    ss["waypointOID"] = wp_oid
                    ss["destinationOID"] = dest_oid
                    ss["machineOID"] = entry["MachineOID"]
                    ss["destinationName"] = _find_destination_name(
                        dest_oid, destinations
                    )
                    lay_wp_oid = wp_oid
                    lay_dest_oid = dest_oid

            layover_entry = {
                "id": entry["MachineOID"],
                "MachineOID": entry["MachineOID"],
                "name": entry["MachineName"],
                "key": "Layover",
                "machine_id": "",
                "MachineClassType": entry["MachineClassType"],
                "MachineClassName": entry["MachineClassName"],
                "Type": entry["Type"],
                "IsActive": entry["IsActive"],
                "SerialNumber": entry["SerialNumber"],
                "ip_address": entry["ip_address"],
                "ID": entry["ID"],
                "MachineClassOID": entry["MachineClassOID"],
                "IgnoreForAssignment": entry["IgnoreForAssignment"],
                "Productivity": entry["Productivity"],
                "FuelCapacity": entry["FuelCapacity"],
                "TKPHRate": entry["TKPHRate"],
                "MaxPayload": entry["MaxPayload"],
                "NominalPayload": entry["NominalPayload"],
                "DefaultMaterial": entry["DefaultMaterial"],
                "Material": entry["Material"],
                "EndpointID": entry["EndpointID"],
                "ExternalRef": entry["ExternalRef"],
                "ExternalDesc": entry["ExternalDesc"],
                "IsCrusher": entry["IsCrusher"],
            }
            # C++ sets waypointOID/destinationOID and initial_conditions inside match loop
            if lay_wp_oid is not None:
                layover_entry["waypointOID"] = lay_wp_oid
                layover_entry["destinationOID"] = lay_dest_oid
                for ss in model.get("service_stations", []):
                    if ss.get("machineOID") == entry["MachineOID"]:
                        layover_entry["initial_conditions"] = {
                            "service_zone_id": ss.get("id", 0),
                        }
                        break
            layover_out.append(layover_entry)

    # Write machines.json
    machines_json = {
        "machines": all_machines,
        "haulers": haulers_out,
        "loaders": loaders_out,
        "processors": processors_out,
    }
    machines_path = os.path.join(output_path, "machines.json")
    with open(machines_path, "w", encoding="utf-8") as f:
        json.dump(machines_json, f, indent=4)

    # Update model with all machine arrays
    model["haulers"] = haulers_out
    model["loaders"] = loaders_out
    model["processors"] = processors_out
    if fuel_machines_out:
        model["FuelMachine"] = fuel_machines_out
    if layover_out:
        model["Layover"] = layover_out

    # C++ does NOT modify machine_list — it stays from default.json template

    # Remove empty zones (C++: remove_zones)
    _remove_zones(model)

    # Write updated model.json
    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=4)

    print(f"  Machines: {len(all_machines)} total")
    print(f"  Haulers: {len(haulers_out)}, Loaders: {len(loaders_out)}, "
          f"Processors: {len(processors_out)}")
    if fuel_machines_out:
        print(f"  Fuel Machines: {len(fuel_machines_out)}")
    if layover_out:
        print(f"  Layover: {len(layover_out)}")
    print(f"  Written: {machines_path}")
    print("[Stage 6] Fleet complete.")
