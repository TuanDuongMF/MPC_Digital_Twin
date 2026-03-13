"""Stage 7: Productivity Configuration - replicates Productivity_Configuration from C++.

Reads Production_plans.csv and Materials.csv to add operations/schedules to model.json.
"""

import csv
import json
import os


def _read_csv_rows(filepath):
    """Read CSV into list of dicts."""
    rows = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except FileNotFoundError:
        pass
    return rows


def configure_productivity(output_path):
    """Add production schedules and materials to model.json."""
    print("[Stage 7] Configuring productivity...")

    model_path = os.path.join(output_path, "model.json")
    with open(model_path, "r", encoding="utf-8") as f:
        model = json.load(f)

    prod_plans = _read_csv_rows(os.path.join(output_path, "Production_plans.csv"))
    materials = _read_csv_rows(os.path.join(output_path, "Materials.csv"))

    # Build material lookup
    material_map = {}
    for mat in materials:
        mat_oid = str(mat.get("MATERIAL_OID", "")).strip()
        if mat_oid:
            material_map[mat_oid] = {
                "name": mat.get("NAME", ""),
                "description": mat.get("DESCRIPTION", ""),
                "density": float(mat.get("BANKDENSITY", 0) or 0),
                "loose_density": float(mat.get("LOOSEDENSITY", 0) or 0),
                # C++ stores BANKDENSITY/LOOSEDENSITY as raw CSV strings
                "density_str": mat.get("BANKDENSITY", ""),
                "loose_density_str": mat.get("LOOSEDENSITY", ""),
            }

    # Build operations
    schedule_data = []
    seq_id = 0  # C++ starts at i = 0

    load_zones = model.get("load_zones", [])
    dump_zones = model.get("dump_zones", [])

    for plan in prod_plans:
        prod_oid = str(plan.get("PROD_REQUEST_OID", "")).strip()
        min_rate = plan.get("MIN_RATE", "")
        from_server = str(plan.get("FROM_SERVER", "")).strip()
        to_server = str(plan.get("TO_SERVER", "")).strip()
        material_oid = str(plan.get("MATERIAL", "")).strip()

        # Convert to int for comparison (C++ uses strtoll)
        try:
            from_server_int = int(from_server) if from_server else 0
        except ValueError:
            from_server_int = 0
        try:
            to_server_int = int(to_server) if to_server else 0
        except ValueError:
            to_server_int = 0
        try:
            material_oid_int = int(material_oid) if material_oid else 0
        except ValueError:
            material_oid_int = 0

        entry = {
            "id": seq_id,
            "productionplanOID": prod_oid,
        }
        seq_id += 1
        entry["route"] = ""

        # C++ linear search: lz["machineOID"] == strtoll(production_goals[3])
        for lz in load_zones:
            lz_moid = lz.get("machineOID", "")
            try:
                lz_moid_int = int(lz_moid) if lz_moid else 0
            except (ValueError, TypeError):
                lz_moid_int = 0
            if lz_moid_int == from_server_int and from_server_int != 0:
                entry["load_zone"] = lz.get("name", "")

        # C++ linear search: dz["machineOID"] == strtoll(production_goals[4])
        for dz in dump_zones:
            dz_moid = dz.get("machineOID", "")
            try:
                dz_moid_int = int(dz_moid) if dz_moid else 0
            except (ValueError, TypeError):
                dz_moid_int = 0
            if dz_moid_int == to_server_int and to_server_int != 0:
                entry["dump_zone"] = dz.get("name", "")

        # C++ material matching: strtoll comparison
        for mat_oid_key, mat_info in material_map.items():
            try:
                mat_key_int = int(mat_oid_key) if mat_oid_key else 0
            except ValueError:
                mat_key_int = 0
            if mat_key_int == material_oid_int and material_oid_int != 0:
                entry["material"] = mat_info.get("name", "")
                entry["materialOID"] = material_oid_int  # C++ strtoll -> int
                entry["materialBankDensity"] = mat_info.get("density_str", "")
                entry["materialLooseDensity"] = mat_info.get("loose_density_str", "")
                entry["density"] = mat_info.get("density_str", "")

        entry["multiple_routes"] = False
        entry["auto_generate_route"] = True

        # C++ uses production_goals[1] which is MIN_RATE
        try:
            quantity = float(min_rate) if min_rate else 0.0
        except (ValueError, TypeError):
            quantity = 0.0
        entry["quantity"] = quantity

        schedule_data.append(entry)

    operations = {
        "material_schedules": {
            "all_material_schedule": [
                {
                    "id": 1,
                    "name": "Material Schedule 1",
                    "hauler_assignment": {
                        "scheduling_method": "default_production_target_based",
                    },
                    "data": schedule_data,
                }
            ],
            "selected_material": 1,
        }
    }

    model["operations"] = operations

    # C++ does NOT create a top-level "materials" key in model.json

    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=4)

    print(f"  Production entries: {len(schedule_data)}")
    print(f"  Materials: {len(material_map)}")
    print(f"  Written: {model_path}")
    print("[Stage 7] Productivity complete.")
