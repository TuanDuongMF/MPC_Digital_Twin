#!/usr/bin/env python3
"""
Minestar data importer pipeline (Python version).

Stages (C++ parity):
  TestSql -> PreProcessor -> TSM -> Roads -> Zones -> Fleet -> Productivity

This module is integrated into the Flask backend via run_pipeline().
"""

import os
import time
import traceback

from ..config import MSSM_CONFIG, OUTPUT_PATH, EXAMPLE_JSON_PATH
from .extract import extract_all
from .fleet import configure_fleet
from .preprocess import preprocess
from .productivity import configure_productivity
from .roads import create_road_network
from .tsm import compute_tsm
from .zones import configure_zones


def run_pipeline(
    site_id="1",
    site_name="TestSite",
    fidelity=None,
):
    """Run the full Minestar importer pipeline.

    SQL Server credentials and paths are always taken from environment
    variables via backend.config (MSSM_CONFIG, OUTPUT_PATH,
    EXAMPLE_JSON_PATH). Callers must not override them.

    Returns:
        dict with:
            - output_path: directory containing CSV/JSON outputs
            - stages: {stage_name: bool}
            - elapsed: total runtime in seconds
    """
    # Environment defaults (MSSM_* from shared config)
    mssm_cfg = MSSM_CONFIG
    fidelity = fidelity or "Low"

    if not mssm_cfg.password:
        # Required for full pipeline, even though extract stage currently
        # skips SQL I/O; keep validation explicit.
        raise ValueError("MSSM_PASSWORD must be configured in environment.")

    # Resolve paths: must align with main backend OUTPUT_PATH / EXAMPLE_JSON_PATH
    if not OUTPUT_PATH:
        raise ValueError("OUTPUT_PATH must be configured for roads_network_pipeline.")
    if not EXAMPLE_JSON_PATH:
        raise ValueError("EXAMPLE_JSON_PATH must be configured for roads_network_pipeline.")

    output_path = os.path.join(OUTPUT_PATH, f"{site_id}_{site_name}")
    template_path = EXAMPLE_JSON_PATH

    print("=" * 60)
    print("ADES Minestar Data Importer Pipeline (Python)")
    print("=" * 60)
    print(f"  MSSM Server: {mssm_cfg.server}")
    print(f"  MSSM User:   {mssm_cfg.user}")
    print(f"  Site:        {site_id}_{site_name}")
    print(f"  Fidelity:    {fidelity}")
    print(f"  Output:      {output_path}")
    print(f"  Templates:   {template_path}")
    print("=" * 60)

    os.makedirs(output_path, exist_ok=True)
    start_time = time.time()
    stages = {}

    # Stage 1: SQL Extraction - match original Python pipeline behaviour
    try:
        print("[Stage 1] Extracting data from SQL Server...")
        extract_all(mssm_cfg.server, mssm_cfg.user, mssm_cfg.password, output_path)
        stages["extract"] = True
    except Exception as e:
        print(f"[Stage 1] FAILED: {e}")
        stages["extract"] = False
        traceback.print_exc()

    # Stage 2: PreProcessor
    try:
        preprocess(output_path)
        stages["preprocess"] = True
    except Exception as e:
        print(f"[Stage 2] FAILED: {e}")
        stages["preprocess"] = False
        traceback.print_exc()

    # Stage 3: TSM
    try:
        compute_tsm(output_path)
        stages["tsm"] = True
    except Exception as e:
        print(f"[Stage 3] FAILED: {e}")
        stages["tsm"] = False
        traceback.print_exc()

    # Stage 4: Road Network
    try:
        create_road_network(output_path, fidelity)
        stages["roads"] = True
    except Exception as e:
        print(f"[Stage 4] FAILED: {e}")
        stages["roads"] = False
        traceback.print_exc()

    # Stage 5: Zones
    try:
        configure_zones(output_path, template_path)
        stages["zones"] = True
    except Exception as e:
        print(f"[Stage 5] FAILED: {e}")
        stages["zones"] = False
        traceback.print_exc()

    # Stage 6: Fleet
    try:
        configure_fleet(output_path, template_path)
        stages["fleet"] = True
    except Exception as e:
        print(f"[Stage 6] FAILED: {e}")
        stages["fleet"] = False
        traceback.print_exc()

    # Stage 7: Productivity
    try:
        configure_productivity(output_path)
        stages["productivity"] = True
    except Exception as e:
        print(f"[Stage 7] FAILED: {e}")
        stages["productivity"] = False
        traceback.print_exc()

    # Summary
    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print("Pipeline Summary")
    print("=" * 60)
    for name, ok in stages.items():
        status = "OK" if ok else "FAILED"
        print(f"  {name:20s} [{status}]")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"  Output:  {output_path}")
    print("=" * 60)

    return {
        "output_path": output_path,
        "stages": stages,
        "elapsed": elapsed,
    }
