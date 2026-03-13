#!/usr/bin/env python3
"""
ADES Minestar Data Importer Pipeline - Python Implementation

Replicates the C++ pipeline:
  TestSql -> PreProcessor -> TSM -> Roads -> Zones -> Fleet -> Productivity

Usage:
  python -m python_pipeline.run --server localhost --user sa --password cocapizza2
  python -m python_pipeline.run --server host.docker.internal --user sa --password pass --site-id 3 --site-name Desig
"""

import argparse
import json
import os
import sys
import time

# Add parent directory to path for module imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ADES_DIR = os.path.dirname(SCRIPT_DIR)


def _load_request_json():
    """Load config from request.json (same format as C++ Docker pipeline)."""
    for path in [
        os.path.join(ADES_DIR, "request.json"),
        os.path.join(os.getcwd(), "request.json"),
    ]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f), path
    return None, None


def main():
    parser = argparse.ArgumentParser(
        description="ADES Minestar Data Importer Pipeline (Python)",
        epilog="Config can also be provided via request.json (ip, user, password, fidelity).",
    )
    parser.add_argument("--server", default=None, help="SQL Server address")
    parser.add_argument("--user", default=None, help="SQL Server username")
    parser.add_argument("--password", default=None, help="SQL Server password")
    parser.add_argument("--site-id", default="1", help="Site ID (for output folder naming)")
    parser.add_argument("--site-name", default="TestSite", help="Site name (for output folder naming)")
    parser.add_argument("--fidelity", default=None, help="Model fidelity level")
    parser.add_argument("--output-base", default=None, help="Base output directory (default: ./io)")
    parser.add_argument("--template-path", default=None, help="Path to default templates")
    parser.add_argument("--skip-extract", action="store_true", help="Skip SQL extraction (use existing CSVs)")
    parser.add_argument("--request", default=None, help="Path to request.json config file")
    args = parser.parse_args()

    # Load request.json as base config (CLI args override)
    req_cfg, req_path = None, None
    if args.request:
        if os.path.exists(args.request):
            with open(args.request, "r", encoding="utf-8") as f:
                req_cfg = json.load(f)
            req_path = args.request
    else:
        req_cfg, req_path = _load_request_json()

    if req_cfg:
        print(f"  Config:    {req_path}")

    # Merge: CLI args > request.json > defaults
    args.server = args.server or (req_cfg or {}).get("ip") or "localhost"
    args.user = args.user or (req_cfg or {}).get("user") or "sa"
    args.password = args.password or (req_cfg or {}).get("password")
    args.fidelity = args.fidelity or (req_cfg or {}).get("fidelity") or "Low"

    if not args.password and not args.skip_extract:
        parser.error("--password is required (or provide it in request.json)")

    # Resolve paths
    output_base = args.output_base or os.path.join(ADES_DIR, "io")
    output_path = os.path.join(output_base, f"{args.site_id}_{args.site_name}")
    template_path = args.template_path or os.path.join(ADES_DIR, "minestar_importer", "default")

    print("=" * 60)
    print("ADES Minestar Data Importer Pipeline (Python)")
    print("=" * 60)
    print(f"  Server:    {args.server}")
    print(f"  User:      {args.user}")
    print(f"  Site:      {args.site_id}_{args.site_name}")
    print(f"  Fidelity:  {args.fidelity}")
    print(f"  Output:    {output_path}")
    print(f"  Templates: {template_path}")
    print("=" * 60)

    os.makedirs(output_path, exist_ok=True)
    start_time = time.time()
    stages = {}

    # Stage 1: SQL Extraction
    if not args.skip_extract:
        try:
            from .extract import extract_all
            extract_all(args.server, args.user, args.password, output_path)
            stages["extract"] = True
        except Exception as e:
            print(f"[Stage 1] FAILED: {e}")
            stages["extract"] = False
            import traceback
            traceback.print_exc()
    else:
        print("[Stage 1] Skipped (--skip-extract)")
        stages["extract"] = True

    # Stage 2: PreProcessor
    try:
        from .preprocess import preprocess
        preprocess(output_path)
        stages["preprocess"] = True
    except Exception as e:
        print(f"[Stage 2] FAILED: {e}")
        stages["preprocess"] = False
        import traceback
        traceback.print_exc()

    # Stage 3: TSM
    try:
        from .tsm import compute_tsm
        compute_tsm(output_path)
        stages["tsm"] = True
    except Exception as e:
        print(f"[Stage 3] FAILED: {e}")
        stages["tsm"] = False
        import traceback
        traceback.print_exc()

    # Stage 4: Road Network
    try:
        from .roads import create_road_network
        create_road_network(output_path, args.fidelity)
        stages["roads"] = True
    except Exception as e:
        print(f"[Stage 4] FAILED: {e}")
        stages["roads"] = False
        import traceback
        traceback.print_exc()

    # Stage 5: Zones
    try:
        from .zones import configure_zones
        configure_zones(output_path, template_path)
        stages["zones"] = True
    except Exception as e:
        print(f"[Stage 5] FAILED: {e}")
        stages["zones"] = False
        import traceback
        traceback.print_exc()

    # Stage 6: Fleet
    try:
        from .fleet import configure_fleet
        configure_fleet(output_path, template_path)
        stages["fleet"] = True
    except Exception as e:
        print(f"[Stage 6] FAILED: {e}")
        stages["fleet"] = False
        import traceback
        traceback.print_exc()

    # Stage 7: Productivity
    try:
        from .productivity import configure_productivity
        configure_productivity(output_path)
        stages["productivity"] = True
    except Exception as e:
        print(f"[Stage 7] FAILED: {e}")
        stages["productivity"] = False
        import traceback
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

    if all(stages.values()):
        print("Pipeline completed successfully.")
        return 0
    else:
        print("Pipeline completed with errors.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
