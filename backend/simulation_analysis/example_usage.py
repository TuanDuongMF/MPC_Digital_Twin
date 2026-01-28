"""
Example Usage - GPS to Events Converter

This script demonstrates how to use the simulation_analysis module
to convert AMT telemetry data to simulation events.

Three usage patterns are shown:
1. Convert from Cycle objects (using AMTCycleProdInfoReader)
2. Convert from raw database telemetry
3. Convert from a model file with manual data
"""

import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation_analysis import GPSToEventsConverter


def example_1_from_model_file():
    """
    Example 1: Load model file and convert sample telemetry data.
    """
    print("=" * 60)
    print("Example 1: Convert from model file with sample data")
    print("=" * 60)
    
    # Path to model file
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "exampleJSON",
        "example_model.json"
    )
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return
    
    # Initialize converter
    converter = GPSToEventsConverter(model_path=model_path)
    
    print(f"Loaded model with {len(converter.model.get('nodes', []))} nodes")
    print(f"Loaded model with {len(converter.model.get('roads', []))} roads")
    
    # Sample telemetry data (simulating GPS points along road)
    # Format: list of dicts with pathEasting, pathNorthing, pathElevation,
    #         actualSpeed, payloadPercent, segmentId
    sample_data = [
        {
            "pathEasting": 1357019,  # in mm
            "pathNorthing": -936497,
            "pathElevation": -91181,
            "actualSpeed": 0,
            "payloadPercent": 10,
            "segmentId": 1400000000,
            "actualElapsedTime": 0,
        },
        {
            "pathEasting": 1247045,
            "pathNorthing": -1012015,
            "pathElevation": -97268,
            "actualSpeed": 25,
            "payloadPercent": 15,
            "segmentId": 1400000000,
            "actualElapsedTime": 30000,
        },
        {
            "pathEasting": 1141527,
            "pathNorthing": -1023594,
            "pathElevation": -102625,
            "actualSpeed": 30,
            "payloadPercent": 80,  # Loaded now
            "segmentId": 1400000000,
            "actualElapsedTime": 60000,
        },
        {
            "pathEasting": 1009295,
            "pathNorthing": -917816,
            "pathElevation": -112168,
            "actualSpeed": 28,
            "payloadPercent": 80,
            "segmentId": 1400000000,
            "actualElapsedTime": 120000,
        },
        {
            "pathEasting": 767406,
            "pathNorthing": -768266,
            "pathElevation": -93630,
            "actualSpeed": 32,
            "payloadPercent": 80,
            "segmentId": 1400000000,
            "actualElapsedTime": 180000,
        },
    ]
    
    # Convert messages
    events = converter.convert_messages(
        messages=sample_data,
        machine_id=1001,
        machine_name="AMT_T01",
        min_node_distance=10.0,
        max_search_distance=100.0,
    )
    
    print(f"\nGenerated {len(events)} events:")
    for event in events:
        print(f"  [{event['eid']}] {event['etype']} at t={event['time']:.2f}min")
        if event.get('hauler'):
            h = event['hauler']
            print(f"       Speed: {h['speed']:.1f} km/h, Payload: {h['payload']:.1f}t")
        if event.get('node'):
            print(f"       Node: {event['node']['id']}")
    
    # Create output structure
    output = converter.create_output(events, include_summary=True)
    
    print(f"\nOutput summary:")
    if output['data'].get('summary'):
        summary = output['data']['summary']
        print(f"  Total events: {summary.get('total_events', 0)}")
        print(f"  Total haulers: {summary.get('total_haulers', 0)}")
        print(f"  Duration: {summary.get('simulation_duration_minutes', 0):.1f} min")


def example_2_programmatic():
    """
    Example 2: Create model programmatically and convert data.
    """
    print("\n" + "=" * 60)
    print("Example 2: Programmatic model creation")
    print("=" * 60)
    
    # Define a simple road network
    model = {
        "nodes": [
            {"id": 1, "coords": [0, 0, 0]},
            {"id": 2, "coords": [100, 0, 0]},
            {"id": 3, "coords": [200, 50, 5]},
            {"id": 4, "coords": [300, 100, 10]},
            {"id": 5, "coords": [400, 100, 10]},
        ],
        "roads": [
            {"id": 1, "name": "Main Road", "nodes": [1, 2, 3, 4, 5]},
        ],
    }
    
    # Initialize converter with model data
    converter = GPSToEventsConverter(model_data=model)
    
    # Simulate a truck path
    path_data = [
        {"pathEasting": 5, "pathNorthing": 2, "pathElevation": 0,
         "actualSpeed": 0, "payloadPercent": 0, "segmentId": 1500000000, "actualElapsedTime": 0},
        {"pathEasting": 98, "pathNorthing": 5, "pathElevation": 0,
         "actualSpeed": 20, "payloadPercent": 0, "segmentId": 1500000000, "actualElapsedTime": 10000},
        {"pathEasting": 195, "pathNorthing": 48, "pathElevation": 5,
         "actualSpeed": 25, "payloadPercent": 85, "segmentId": 1500000000, "actualElapsedTime": 20000},
        {"pathEasting": 305, "pathNorthing": 102, "pathElevation": 10,
         "actualSpeed": 20, "payloadPercent": 85, "segmentId": 1500000000, "actualElapsedTime": 30000},
        {"pathEasting": 395, "pathNorthing": 98, "pathElevation": 10,
         "actualSpeed": 0, "payloadPercent": 10, "segmentId": 1500000000, "actualElapsedTime": 40000},
    ]
    
    events = converter.convert_messages(
        messages=path_data,
        machine_id=2001,
        machine_name="TEST_TRUCK",
        min_node_distance=50.0,
        max_search_distance=20.0,
    )
    
    print(f"Generated {len(events)} events from programmatic model")
    for event in events:
        etype = event['etype']
        time = event['time']
        node_id = event.get('node', {}).get('id', 'N/A')
        print(f"  {etype}: time={time:.2f}min, node={node_id}")


def example_3_with_reader():
    """
    Example 3: Integrate with AMTCycleProdInfoReader (requires database).
    
    This example shows the integration pattern - actual execution requires
    database connection and valid machine data.
    """
    print("\n" + "=" * 60)
    print("Example 3: Integration with AMTCycleProdInfoReader")
    print("=" * 60)
    
    print("""
    # This example shows the integration pattern:
    
    from Reader import AMTCycleProdInfoReader
    from simulation_analysis import GPSToEventsConverter
    
    # 1. Parse CP1 data using existing reader
    cycles, zones = AMTCycleProdInfoReader.parse_cp1_data(
        data=raw_database_data,
        machine_info={"Name": "AMT001", "TypeName": "Cat 797F"}
    )
    
    # 2. Load model (generated from convert_to_model.py)
    converter = GPSToEventsConverter(model_path="model_MySite.json")
    
    # 3. Convert cycles to events
    events = converter.convert_cycles(
        cycles=cycles,
        machine_id=1,
        machine_name="AMT001",
    )
    
    # 4. Save events for animation
    converter.save_events(events, "simulation_events.json")
    
    # The output file can be used with the animation player.
    """)


def main():
    """Run all examples."""
    example_1_from_model_file()
    example_2_programmatic()
    example_3_with_reader()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
