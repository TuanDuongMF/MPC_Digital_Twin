
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from sklearn.cluster import DBSCAN
from .constants import SegmentType, ZoneType
from .AMTCycleProdInfoMessage import AMTCycleProdInfoMessage
from .Cycle import Cycle
from .Segment import Segment
from .Zone import Zone


class AMTCycleProdInfoReader:

    @classmethod
    def createLoadDumpAreas(
        cls,
        data_df,
        zone_type,
        eps=50,
    ):
        """creates polygon areas around the zone points

        :param data_df: dataframe containing zone information
        :type data_df: pandas.Dataframe
        :param zone_type: type of zone (dump/load)
        :type zone_type: str
        :param eps: epsilon, defaults to 50
        :type eps: int, optional
        """
        zones = []
        dbscan = DBSCAN(eps=eps)
        dbscan.fit(data_df[["x", "y"]])
        data_df["cluster"] = dbscan.labels_

        # Filter out noise points (label = -1)
        valid_data = data_df[data_df["cluster"] != -1]
        if valid_data.empty:
            return zones

        for cluster_id, points_data in valid_data.groupby("cluster"):
            zone = Zone(zone_type=zone_type, zone_id=None)
            zone.updatePoints(
                list(zip(points_data["x"], points_data["y"])),
                cycle_ids=points_data["cycleId"].tolist() if "cycleId" in points_data.columns else None,
            )
            zone.updateProperties()
            zones.append(zone)

        return zones

    @classmethod
    def parse_cp1_data(cls, data: List[tuple], machine_info: Dict[str, Any]):
        """Parse CP1 data into messages and extract zone points"""
        if not data:
            return None, None

        machine_name = machine_info["Name"]
        machine_type = machine_info["TypeName"]

        messages: List[AMTCycleProdInfoMessage] = [
            AMTCycleProdInfoMessage(row, machine_name, machine_type, message_type="CP1") for row in data
        ]

        if not messages:
            return None, None

        cycles: List[Cycle] = []
        zones: List[Zone] = []
        dump_zone_points: List[Tuple[float, float, int, int, str]] = []
        load_zone_points: List[Tuple[float, float, int, int, str]] = []

        # Group contiguous messages by segmentId
        segments: List[Tuple[int, List[AMTCycleProdInfoMessage], List[int]]] = []
        current_segment_id: Optional[int] = None
        current_segment_messages: List[AMTCycleProdInfoMessage] = []
        current_indexes: List[int] = []

        for idx, msg in enumerate(messages):
            if current_segment_id is None:
                current_segment_id = msg.segmentId

            if msg.segmentId != current_segment_id:
                segments.append((current_segment_id, current_segment_messages, current_indexes))
                current_segment_id = msg.segmentId
                current_segment_messages = []
                current_indexes = []

            current_segment_messages.append(msg)
            current_indexes.append(idx)

        if current_segment_messages:
            segments.append((current_segment_id, current_segment_messages, current_indexes))

        cycle: Optional[Cycle] = None

        for segment_position, (_, segment_messages, segment_indexes) in enumerate(segments):
            if not segment_messages:
                continue

            if cycle is None:
                first_message = segment_messages[0]
                cycle = Cycle(first_message.segmentId, first_message.machineId, first_message.start_time)

            payloads = [message.payloadPercent for message in segment_messages if message.payloadPercent is not None]
            payload = max(set(payloads), key=payloads.count) if payloads else None

            segment_obj = Segment(cycle.cycleId, cycle.machineId)
            segment_obj.addMessagesInfo(segment_messages, segment_indexes, payload)

            # Determine next payload for segment classification
            next_payload: Optional[float] = None
            if segment_position + 1 < len(segments):
                next_segment_messages = segments[segment_position + 1][1]
                next_payloads = [
                    message.payloadPercent for message in next_segment_messages if message.payloadPercent is not None
                ]
                if next_payloads:
                    next_payload = max(set(next_payloads), key=next_payloads.count)

            segment_obj.updateType(next_payload=next_payload)

            for message in segment_messages:
                message.cycleId = cycle.cycleId
            cycle.addSegment(segment_obj)
            cycle.messages.extend(segment_messages)

            if segment_obj.segmentType == SegmentType.SPOTTING_AT_SINK:
                for message in segment_messages:
                    message.updateQueuing()
                if segment_obj.preferredPath:
                    dump_zone_points.extend(
                        [
                            (point[0], point[1], cycle.cycleId, cycle.machineId, machine_name)
                            for point in segment_obj.preferredPath
                            if point
                        ]
                    )

            if segment_obj.segmentType == SegmentType.SPOTTING_AT_SOURCE:
                for message in segment_messages:
                    message.updateQueuing()
                if segment_obj.preferredPath:
                    load_zone_points.extend(
                        [
                            (point[0], point[1], cycle.cycleId, cycle.machineId, machine_name)
                            for point in segment_obj.preferredPath
                            if point
                        ]
                    )

            if segment_obj.isCycleEnd:
                cycle.updateLossSummary()
                cycle.updateIsFullCycle()
                cycles.append(cycle)
                cycle = None

        if cycle is not None:
            cycle.updateLossSummary()
            cycle.updateIsFullCycle()
            cycles.append(cycle)

        if load_zone_points:
            load_data_df = pd.DataFrame(load_zone_points, columns=["x", "y", "cycleId", "machineId", "machineName"])
            zones.extend(cls.createLoadDumpAreas(load_data_df, ZoneType.LOAD))
        if dump_zone_points:
            dump_data_df = pd.DataFrame(dump_zone_points, columns=["x", "y", "cycleId", "machineId", "machineName"])
            zones.extend(cls.createLoadDumpAreas(dump_data_df, ZoneType.DUMP))

        return cycles, zones

    @classmethod
    def parse_cp2_data(cls, data: List[tuple], machine_info: Dict[str, Any]):
        """Parse CP2 data into messages and extract zone points

        CP2 data is structured differently from CP1:
        - Cycles are grouped by cycleId (not segmentId)
        - Uses different segment type classification logic
        - Simpler cycle end detection based on cycleId changes
        """
        if not data:
            return None, None

        machine_name = machine_info["Name"]
        machine_type = machine_info["TypeName"]

        messages: List[AMTCycleProdInfoMessage] = [
            AMTCycleProdInfoMessage(row, machine_name, machine_type, message_type="CP2") for row in data
        ]

        if not messages:
            return None, None

        cycles: List[Cycle] = []
        zones: List[Zone] = []
        dump_zone_points: List[Tuple[float, float, int, int, str]] = []
        load_zone_points: List[Tuple[float, float, int, int, str]] = []

        # Group contiguous messages by cycleId
        cycle_groups: List[Tuple[int, List[AMTCycleProdInfoMessage], List[int]]] = []
        current_cycle_id: Optional[int] = None
        current_cycle_messages: List[AMTCycleProdInfoMessage] = []
        current_indexes: List[int] = []

        for idx, msg in enumerate(messages):
            if current_cycle_id is None:
                current_cycle_id = msg.cycleId

            if msg.cycleId != current_cycle_id:
                cycle_groups.append((current_cycle_id, current_cycle_messages, current_indexes))
                current_cycle_id = msg.cycleId
                current_cycle_messages = []
                current_indexes = []

            current_cycle_messages.append(msg)
            current_indexes.append(idx)

        if current_cycle_messages:
            cycle_groups.append((current_cycle_id, current_cycle_messages, current_indexes))

        for cycle_id, cycle_messages, cycle_indexes in cycle_groups:
            if not cycle_messages:
                continue

            first_message = cycle_messages[0]
            cycle = Cycle(first_message.segmentId, first_message.machineId, first_message.start_time)
            cycle.cycleId = cycle_id

            for message in cycle_messages:
                message.cycleId = cycle.cycleId

            cycle.createSegments(cycle_messages, cycle_indexes)
            cycle.updateLossSummary()
            cycle.updateIsFullCycle()
            cycle.messages.extend(cycle_messages)
            cycles.append(cycle)

            for segment in cycle.segments:
                if segment.segmentType == SegmentType.TRAVELLING_EMPTY:
                    if segment.preferredPath:
                        dump_zone_points.extend(
                            [
                                (point[0], point[1], cycle.cycleId, cycle.machineId, machine_name)
                                for point in segment.preferredPath[-10:]
                                if point
                            ]
                        )
                else:
                    if segment.preferredPath:
                        load_zone_points.extend(
                            [
                                (point[0], point[1], cycle.cycleId, cycle.machineId, machine_name)
                                for point in segment.preferredPath[-10:]
                                if point
                            ]
                        )

        # Create zones from extracted points
        if load_zone_points:
            load_data_df = pd.DataFrame(load_zone_points, columns=["x", "y", "cycleId", "machineId", "machineName"])
            zones.extend(cls.createLoadDumpAreas(load_data_df, ZoneType.LOAD))
        if dump_zone_points:
            dump_data_df = pd.DataFrame(dump_zone_points, columns=["x", "y", "cycleId", "machineId", "machineName"])
            zones.extend(cls.createLoadDumpAreas(dump_data_df, ZoneType.DUMP))

        return cycles, zones