
from typing import Any, Dict, Iterator

from .constants import SegmentClass
from .Segment import Segment
from .AMTCycleProdInfoMessage import AMTCycleProdInfoMessage
from .MapClasses import reasonSummary


class Cycle:
    """This is class representation of the cycle (dump to dump).

    Attributes:
    cycleId: unique id of cycle.
    machineId: unique id of machine undergoing current cycle.
    machineName: name of the machine.
    machineType: type of the machine.
    segments: dictionary containing list of empty and loaded segments.
    cycleStartTime: start time (UTC) of the cycle.
    cycleEndTime: end time (UTC) of the cycle.
    loadZone: id of load zone in the cycle.
    dumpZoneStart: id of dump zone from which the empty machine started in this cycle.
    dumpZoneEnd: id of dump zone from which the loaded machine reached for dumping.
    lossSummary: summary of effect of aslr reasons on the cycle productivity.
    uniqueReasons: list of all aslr reasons occurred in the cycle.
    startIndex: reference to index of first message of cycle in messages attribute in reader object.
    endIndex: reference to index of last message of cycle in messages attribute in reader object.
    isFullCycle: whether the cycle is complete or not.
    loss: productivity loss occurred in the cycle in seconds
    totalDistanceTravelled: distance travelled by machine in the cycle in meters
    """

    def __init__(self, cycle_id, machine_id, start_time):
        """constructor

        :param cycle_id: unique id of cycle
        :type cycle_id: int
        :param machine_id: unique id of machine undergoing current cycle.
        :type machine_id: int
        :param machine_name: name of the machine.
        :type machine_name: str
        :param machine_type: type of the machine.
        :type machine_type: str
        :param start_time: start time (UTC) of the cycle.
        :type start_time: datetime
        """

        self.cycleId = cycle_id
        self.machineId = machine_id
        # self.machineName = machine_name
        # self.machineType = machine_type
        self.segments = []
        self.cycleStartTime = start_time
        self.cycleEndTime = None
        self.loss = 0
        self.lossSummary = {}
        self.uniqueReasons = None
        self.messages: list[AMTCycleProdInfoMessage] = []
        self.loadZone = None
        self.dumpZoneStart = None
        self.dumpZoneEnd = None

        # Message Index Optimization: Store compressed indexes
        self.indexes = []
        self.isFullCycle = None
        self.totalDistanceTravelled = 0
        self.loadedPath = []
        self.emptyPath = []

        self.sourceDestinationName = None
        self.sinkDestinationName = None
        self.secondaryMachineName = None
        self.secondaryMachineClass = None
        self.processor = None
        self.minestarPayload = None

        self.grid = {}

    def to_dict(self):
        """Returns Cycle Object as a dictionary

        :return: dictionary containing all the properties of this cycle object
        :rtype: dict
        """

        return {
            "cycleId": self.cycleId,
            "machineId": self.machineId,
            # 'machineName': self.machineName,
            # 'machineType': self.machineType,
            "emptySegments": [segment.to_dict() for segment in self.segments if segment.segmentClass == SegmentClass.EMPTY],
            "loadedSegments": [segment.to_dict() for segment in self.segments if segment.segmentClass == SegmentClass.FULL],
            "cycleStartTime": self.cycleStartTime.timestamp() if self.cycleStartTime else None,
            "cycleEndTime": self.cycleEndTime.timestamp() if self.cycleEndTime else None,
            "loadZone": self.loadZone,
            "dumpZoneStart": self.dumpZoneStart,
            "dumpZoneEnd": self.dumpZoneEnd,
            "lossSummary": {reason: summary.to_dict() for reason, summary in self.lossSummary.items()},
            "uniqueReasons": list(self.uniqueReasons) if self.uniqueReasons else None,
            "indexes": self.indexes,  # Message indexes stored directly
            "isFullCycle": self.isFullCycle,
            "loss": self.loss,
            "totalDistanceTravelled": self.totalDistanceTravelled,
        }

    def addSegment(self, segment: Segment):
        """Adds Segment to the cycle. Segments with payload percent less than 50 are added as emptySegments else segment
        is added as loadedSegment. Updates cycle properties like loss, end time and total distance travelled based
        on the properties of the segment. And also updates loaded and empty paths

        :param segment: segment which belongs to current cycle.
        :type segment: : Segment
        """

        self.segments.append(segment)
        self.cycleEndTime = segment.actualEndTime
        self.loss += segment.loss
        self.totalDistanceTravelled += segment.totalDistanceTravelled
        if segment.indexes:
            self.indexes.extend(segment.indexes)
        if segment.segmentClass == SegmentClass.EMPTY:
            self.emptyPath.extend(segment.path)
            self.emptyPath.append((None, None))
        if segment.segmentClass == SegmentClass.FULL:
            self.loadedPath.extend(segment.path)
            self.loadedPath.append((None, None))

    def updateLossSummary(self):
        """Updates the loss summary of the cycle by adding the loss summaries of each segment in the cycle"""
        summary_list = []
        summary_list.extend([segment.lossSummary for segment in self.segments])
        reasons = []
        [reasons.extend(summary.keys()) for summary in summary_list]
        self.uniqueReasons = set(reasons)
        self.lossSummary = reasonSummary.addSummaries(summary_list, self.uniqueReasons, "segments", self.cycleId)
        for reason_summary in self.lossSummary.values():
            reason_summary.updateMapProperties()

    def updateIsFullCycle(self):
        """Updates whether the cycle is a full cycle."""
        if self.dumpZoneStart and self.dumpZoneEnd:
            self.isFullCycle = True
        else:
            self.isFullCycle = False

    def updateIndexes(self, increment):
        """Updates the index of messages of cycle in messages attribute in reader object when multiple reader objects is
        merged.

        :param increment: value by which indexes of messages should be changed.
        :type increment: int
        """
        # Update regular indexes
        self.indexes = [i + increment for i in self.indexes]

        # Update segment indexes
        for segment in self.segments:
            segment.updateIndexes(index_increment=increment)

    def to_array(self):
        """Returns Cycle Object as a list of values with original float precision"""

        return [
            [segment.to_array() if hasattr(segment, "to_array") else segment for segment in self.segments],
            self.cycleStartTime.timestamp() if self.cycleStartTime else None,
            self.cycleEndTime.timestamp() if self.cycleEndTime else None,
            self.loadZone,
            self.dumpZoneStart,
            self.dumpZoneEnd,
            {
                reason: summary.to_array() if hasattr(summary, "to_array") else summary
                for reason, summary in self.lossSummary.items()
            },
            list(self.uniqueReasons) if self.uniqueReasons else None,
            self.isFullCycle,
            self.loss,
            self.totalDistanceTravelled,
            [([coord for coord in point] if point else None) for point in self.loadedPath],
            [([coord for coord in point] if point else None) for point in self.emptyPath],
            self.sourceDestinationName,
            self.sinkDestinationName,
            self.secondaryMachineName,
            self.secondaryMachineClass,
            self.processor,
            self.minestarPayload,
            self.grid,
        ]

    def iter_loss_summary_records(self) -> Iterator[Dict[str, Any]]:
        """Yield loss summary records for each segment in the cycle."""

        for segment in self.segments:
            if hasattr(segment, "iter_loss_summary_records"):
                yield from segment.iter_loss_summary_records()

    @staticmethod
    def array_fields() -> list[str]:
        """Returns list of field names corresponding to the array indices in to_array()"""
        return [
            "segments",
            "cycleStartTime",
            "cycleEndTime",
            "loadZone",
            "dumpZoneStart",
            "dumpZoneEnd",
            "lossSummary",
            "uniqueReasons",
            "indexes",  # Now optimized with compression
            "isFullCycle",
            "loss",
            "totalDistanceTravelled",
            "loadedPath",
            "emptyPath",
            "sourceDestinationName",
            "sinkDestinationName",
            "secondaryMachineName",
            "secondaryMachineClass",
            "processor",
            "minestarPayload",
            "grid",
        ]

    def createSegments(self, messages, indexes):
        payload_change_points = [
            i for i in range(1, len(messages)) if (messages[i - 1].payloadPercent <= 50) and (messages[i].payloadPercent > 50)
        ]
        payload_change_points = payload_change_points if payload_change_points else [len(messages)]
        payloads = [m.payloadPercent for m in messages]

        empty_segment_messages = messages[0: payload_change_points[-1]]
        empty_segment_indexes = indexes[0: payload_change_points[-1]]
        empty_segment_payload = max(set(payloads[0: payload_change_points[-1]]), key=payloads.count)

        empty_segment = Segment(self.cycleId, self.machineId)
        empty_segment.addMessagesInfo(empty_segment_messages, empty_segment_indexes, empty_segment_payload)
        empty_segment.updateType(next_payload=None)
        self.addSegment(empty_segment)

        if payload_change_points[-1] != len(messages):
            loaded_segment_messages = messages[payload_change_points[-1]:]
            loaded_segment_indexes = indexes[payload_change_points[-1]:]
            loaded_segment_payload = max(set(payloads[payload_change_points[-1]:]), key=payloads.count)

            loaded_segment = Segment(self.cycleId, self.machineId)
            loaded_segment.addMessagesInfo(loaded_segment_messages, loaded_segment_indexes, loaded_segment_payload)
            loaded_segment.updateType(next_payload=None)
            self.addSegment(loaded_segment)
