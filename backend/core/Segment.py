from datetime import datetime
from typing import Any, Dict, Iterator, Optional

from .constants import SegmentClass, SegmentType
from .LossBucket import lossBucket
from .MapClasses import reasonSummary


class Segment:
    """This is class representation of a segment. 
    """

    def __init__(self, cycle_id, machine):
        """constructor

        :param cycle_id: unique id of cycle to which this segment belongs.
        :type cycle_id: int
        :param messages: list of messages of this segment
        :type messages: list
        :param start_index: reference to index of first message of this segment in messages attribute of 
        reader object.
        :type start_index: int
        :param end_index: reference to index of last message of this segment in messages attribute of 
        reader object.
        :type end_index: int
        :param payload: mode of all payload values in this segment
        :type payload: float
        """
        self.segmentId = None
        self.machineId = machine
        self.cycleId = cycle_id
        self.actualStartTime = None
        self.actualEndTime = None
        self.expectedStartTime = None
        self.expectedEndTime = None
        self.payload = None
        self.segmentType = None
        self.segmentClass = None
        self.loss = 0
        self.lossSummary = {}

        self.indexes = []
        self.isReversing = True
        self.isCycleEnd = False
        self.bucketsCount = 0
        self.lossBuckets = []

        self.path = []
        self.preferredPath = []
        self.totalDistanceTravelled = 0

        # minestar data
        self.delayDescription = None
        self.delayCategory = None

        # uncomment for grid data
        # self.grid = {}

    def addMessagesInfo(self, messages, segment_indexes, payload=None):
        """this function updates properties of the segments using messages which belong to the segment

        :param messages: list of messages of the segment
        :type messages: list
        :param segment_indexes: indexes of messages in 'reader.messages' list
        :type segment_indexes: list
        :param payload: payload percent of segment, defaults to None
        :type payload: int, optional
        """
        first_msg = messages[0]
        last_msg = messages[-1]
        self.segmentId = first_msg.segmentId
        if self.actualStartTime is None:
            self.actualStartTime, self.actualEndTime = first_msg.actualTime, last_msg.actualTime
        if self.expectedStartTime is None:
            self.expectedStartTime, self.expectedEndTime = first_msg.expectedTime, last_msg.expectedTime

        self.indexes = segment_indexes
        if not payload:
            payloads = [msg.payloadPercent for msg in messages]
            payload = max(set(payloads), key=payloads.count)
        self.payload = payload
        self.segmentClass = SegmentClass.EMPTY if self.payload <= 50 else SegmentClass.FULL

        self.createLossBucketsAndTiles(messages, segment_indexes[0])
        self.updatePath(messages)

    def to_dict(self):
        """this function returns a dictionary containing the information about the segment

        :return: segment data 
        :rtype: dict
        """
        # Convert enum to serializable value
        segment_type_value = None
        if self.segmentType is not None:
            if isinstance(self.segmentType, SegmentType):
                segment_type_value = self.segmentType.value if hasattr(self.segmentType, 'value') else str(self.segmentType)
            else:
                segment_type_value = self.segmentType
        
        # Convert datetime objects to timestamps
        actual_start_time = self.actualStartTime.timestamp() if self.actualStartTime and isinstance(self.actualStartTime, datetime) else self.actualStartTime
        expected_start_time = self.expectedStartTime.timestamp() if self.expectedStartTime and isinstance(self.expectedStartTime, datetime) else self.expectedStartTime
        actual_end_time = self.actualEndTime.timestamp() if self.actualEndTime and isinstance(self.actualEndTime, datetime) else self.actualEndTime
        expected_end_time = self.expectedEndTime.timestamp() if self.expectedEndTime and isinstance(self.expectedEndTime, datetime) else self.expectedEndTime

        return {
            'machineId': self.machineId,
            'cycleId': self.cycleId,
            'segmentId': self.segmentId,
            'actualStartTime': actual_start_time,
            'expectedStartTime': expected_start_time,
            'actualEndTime': actual_end_time,
            'expectedEndTime': expected_end_time,
            'loss': self.loss,
            'payload': self.payload,
            'segmentType': segment_type_value,
            'isReversing': self.isReversing,
            'isCycleEnd': self.isCycleEnd,
            'bucketsCount': self.bucketsCount,
            'lossSummary': {reason: summary.to_dict() for reason, summary in self.lossSummary.items()},
        }

    def createLossBucketsAndTiles(self, messages, start_index, tile_length=10):
        """This functions divides the segment into loss buckets based on change in actual aslr reason 
        and computes the various metrics of that loss bucket

        :param messages: list of messages of the segment
        :type messages: list
        :param start_index: reference to index of first message of this segment in messages attribute of 
        reader object.
        :type start_index: int
        :param tile_length:
        :type tile_length: int
        """
        # if messages[0].actualTime != self.actualStartTime:
        #     bucket = lossBucket(self.bucketsCount)
        #     bucket.updateBucketLossInfo(self.actualStartTime, self.expectedStartTime, messages[0].actualTime,
        #                                 messages[0].expectedTime, start_index - 1, start_index)
        #     self.lossBuckets.append(bucket)
        #     self.updateLossSummary(bucket)

        # uncomment for grid data
        # grid_data = {}
        prev_msg = None
        bucket_start = start_index
        bucket_end = start_index
        self.totalDistanceTravelled = messages[-1].cycleDistance - messages[0].cycleDistance
        for msg in messages:
            if msg.actualSpeed > 0 and msg.expectedSpeed > 0 and self.isReversing:
                self.isReversing = False
            if prev_msg is not None:
                if prev_msg.actualASLR != msg.actualASLR or prev_msg.expectedASLR != msg.expectedASLR:
                    self.bucketsCount += 1
                    first_msg = messages[bucket_start - start_index]
                    bucket = lossBucket(self.bucketsCount, first_msg)
                    bucket.updateBucketLossInfo(first_msg.actualTime, first_msg.expectedTime, msg.actualTime,
                                                msg.expectedTime, bucket_start, bucket_end, first_msg.cycleDistance,
                                                msg.cycleDistance, (first_msg.pathEasting, first_msg.pathNorthing),
                                                (msg.pathEasting, msg.pathNorthing))
                    bucket.updateIsBucketPowerLimited(messages[bucket_start - start_index: bucket_end - start_index])
                    # if bucket.actualASLRName == 'HealthEvent':
                    #     bucket.updateReasonInfo(messages[bucket_start - start_index: bucket_end - start_index])

                    for m in messages[bucket_start - start_index: bucket_end - start_index]:
                        m.efficiency = bucket.efficiency
                        m.painIndex = bucket.painIndex
                        m.bucketLoss = bucket.bucketLoss

                    self.lossBuckets.append(bucket)
                    self.loss += bucket.bucketLoss
                    self.updateLossSummary(bucket)
                    bucket_start = bucket_end

            # creating tiles # uncomment for grid data
            # grid_data.setdefault((msg.pathEasting // tile_length, msg.pathNorthing // tile_length), []).append(
            #     (msg, bucket_end))

            prev_msg = msg
            bucket_end += 1

        bucket_end = bucket_end if bucket_end - bucket_start > 1 else bucket_start
        self.bucketsCount += 1
        first_msg = messages[bucket_start - start_index]
        bucket = lossBucket(self.bucketsCount, first_msg)
        bucket.updateBucketLossInfo(first_msg.actualTime, first_msg.expectedTime, msg.actualTime,
                                    msg.expectedTime, bucket_start, bucket_end, first_msg.cycleDistance,
                                    msg.cycleDistance, (first_msg.pathEasting, first_msg.pathNorthing),
                                    (msg.pathEasting, msg.pathNorthing))
        bucket.updateIsBucketPowerLimited(messages[bucket_start - start_index: bucket_end - start_index])

        # if bucket.actualASLRName == 'HealthEvent':
        #     bucket.updateReasonInfo(messages[bucket_start - start_index: bucket_end - start_index])

        for m in messages[bucket_start - start_index: bucket_end - start_index]:
            m.efficiency = bucket.efficiency
            m.painIndex = bucket.painIndex
            m.bucketLoss = bucket.bucketLoss

        self.lossBuckets.append(bucket)
        self.loss += bucket.bucketLoss
        self.updateLossSummary(bucket)
        for reason_summary in self.lossSummary.values():
            reason_summary.updateMapProperties()

        # for tile_id, tile_data in grid_data.items():
        #     tile = Tile(tile_id[0], tile_id[1], tile_length)
        #     tile.cycles = [self.cycleId]
        #     tile.machines = [self.machineId]
        #     tile_messages, tile_indexes = list(map(list, zip(*tile_data)))
        #     tile.messageIndexes[self.machineId] = tile_indexes
        #     tile.updateTileData(tile_messages)
        #     self.grid[tile_id] = tile

    def updateCp2type(self, next_payload):
        """This function updates the type of segment based on current segments payload, next segments payload
          and whether the truck is reversing or not.

        :param next_payload: _description_
        :type next_payload: _type_
        """
        payload_threshold = 50
        if next_payload is not None:
            if self.payload <= payload_threshold:
                self.segmentType = SegmentType.TRAVELLING_EMPTY
            else:
                self.segmentType = SegmentType.TRAVELLING_FULL
            self.isCycleEnd = True

    def updateType(self, next_payload):
        """This function updates the type of segment based on current segments payload, next segments payload
          and whether the truck is reversing or not.

        :param next_payload: _description_
        :type next_payload: _type_
        """
        payload_threshold = 50
        if next_payload is not None:
            if self.payload <= payload_threshold:
                if next_payload > payload_threshold and self.isReversing:
                    self.segmentType = SegmentType.SPOTTING_AT_SOURCE
                    # uncomment for queing
                    # if self.lossSummary.get('FollowingSiteAwareVehicle'):
                    #     self.lossSummary['Queing'] = self.lossSummary.pop('FollowingSiteAwareVehicle')
                    #     self.lossSummary['Queing'].updateToQueuingReason()
                else:
                    self.segmentType = SegmentType.TRAVELLING_EMPTY
            else:
                if next_payload <= payload_threshold and self.isReversing:
                    self.segmentType = SegmentType.SPOTTING_AT_SINK
                    # uncomment for queing
                    # if self.lossSummary.get('FollowingSiteAwareVehicle'):
                    #     self.lossSummary['Queing'] = self.lossSummary.pop('FollowingSiteAwareVehicle')
                    #     self.lossSummary['Queing'].updateToQueuingReason()
                else:
                    self.segmentType = SegmentType.TRAVELLING_FULL
                if next_payload <= payload_threshold:
                    self.isCycleEnd = True
        else:
            if self.payload <= payload_threshold:
                self.segmentType = SegmentType.TRAVELLING_EMPTY
            else:
                self.segmentType = SegmentType.TRAVELLING_FULL
            self.isCycleEnd = True

    def updateLossSummary(self, bucket):
        """This function updates the loss summary of the segment when a new loss bucket is added to it

        :param bucket: loss bucket which belongs to the segment
        :type bucket: lossBucket
        """
        if bucket.actualASLRName in self.lossSummary.keys():
            self.lossSummary[bucket.actualASLRName].addBucketData(bucket)
        else:
            self.lossSummary[bucket.actualASLRName] = reasonSummary(bucket.actualASLR, bucket.actualASLRName,
                                                                    bucket.actualASLRCategory, self.segmentId)
            self.lossSummary[bucket.actualASLRName].addBucketData(bucket)

    def updateIndexes(self, index_increment):
        """This function to change the reference of the messages of the segment

        :param index_increment: value by which index of messages have to be changed
        :type index_increment: int
        """
        # self.startIndex += index_increment
        # self.endIndex += index_increment
        self.indexes = [i + index_increment for i in self.indexes]
        for bucket in self.lossBuckets:
            bucket.updateMessageIndexes(index_increment)

        # have to update loss summary also

    def updatePath(self, messages):
        """This function updates the path followed by truck during the segment during the
          segment using leftEdge and right edge points present in each message

        :param messages: list of messages of the segment
        :type messages: list
        """
        left_edge_points, right_edge_points, preferred_path_points = \
            list(map(list, zip(*[(msg.leftEdge, msg.rightEdge, (msg.pathEasting, msg.pathNorthing, msg.pathElevation))
                                 for msg in messages])))
        # self.path = Polygon(left_edge_points.extend(right_edge_points[::-1]))
        left_edge_points.extend(right_edge_points[::-1])
        self.path = left_edge_points
        self.preferredPath = preferred_path_points

    def updateTimeZone(self, timezone_offset):
        self.actualStartTime -= timezone_offset
        self.actualEndTime -= timezone_offset
        self.expectedStartTime -= timezone_offset
        self.expectedEndTime -= timezone_offset
        for bucket in self.lossBuckets:
            bucket.actualStartTime -= timezone_offset
            bucket.actualEndTime -= timezone_offset
            bucket.expectedStartTime -= timezone_offset
            bucket.expectedEndTime -= timezone_offset
        # for reason_summary in self.lossSummary.values():
        #     reason_summary.buckets = [self.lossBuckets[i] for i in reason_summary.children.keys()]

    def iter_loss_summary_records(self) -> Iterator[Dict[str, Any]]:
        """Yield the loss summary records for the segment grouped by ASLR."""

        if not self.lossSummary:
            return

        segment_type_value: Optional[int]
        if self.segmentType is None:
            segment_type_value = None
        else:
            segment_type_value = int(self.segmentType)

        segment_start: Optional[datetime] = self.actualStartTime
        if segment_start is None:
            return

        segment_end: Optional[datetime] = self.actualEndTime
        if segment_end is None:
            return

        for summary in self.lossSummary.values():
            if summary.loss is None:
                summary.updateMapProperties()

            aslr_value = summary.actualAslr if summary.actualAslr is not None else -1

            yield {
                "machineId": self.machineId,
                "segmentId": self.segmentId,
                "cycleId": self.cycleId,
                "segmentStartTime": segment_start,
                "segmentEndTime": segment_end,
                "aslr": aslr_value,
                "loss": float(summary.loss or 0.0) / 3600,
                "actualTimeTaken": float(summary.actualTimeTaken or 0.0),
                "expectedTimeTaken": float(summary.expectedTimeTaken or 0.0),
                "totalCount": int(summary.count or 0),
                "segmentType": segment_type_value,
            }
