import numpy as np

from .SubReasonBucket import SubReasonBucket


class lossBucket:
    """This class represents a loss bucket. A loss bucket is a group of messages where the both actual aslr and
    the expected aslr are same continously for all messages.
    """

    def __init__(self, bucket_id: int, message=None):
        """constructor

        :param bucket_id: id of the bucket
        :type bucket_id: int
        :param message: first message of the bucket
        :type message: AMTCycleProdInfoMessage
        :param next_message: first message of next bucket
        :type next_message: AMTCycleProdInfoMessage
        :param start_index: index of first message of the bucket in Reader.messages
        :type start_index: int
        :param end_index: index of first message of the bucket in Reader.messages
        :type end_index: int
        """
        self.segmentId = message.segmentId
        self.bucket_id = bucket_id
        self.messageIndexes = []
        self.actualASLR, self.expectedASLR = (message.actualASLR, message.expectedASLR) if message else (
            'Unaccounted', 'Unaccounted')
        self.actualASLRName, self.expectedASLRName = message.actualASLRName, message.expectedASLRName if message else (
            'Unaccounted', 'Unaccounted')
        self.actualASLRCategory, self.expectedASLRCategory = message.actualASLRCategory, message.expectedASLRCategory if message else (
            'Unaccounted', 'Unaccounted')
        self.actualStartTime = None
        self.expectedStartTime = None
        self.actualEndTime = None
        self.expectedEndTime = None
        self.actualTimeTaken = None
        self.expectedTimeTaken = None
        self.bucketLoss = None
        self.isImpacting = None
        self.efficiency = None
        self.painIndex = None
        self.isPowerLimited = None
        self.avgActualSpeed = None
        self.avgActualDesiredSpeed = None
        self.distanceTravelled = None
        self.startCycleDistance = None
        self.endCycleDistance = None
        self.startPoint = None
        self.endPoint = None
        self.reasonInfo = []

    def updateBucketLossInfo(self, actual_start, expected_start, actual_end, expected_end, start_index, end_index,
                             start_distance, end_distance, start_point, end_point):
        """this function computes and updates various properties of loss bucket

        :param actual_start: actual start time of the loss bucket
        :type actual_start: datetime
        :param expected_start: expected start time of the loss bucket
        :type expected_start: datetime
        :param actual_end: actual end time of the loss bucket
        :type actual_end: _type_
        :param expected_end: expected end time of the loss bucket
        :type expected_end: _type_
        :param start_index: index of first message of loss bucket in messages property in reader object
        :type start_index: int
        :param end_index: index of last message of loss bucket in messages property in reader object
        :type end_index: int
        :param start_distance: planned distance first message of loss bucket
        :type end_index: int
        :param end_distance: planned distance last message of loss bucket
        :type end_index: int
        """
        self.messageIndexes = [i for i in range(start_index, end_index)] if end_index > start_index else [start_index]
        self.actualStartTime, self.expectedStartTime = actual_start, expected_start
        self.actualEndTime, self.expectedEndTime = actual_end, expected_end
        self.actualTimeTaken = (self.actualEndTime - self.actualStartTime).total_seconds()
        self.expectedTimeTaken = (self.expectedEndTime - self.expectedStartTime).total_seconds()

        self.distanceTravelled = end_distance - start_distance
        self.startCycleDistance = start_distance
        self.endCycleDistance = end_distance

        self.startPoint = start_point
        self.endPoint = end_point

        self.bucketLoss = self.actualTimeTaken - self.expectedTimeTaken
        self.isImpacting = (self.actualASLR != self.expectedASLR)
        self.efficiency = (self.expectedTimeTaken / (self.bucketLoss + self.expectedTimeTaken)) * 100 if \
            end_index - start_index >= 1 and (self.bucketLoss + self.expectedTimeTaken) != 0 else None
        self.painIndex = 1 - (self.efficiency / 100) if self.efficiency is not None else None

    def updateMessageIndexes(self, increment):
        """updates the indexes of all messages of loss based on incrememnt value

        :param increment: amount by which indexes of messages should be changed.
        :type increment: int
        """
        self.messageIndexes = [i + increment for i in self.messageIndexes]

    def  updateIsBucketPowerLimited(self, bucket_messages):
        """checks whether loss bucket is power limited or not using following logic
        Power limited If (bucket.avgActualDesiredSpeed - bucket.avgActualSpeed) / bucket.avgActualDesiredSpeed > 0.05

        (average is calculated by taking mean of speed of all messages in that bucket)

        :param bucket_messages: list of messages belongs to bucket
        :type bucket_messages: list
        """
        if bucket_messages:
            actual_speeds, desired_speeds = \
                list(map(list, zip(*[(msg.actualSpeed, msg.actualDesiredSpeed) for msg in bucket_messages])))
            self.avgActualSpeed, self.avgActualDesiredSpeed = np.mean(actual_speeds), np.mean(desired_speeds)
            self.isPowerLimited = True if self.avgActualDesiredSpeed and (
                    self.avgActualDesiredSpeed - self.avgActualSpeed) / self.avgActualDesiredSpeed > 0.05 else False

    def updateReasonInfo(self, bucket_messages):
        if bucket_messages:
            current_health_actions = {}
            for msg in bucket_messages:
                for health_action, health_action_info in msg.reasonInfo.items():
                    if current_health_actions.get(health_action):
                        current_health_actions[health_action].updateEndTimes(msg.actualTime, msg.expectedTime)
                    else:
                        current_health_actions[health_action] = SubReasonBucket(health_action_info, msg.actualTime, msg.expectedTime)

                for health_action, health_action_bucket in current_health_actions.items():
                    if health_action not in msg.reasonInfo.keys() and health_action_bucket:
                        current_health_actions[health_action].updateEndTimes(msg.actualTime, msg.expectedTime)
                        health_action_bucket.updateBucketProperties()
                        self.reasonInfo.append(health_action_bucket)
                        current_health_actions[health_action] = None

            for health_action, health_action_bucket in current_health_actions.items():
                if health_action_bucket:
                    health_action_bucket.updateEndTimes(self.actualEndTime, self.expectedEndTime)
                    health_action_bucket.updateBucketProperties()
                    self.reasonInfo.append(health_action_bucket)





    def to_dict(self):
        """this function returns the properties of the bucket as a dictionary

        :return: dictionary contain
        ing properties of bucket
        :rtype: dict
        """
        return {
            'segmentId': self.segmentId,
            'bucket_id': self.bucket_id,
            'messageIndexes': self.messageIndexes,
            'actualASLR': self.actualASLR,
            'expectedASLR': self.expectedASLR,
            'actualASLRName': self.actualASLRName,
            'expectedASLRName': self.expectedASLRName,
            'actualASLRCategory': self.actualASLRCategory,
            'expectedASLRCategory': self.expectedASLRCategory,
            'actualStartTime': self.actualStartTime,
            'expectedStartTime': self.expectedStartTime,
            'actualTimeTaken': self.actualTimeTaken,
            'expectedTimeTaken': self.expectedTimeTaken,
            'actualEndTime': self.actualEndTime,
            'expectedEndTime': self.expectedEndTime,
            'bucketLoss': self.bucketLoss,
            # 'distanceCovered': self.distanceCovered,
            'isImpacting': self.isImpacting,
            'efficiency': self.efficiency,
            'painIndex': self.painIndex,
        }

    def getHealthInfo(self):
        print('a')
        return {
            'level': self.reasonInfo['level'],
            'healthAction': self.reasonInfo['healthAction'],
            'bucketLoss': self.bucketLoss,
            'efficiency': self.efficiency,
            'Description':
                self.reasonInfo['healthActionInfo']['Description'] if self.reasonInfo['healthActionInfo'] else None,
            'MID_CID_FMI_WCI': str(self.reasonInfo['healthActionInfo']['MID']) + '_' +
                               str(self.reasonInfo['healthActionInfo']['CID_EID']) + '_' +
                               str(self.reasonInfo['healthActionInfo']['FMI']) + '_' +
                               str(self.reasonInfo['healthActionInfo']['WCI']) if
            self.reasonInfo['healthActionInfo'] else None,
        }

