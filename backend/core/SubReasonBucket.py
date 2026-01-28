class SubReasonBucket:
    def __init__(self, sub_reason_info, actual_start_time, expected_start_time):
        self.subReasonInfo = sub_reason_info
        self.actualStartTime, self.expectedStartTime = actual_start_time, expected_start_time
        self.actualEndTime, self.expectedEndTime = None, None
        self.actualTimeTaken = None
        self.expectedTimeTaken = None
        self.loss = None

    def updateEndTimes(self, actual_end_time, expected_end_time):
        self.actualEndTime, self.expectedEndTime = actual_end_time, expected_end_time

    def updateBucketProperties(self):
        self.actualTimeTaken = (self.actualEndTime - self.actualStartTime).total_seconds()
        self.expectedTimeTaken = (self.expectedEndTime - self.expectedStartTime).total_seconds()
        self.loss = self.actualTimeTaken - self.expectedTimeTaken

    def to_dict(self):
        out_dict = self.subReasonInfo | {
            'actualTimeTaken': self.actualTimeTaken,
            'expectedTimeTaken': self.expectedTimeTaken,
            'loss': self.loss,
        }
        return out_dict
