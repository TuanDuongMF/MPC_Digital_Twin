from .MapClasses import reasonSummary


class Route:
    def __init__(self, source, destination, segment_class):
        self.source = source
        self.destination = destination
        self.segmentClass = segment_class
        self.averageEfficiency = 0
        self.totalLoss = 0
        self.laps = {}
        self.uniqueReasons = set()
        self.totalLossSummary = {}
        self.machineWiseLossSummary = {}
        self.lapCount = 0
        self.bestLapActualTime = float('inf')
        self.bestLapExpectedTime = float('inf')

    def addLap(self, lap):
        self.laps.setdefault(lap.machineName, []).append(lap)
        self.uniqueReasons.update(lap.uniqueReasons)
        self.totalLoss += lap.loss
        self.averageEfficiency = ((self.averageEfficiency * self.lapCount) + lap.efficiency) / (self.lapCount + 1)
        self.lapCount += 1
        self.bestLapActualTime = self.bestLapActualTime if self.bestLapActualTime < lap.actualTimeTaken else \
            lap.actualTimeTaken
        self.bestLapExpectedTime = self.bestLapExpectedTime if self.bestLapExpectedTime < lap.expectedTimeTaken else \
            lap.expectedTimeTaken

    def calculateRouteProperties(self):
        # summary_list = [lap.lossSummary for lap in self.laps]
        # self.lossSummary = reasonSummary.addSummaries(summary_list, self.uniqueReasons, 'cycles',
        #                                               f"{self.source}-{self.destination}")
        # for reason_summary in self.lossSummary.values():
        #     reason_summary.updateMapProperties()
        machine_wise_summary_list = {machine: [lap.lossSummary for lap in machine_laps]
                                     for machine, machine_laps in self.laps.items()}
        total_summary_list = []
        for machine, machine_summaries in machine_wise_summary_list.items():
            self.machineWiseLossSummary[machine] = reasonSummary.addSummaries(machine_summaries, self.uniqueReasons,
                                                                              'cycles', machine)
            for reason_summary in self.machineWiseLossSummary[machine].values():
                reason_summary.updateMapProperties()

            total_summary_list.extend(machine_summaries)

        self.totalLossSummary = reasonSummary.addSummaries(total_summary_list, self.uniqueReasons, 'cycles',
                                                           f"{self.source}-{self.destination}")
        for reason_summary in self.totalLossSummary.values():
            reason_summary.updateMapProperties()

    def updateTimeZone(self, time_offset):
        for laps_list in self.laps.values():
            for lap in laps_list:
                lap.actualStartTime -= time_offset
                lap.expectedStartTime -= time_offset
                lap.actualEndTime -= time_offset
                lap.expectedEndTime -= time_offset

    # def getRouteProperties(self):
    #     efficiencies, losses, actual_time_taken, expected_time_taken = list(zip(*[
    #         (lap.efficiency, lap.loss, lap.actualTimeTaken, lap.expectedTimeTaken) for lap in self.laps]))
    #     lap_count = len(self.laps)
    #     return efficiencies, losses, actual_time_taken, expected_time_taken, lap_count

    # def getLapPaths(self, messages):
    #     return [[(messages[lap.machineId][i].pathEasting, messages[lap.machineId][i].pathNorthing)
    #              for i in lap.indexes] for lap in self.laps]


class Lap:
    def __init__(self, machine_id, machine_name, cycle_id, segments):
        first_segment, last_segment = segments[0], segments[-1]
        self.cycleId = cycle_id
        self.machineId = machine_id
        self.machineName = machine_name
        self.indexes = []
        self.actualStartTime = first_segment.actualStartTime
        self.expectedStartTime = first_segment.expectedStartTime
        self.actualEndTime = last_segment.actualEndTime
        self.expectedEndTime = last_segment.expectedEndTime
        self.actualTimeTaken = 0
        self.expectedTimeTaken = 0
        summary_list, unique_reasons = [], []
        for segment in segments:
            self.indexes.extend(segment.indexes)
            self.actualTimeTaken += (segment.actualEndTime - segment.actualStartTime).seconds
            self.expectedTimeTaken += (segment.expectedEndTime - segment.expectedStartTime).seconds
            summary_list.append(segment.lossSummary)
            unique_reasons.extend(segment.lossSummary.keys())
        self.loss = self.actualTimeTaken - self.expectedTimeTaken
        self.efficiency = (self.expectedTimeTaken / self.actualTimeTaken) * 100
        self.uniqueReasons = set(unique_reasons)
        self.lossSummary = reasonSummary.addSummaries(summary_list, self.uniqueReasons, 'segments', self.cycleId)
        for reason_summary in self.lossSummary.values():
            reason_summary.updateMapProperties()
        self.segments = segments

#
# class Lap:
#     def __init__(self, first_msg, last_msg, indexes):
#         self.cycleId = first_msg.cycleId
#         self.machineId = first_msg.machineId
#         self.machineName = first_msg.machineName
#         self.indexes = indexes
#         self.actualStartTime = first_msg.actualTime
#         self.expectedStartTime = first_msg.expectedTime
#         self.actualEndTime = last_msg.actualTime
#         self.expectedEndTime = last_msg.expectedTime
#         self.actualTimeTaken = (self.actualEndTime - self.actualStartTime).seconds
#         self.expectedTimeTaken = (self.expectedEndTime - self.expectedStartTime).seconds
#         self.loss = self.actualTimeTaken - self.expectedTimeTaken
#         self.efficiency = (self.expectedTimeTaken / self.actualTimeTaken) * 100
#         self.distanceTravelled = last_msg.cycleDistance - last_msg.cycleDistance
