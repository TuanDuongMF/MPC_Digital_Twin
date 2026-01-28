import numpy as np


class Tile:
    def __init__(self, x_id, y_id, tile_length):
        self.XId = x_id
        self.YId = y_id
        self.center = ((x_id * tile_length) + (tile_length / 2), (y_id * tile_length) + (tile_length / 2))
        self.messagesCount = None
        self.efficiencyMean = None
        self.efficiencyMax = None
        self.efficiencyMin = None
        self.actualSpeedMean = None
        self.actualSpeedMax = None
        self.actualSpeedMin = None
        self.expectedSpeedMean = None
        self.expectedSpeedMax = None
        self.expectedSpeedMin = None
        self.tileSummary = {}
        self.uniqueReasons = None
        self.cycles = []
        self.machines = []
        self.messageIndexes = {}

    def updateTileData(self, messages):
        tile_data = np.array([(msg.efficiency, msg.actualSpeed, msg.expectedSpeed, msg.actualASLR) for msg in messages
                              if msg.efficiency is not None])

        if len(tile_data) > 0:
            self.messagesCount = len(tile_data)
            tile_mean = np.mean(tile_data[:, 0:3], axis=0)
            tile_max = np.max(tile_data[:, 0:3], axis=0)
            tile_min = np.min(tile_data[:, 0:3], axis=0)

            self.efficiencyMean, self.actualSpeedMean, self.expectedSpeedMean = tile_mean[0], tile_mean[1], tile_mean[2]
            self.efficiencyMax, self.actualSpeedMax, self.expectedSpeedMax = tile_max[0], tile_max[1], tile_max[2]
            self.efficiencyMin, self.actualSpeedMin, self.expectedSpeedMin = tile_min[0], tile_min[1], tile_min[2]

            self.uniqueReasons, indices, counts = np.unique(tile_data[:, 3], return_inverse=True,
                                                            return_counts=True)
            reasons_data = [tile_data[indices == k, :3] for k in range(len(self.uniqueReasons))]
            for reason, count, reason_data in zip(self.uniqueReasons, counts, reasons_data):
                reason_summary = ReasonSummary(reason, count)
                reason_summary.updateReasonSummaryData(reason_data)
                self.tileSummary[reason] = reason_summary



    def getTileData(self):
        return (self.messagesCount, self.efficiencyMean * self.messagesCount, self.actualSpeedMean * self.messagesCount,
                self.expectedSpeedMean * self.messagesCount, self.efficiencyMax, self.actualSpeedMax,
                self.expectedSpeedMax, self.efficiencyMin, self.actualSpeedMin, self.expectedSpeedMin,
                self.tileSummary, self.cycles, self.machines, self.messageIndexes)

    @staticmethod
    def addTiles(x_id, y_id, tile_length, tiles):
        new_tile = Tile(x_id, y_id, tile_length)
        # total_msgs, eff_sum, eff_min, eff_max, act_speed_sum, act_speed
        tile_data = [tile.getTileData() for tile in tiles]
        msgs_count, eff_sum, act_speed_sum, exp_speed_sum, eff_max, act_speed_max, exp_speed_max, eff_min,\
        act_speed_min, exp_speed_min, tile_summaries, cycles, machines, indexes = list(map(list, zip(*tile_data)))

        new_tile.messagesCount = sum(msgs_count)
        new_tile.efficiencyMean, new_tile.actualSpeedMean, new_tile.expectedSpeedMean = \
            sum(eff_sum) / new_tile.messagesCount, sum(act_speed_sum) / new_tile.messagesCount,\
            sum(exp_speed_sum) / new_tile.messagesCount
        new_tile.efficiencyMax, new_tile.actualSpeedMax, new_tile.expectedSpeedMax = \
            max(eff_max), max(act_speed_max), max(exp_speed_max)
        new_tile.efficiencyMin, new_tile.actualSpeedMin, new_tile.expectedSpeedMin = \
            min(eff_min), min(act_speed_min), min(exp_speed_min)

        reasons_data = {}
        [reasons_data.setdefault(reason, []).append(reason_summary)
         for tile_summary in tile_summaries for reason, reason_summary in tile_summary.items()]

        new_tile.tileSummary = {reason: ReasonSummary.addReasonSummaries(reason, reason_summaries)
                                for reason, reason_summaries in reasons_data.items()}

        [new_tile.messageIndexes.setdefault(machine, []).extend(index_list)
         for index_dict in indexes for machine, index_list in index_dict.items()]

        new_tile.uniqueReasons = list(new_tile.tileSummary.keys())
        new_tile.cycles = list(set([cycle for cycle_list in cycles for cycle in cycle_list]))
        new_tile.machines = list(set([machine for machine_list in machines for machine in machine_list]))
        return new_tile


class ReasonSummary:
    def __init__(self, reason, count):
        self.reason = reason
        self.messagesCount = count
        self.efficiencyMean = None
        self.efficiencyMax = None
        self.efficiencyMin = None
        self.actualSpeedMean = None
        self.actualSpeedMax = None
        self.actualSpeedMin = None
        self.expectedSpeedMean = None
        self.expectedSpeedMax = None
        self.expectedSpeedMin = None

    def updateReasonSummaryData(self, reason_data):
        reason_mean = np.mean(reason_data[:, 0:3], axis=0)
        reason_max = np.max(reason_data[:, 0:3], axis=0)
        reason_min = np.min(reason_data[:, 0:3], axis=0)

        self.efficiencyMean, self.actualSpeedMean, self.expectedSpeedMean = reason_mean[0], reason_mean[1], reason_mean[2]
        self.efficiencyMax, self.actualSpeedMax, self.expectedSpeedMax = reason_max[0], reason_max[1], reason_max[2]
        self.efficiencyMin, self.actualSpeedMin, self.expectedSpeedMin = reason_min[0], reason_min[1], reason_min[2]

    def getReasonSummaryData(self):
        return (self.messagesCount, self.efficiencyMean * self.messagesCount, self.actualSpeedMean * self.messagesCount,
                self.expectedSpeedMean * self.messagesCount, self.efficiencyMax, self.actualSpeedMax,
                self.expectedSpeedMax, self.efficiencyMin, self.actualSpeedMin, self.expectedSpeedMin)

    @staticmethod
    def addReasonSummaries(reason, reason_summaries):
        reason_summary_data = [reason_summary.getReasonSummaryData() for reason_summary in reason_summaries]
        msgs_count, eff_sum, act_speed_sum, exp_speed_sum, eff_max, act_speed_max, exp_speed_max, eff_min, \
        act_speed_min, exp_speed_min = list(map(list, zip(*reason_summary_data)))

        new_reason_summary = ReasonSummary(reason, sum(msgs_count))

        new_reason_summary.efficiencyMean, new_reason_summary.actualSpeedMean, new_reason_summary.expectedSpeedMean = \
            sum(eff_sum) / sum(msgs_count), sum(act_speed_sum) / sum(msgs_count), sum(exp_speed_sum) / sum(msgs_count)
        new_reason_summary.efficiencyMax, new_reason_summary.actualSpeedMax, new_reason_summary.expectedSpeedMax = \
            max(eff_max), max(act_speed_max), max(exp_speed_max)
        new_reason_summary.efficiencyMin, new_reason_summary.actualSpeedMin, new_reason_summary.expectedSpeedMin = \
            min(eff_min), min(act_speed_min), min(exp_speed_min)

        return new_reason_summary

