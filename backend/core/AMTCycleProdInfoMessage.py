import datetime
import math
import numpy as np
from datetime import timedelta
from .readerConfig import reasonMap
from .constants import gps_epoch, leap_seconds


class AMTCycleProdInfoMessage:
    """This class represents the cycle prod message"""

    def __init__(self, data, machine_name, machine_type, message_type="CP1"):
        """constructor

        :param data: values of attributes of the message fetched
        :type data: tuple/series
        :param machine_name: machine name to which message belongs to
        :type machine_name: string
        :param machine_type: type of the machine to which message belongs to
        :type machine_type: string
        """
        if len(data) == 27:
            # Data from to_array() method - deserialize
            self.machineId = None  # Not included in to_array()
            self.machineName = machine_name
            self.machineType = machine_type
            self.start_time = datetime.datetime.fromtimestamp(data[0], datetime.timezone.utc)
            self.segmentId = data[1]
            self.expectedElapsedTime = float(data[2])
            self.expectedTimeGPS = float(data[3])
            self.expectedTime = datetime.datetime.fromtimestamp(data[4], datetime.timezone.utc)
            self.actualElapsedTime = float(data[5])
            self.actualTimeGPS = float(data[6])
            self.actualTime = datetime.datetime.fromtimestamp(data[7], datetime.timezone.utc)
            self.pathEasting = float(data[8])
            self.pathNorthing = float(data[9])
            self.pathElevation = float(data[10])
            self.plannedDistance = float(data[11])
            self.expectedSpeed = float(data[12])
            self.actualSpeed = float(data[13])
            self.expectedDesiredSpeed = float(data[14])
            self.actualDesiredSpeed = float(data[15])
            self.leftWidth = float(data[16])
            self.rightWidth = float(data[17])
            self.pathBank = float(data[18])
            self.pathHeading = float(data[19])
            self.payloadPercent = data[20] if data[20] <= 200 else data[20] - 255
            self.expectedSpeedSource = data[21]
            self.expectedASLR = data[22]
            self.expectedRegModEnum = data[23]
            self.actualSpeedSource = data[24]
            self.actualASLR = data[25]
            self.actualRegModEnum = data[26]
            self.cycleDistance = float(0)  # Not included in to_array()
        else:
            self.machineId = data[0]
            self.machineName = machine_name
            self.machineType = machine_type
            self.segmentId = data[1]
            self.start_time = datetime.datetime.fromisoformat(data[2]).replace(tzinfo=datetime.timezone.utc) if isinstance(data[2], str) else datetime.datetime.fromtimestamp(data[2], datetime.timezone.utc)
            self.expectedElapsedTime = float(data[3])
            self.expectedTimeGPS = self.segmentId + self.expectedElapsedTime
            self.expectedTime = gps_epoch + timedelta(seconds=self.expectedTimeGPS) - leap_seconds
            self.actualElapsedTime = float(data[4])
            self.actualTimeGPS = self.segmentId + self.actualElapsedTime
            self.actualTime = gps_epoch + timedelta(seconds=self.actualTimeGPS) - leap_seconds
            self.pathEasting = float(data[5])
            self.pathNorthing = float(data[6])
            self.pathElevation = float(data[7])
            self.plannedDistance = float(data[8])
            self.expectedSpeed = float(data[9])
            self.actualSpeed = float(data[10])
            self.expectedDesiredSpeed = float(data[11])
            self.actualDesiredSpeed = float(data[12])
            self.leftWidth = float(data[13])
            self.rightWidth = float(data[14])
            self.pathBank = float(data[15])
            self.pathHeading = float(data[16])
            self.payloadPercent = data[17] if data[17] <= 200 else data[17] - 255
            self.expectedSpeedSource = data[18]
            self.expectedASLR = data[19]
            self.expectedRegModEnum = data[20]
            self.actualSpeedSource = data[21]
            self.actualASLR = data[22]
            self.actualRegModEnum = data[23]

            self.cycleDistance = float(data[24]) if len(data) > 24 else float(0)

        self.actualASLRName = reasonMap[self.actualASLR]["Reason"]
        self.expectedASLRName = reasonMap[self.expectedASLR]["Reason"]
        self.actualASLRCategory = reasonMap[self.actualASLR]["Category"]
        self.expectedASLRCategory = reasonMap[self.expectedASLR]["Category"]

        self.efficiency = None
        self.painIndex = None
        self.bucketLoss = None
        self.cycleId = None if message_type == "CP1" else data[1]

        self.leftEdge = None
        self.rightEdge = None
        self.updateEdgePoints()

    def updateEdgePoints(self):
        """This function calculates the coordinates the left edge and right edge points using coordinates of preferred
        path points, path heading, leftWidth and right width
        """
        heading_angle = math.radians(self.pathHeading)
        bank_angle = np.arctan(self.pathBank)

        # direction vector
        direction_vector = np.array([np.cos(heading_angle), np.sin(heading_angle), 0])

        # normal vector
        normal_vector = np.cross(direction_vector, np.array([0, 0, 1]))

        # adjusting normal vector based on banking
        normal_rotated = normal_vector * np.cos(bank_angle) + np.cross(np.array([0, 0, 1]), normal_vector) * np.sin(bank_angle)

        center_point = np.array([self.pathEasting, self.pathNorthing, self.pathElevation])
        self.leftEdge = tuple(center_point + normal_rotated * self.leftWidth)
        self.rightEdge = tuple(center_point - normal_rotated * self.rightWidth)

    def updateQueuing(self):
        """This function updates the aslr information from following site awareness to queuing if the message
        occurred in queuing area
        """
        if self.actualASLR == 7:
            self.actualASLR = 254
            self.actualASLRName = reasonMap[self.actualASLR]["Reason"]
            self.actualASLRCategory = reasonMap[self.actualASLR]["Category"]

    def to_array(self):
        """Returns CycleInfoMessage Object as a list of values with scaled integers"""

        return [
            self.start_time.replace(tzinfo=datetime.timezone.utc).timestamp(),
            self.segmentId,
            self.expectedElapsedTime,
            self.expectedTimeGPS,
            self.expectedTime.replace(tzinfo=datetime.timezone.utc).timestamp(),
            self.actualElapsedTime,
            self.actualTimeGPS,
            self.actualTime.replace(tzinfo=datetime.timezone.utc).timestamp(),
            self.pathEasting,
            self.pathNorthing,
            self.pathElevation,
            self.plannedDistance,
            self.expectedSpeed,
            self.actualSpeed,
            self.expectedDesiredSpeed,
            self.actualDesiredSpeed,
            self.leftWidth,
            self.rightWidth,
            self.pathBank,
            self.pathHeading,
            self.payloadPercent,
            self.expectedSpeedSource,
            self.expectedASLR,
            self.expectedRegModEnum,
            self.actualSpeedSource,
            self.actualASLR,
            self.actualRegModEnum,
        ]

    @staticmethod
    def array_fields() -> list[str]:
        """Returns list of field names corresponding to the array indices in to_array()"""
        return [
            "start_time",
            "segmentId",
            "expectedElapsedTime",
            "expectedTimeGPS",
            "expectedTime",
            "actualElapsedTime",
            "actualTimeGPS",
            "actualTime",
            "pathEasting",
            "pathNorthing",
            "pathElevation",
            "plannedDistance",
            "expectedSpeed",
            "actualSpeed",
            "expectedDesiredSpeed",
            "actualDesiredSpeed",
            "leftWidth",
            "rightWidth",
            "pathBank",
            "pathHeading",
            "payloadPercent",
            "expectedSpeedSource",
            "expectedASLR",
            "expectedRegModEnum",
            "actualSpeedSource",
            "actualASLR",
            "actualRegModEnum",
        ]
