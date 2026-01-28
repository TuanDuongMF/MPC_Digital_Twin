# set list of segments whose points can be considered as zone points
load_zone_segments = ['Spotting.At.Source', 'WaitForLoad', 'Loading']
dump_zone_segments = ['Spotting.At.Sink', 'WaitForDump', 'Dumping']

reasonMap = {
    0:
        {
            "Reason": "AssignmentLimit",
            "Description": "Machine has no assignment to execute.",
            "Id": 0,
            "Category": "Operations-Office",
            "color": (230, 25, 75),
            "Color": "RGBA(230, 25, 75, 0.25)"
        },
    1:
        {
            "Reason": "AvoidDrivingSamePathArea",
            "Description": "Machine is in an area where it must avoid driving the same path.",
            "Id": 1,
            "Category": "Non-Diagnostic Limited",
            "color": (60, 180, 75),
            "Color": "RGBA(60, 180, 75, 0.25)"
        },
    2:
        {
            "Reason": "AStop",
            "Description": "AStop Machine has detected an active A-Stop.",
            "Id": 2,
            "Category": "Non-Diagnostic Limited",
            "color": (255, 225, 25),
            "Color": "RGBA(255, 225, 25, 0.25)"
        },
    3:
        {
            "Reason": "BaseMachineSpeedLimit",
            "Description": "Base machine is limiting the speed of the truck.",
            "Id": 3,
            "Category": "Machine Limited",
            "color": (67, 99, 216),
            "Color": "RGBA(67, 99, 216, 0.25)"
        },
    4:
        {
            "Reason": "BedDownArea",
            "Description": "Machine is slowing or stopped for a bed down area.",
            "Id": 4,
            "Category": "Operations-Office",
            "color": (245, 130, 49),
            "Color": "RGBA(245, 130, 49, 0.25)"
        },
    5:
        {
            "Reason": "CurvatureLimit",
            "Description": "Machine is speed limited due to the curvature of the path.",
            "Id": 5,
            "Category": "Machine Limited",
            "color": (145, 30, 180),
            "Color": "RGBA(145, 30, 180, 0.25)"
        },
    6:
        {
            "Reason": "ExclusionArea",
            "Description": "Machine is speed limited due to an exclusion area.",
            "Id": 6,
            "Category": "Operations-Office",
            "color": (70, 240, 240),
            "Color": "RGBA(70, 240, 240, 0.25)"
        },
    7:
        {
            "Reason": "FollowingSiteAwareVehicle",
            "Description": "Machine is slowing or has stopped for a site awareness machine it is following.",
            "Id": 7,
            "Category": "Non-Diagnostic Limited",
            "color": (240, 50, 230),
            "Color": "RGBA(240, 50, 230, 0.25)"
        },
    8:
        {
            "Reason": "GlobalSpeedLimit",
            "Description": "Machine is following a global speed limit.",
            "Id": 8,
            "Category": "Operations-Office",
            "color": (188, 246, 12),
            "Color": "RGBA(188, 246, 12, 0.25)"
        },
    9:
        {
            "Reason": "GlobalTractionLimit",
            "Description": "Machine is speed limited due to a global traction limit.",
            "Id": 9,
            "Category": "Operations-Office",
            "color": (250, 190, 190),
            "Color": "RGBA (250, 190, 190, 0.25)"
        },
    10:
        {
            "Reason": "GradeLimit",
            "Description": "Machine is speed limited due the the grade of the path.",
            "Id": 10,
            "Category": "Machine Limited",
            "color": (0, 128, 128),
            "Color": "RGBA(0, 128, 128, 0.25)"
        },
    11:
        {
            "Reason": "HealthEvent",
            "Description": "Machine has an active health event.",
            "Id": 11,
            "Category": "Dignostic Limited",
            "color": (230, 190, 255),
            "Color": "RGBA(230, 190, 255, 0.25)"
        },
    12:
        {
            "Reason": "HighRollingResistanceArea",
            "Description": "Machine is speed limited due to a high rolling resistance area.",
            "Id": 12,
            "Category": "Non-Diagnostic Limited",
            "color": (60, 180, 75),
            "Color": "RGBA(154, 99, 36, 0.25)"
        },
    13:
        {
            "Reason": "InCyclePlannerCheckout",
            "Description": "Machine is currently verifying planning related capabilities.",
            "Id": 13,
            "Category": "Service Checkout",
            "color": (255, 250, 200),
            "Color": "RGBA(255, 250, 200, 0.25)"
        },
    14:
        {
            "Reason": "LaneEdgeLimit",
            "Description": "Machine is too close to the lane edge.",
            "Id": 14,
            "Category": "Non-Diagnostic Limited",
            "color": (128, 0, 0),
            "Color": "RGBA(128, 0, 0, 0.25)"
        },
    15:
        {
            "Reason": "LaneSpeedLimit",
            "Description": "Machine is following a lane speed limit.",
            "Id": 15,
            "Category": "Drivable Region Geometry",
            "color": (170, 255, 195),
            "Color": "RGBA(170, 255, 195, 0.25)"
        },
    16:
        {
            "Reason": "LowTractionDetectedArea",
            "Description": "Machine is speed limited due to a poor traction area detected by an autonomous machine.",
            "Id": 16,
            "Category": "Traction Feature Derate",
            "color": (128, 128, 0),
            "Color": "RGBA(128, 128, 0, 0.25)"
        },
    17:
        {
            "Reason": "MineModelIncorrect",
            "Description": "The mine model does not match the physical world detected by the machine.",
            "Id": 17,
            "Category": "Drivable Region Geometry",
            "color": (255, 216, 177),
            "Color": "RGBA(255, 216, 177, 0.25)"
        },
    18:
        {
            "Reason": "MineModelVerificationLimit",
            "Description": "Machine is speed limited while verifying a mine model element.",
            "Id": 18,
            "Category": "Non-Diagnostic Limited",
            "color": (0, 0, 117),
            "Color": "RGBA(0, 0, 117, 0.25)"
        },
    19:
        {
            "Reason": "NarrowLaneLimit",
            "Description": "Machine is speed limited as lane width does not support full speed operation.",
            "Id": 19,
            "Category": "Drivable Region Geometry",
            "color": (128, 128, 128),
            "Color": "RGBA(128, 128, 128, 0.25)"
        },
    20:
        {
            "Reason": "ObstacleDetected",
            "Description": "Machine is slowing or stopping for an obstacle detected in its path.",
            "Id": 20,
            "Category": "External Disturbance",
            "color": (139, 69, 19),
            "Color": "RGBA(139, 69, 19, 0.25)"
        },
    21:
        {
            "Reason": "PermissionLimit",
            "Description": "Machine is nearing the end of its assignment or permissions.",
            "Id": 21,
            "Category": "Operations-Office",
            "color": (176, 224, 230),
            "Color": "RGBA(176, 224, 230, 0.25)"
        },
    22:
        {
            "Reason": "RoughTerrainObjectDetectionArea",
            "Description": "Machine is speed limited due to a rough terrain area.",
            "Id": 22,
            "Category": "Non-Diagnostic Limited",
            "color": (169, 169, 169),
            "Color": "RGBA(169, 169, 169, 0.25)"
        },
    23:
        {
            "Reason": "ServiceTest",
            "Description": "Machine is performing a service test or calibration.",
            "Id": 23,
            "Category": "Service Checkout",
            "color": (255, 105, 180),
            "Color": "RGBA(255, 105, 180, 0.25)"
        },
    24:
        {
            "Reason": "SiteAwarenessPathPredictionLimit",
            "Description": "Machine is speed limited due to a path prediction of another site aware machine.",
            "Id": 24,
            "Category": "Non-Diagnostic Limited",
            "color": (138, 43, 226),
            "Color": "RGBA(138, 43, 226, 0.25)"
        },
    25:
        {
            "Reason": "SpeedControlSoftUnderfootArea",
            "Description": "Machine is in an area designated as soft underfoot with speed controls.",
            "Id": 25,
            "Category": "Non-Diagnostic Limited",
            "color": (255, 69, 0),
            "Color": "RGBA(255, 69, 0, 0.25)"
        },
    26:
        {
            "Reason": "SpeedLimitArea",
            "Description": "Machine is following an area speed limit.",
            "Id": 26,
            "Category": "Operations-Office",
            "color": (46, 139, 87),
            "Color": "RGBA(46, 139, 87, 0.25)"
        },
    27:
        {
            "Reason": "TractionArea",
            "Description": "Machine is speed limited due to a traction area.",
            "Id": 27,
            "Category": "Operations-Office",
            "color": (255, 99, 71),
            "Color": "RGBA(255, 99, 71, 0.25)"
        },
    28:
        {
            "Reason": "Visibility",
            "Description": "Machine is speed limited due to limited visibility of on-board perception systems in the "
                           "current conditions.",
            "Id": 28,
            "Category": "Non-Diagnostic Limited",
            "color": (70, 130, 180),
            "Color": "RGBA(70, 130, 180, 0.25)"
        },
    29:
        {
            "Reason": "WaterTruckPathLimit",
            "Description": "Machine is speed limited due to a recent traversal of a water truck.",
            "Id": 29,
            "Category": "External Disturbance",
            "color": (221, 160, 221),
            "Color": "RGBA(221, 160, 221, 0.25)"
        },
    30:
        {
            "Reason": "CurvatureRateOfChange",
            "Description": "Machine is slowing due to the rate of change in curvature within the lane.",
            "Id": 30,
            "Category": "Drivable Region Geometry",
            "color": (0, 255, 127),
            "Color": "RGBA(0, 255, 127, 0.25)"
        },
    31:
        {
            "Reason": "UnfavourableBank",
            "Description": "Machine is being slowed due to an unfavourable bank in the road.",
            "Id": 31,
            "Category": "Drivable Region Geometry",
            "color": (255, 20, 147),
            "Color": "RGBA(255, 20, 147, 0.25)"
        },
    32:
        {
            "Reason": "InCyclePerceptionVerificationLimit",
            "Description": "Machine is being slowed by in cycle perception verification limit (AODV limit).",
            "Id": 32,
            "Category": "Service Checkout",
            "color": (0, 206, 209),
            "Color": "RGBA(0, 206, 209, 0.25)"
        },
    33:
        {
            "Reason": "OfflineBrakingVerificationLimit",
            "Description": "Machine is being slowed by offline Braking verification limit.",
            "Id": 33,
            "Category": "Service Checkout",
            "color": (255, 0, 255),
            "Color": "RGBA(255, 0, 255, 0.25)"
        },
    34:
        {
            "Reason": "OfflinePerceptionVerificationLimit",
            "Description": "Machine is being slowed by offline perception verification limit.",
            "Id": 34,
            "Category": "Service Checkout",
            "color": (30, 144, 255),
            "Color": "RGBA(30, 144, 255, 0.25)"
        },
    35:
        {
            "Reason": "BrakeHoldTestInProgress",
            "Description": "Machine is being slowed by service brake hold test.",
            "Id": 35,
            "Category": "Service Checkout",
            "color": (255, 182, 193),
            "Color": "RGBA(255, 182, 193, 0.25)"
        },
    254:
        {
            "Reason": "Queuing",
            "Description": "Machine is slowing or has stopped due to queuing.",
            "Id": 254,
            "Category": "Non-Diagnostic Limited",
            "color": (32, 178, 170),
            "Color": "RGBA(32, 178, 170, 0.25)"
        },
    255:
        {
            "Reason": "NO SPEED LIMIT REASON",
            "Description": "When ASLR sends an empty message",
            "Id": 255,
            "Category": "NONE",
            "color": (255, 222, 173),
            "Color": "RGBA(255, 222, 173, 0.25)"
        }
}
