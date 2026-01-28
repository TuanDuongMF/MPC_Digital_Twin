"""
Constants for AMT Cycle Productivity Message Reader

This module defines enums and constants used throughout the application.
"""

from enum import Enum
from datetime import datetime, timedelta, timezone


class SegmentType(Enum):
    """Segment type enumeration."""
    SPOTTING_AT_SOURCE = "Spotting.At.Source"
    SPOTTING_AT_SINK = "Spotting.At.Sink"
    TRAVELLING_EMPTY = "Travelling.Empty"
    TRAVELLING_FULL = "Travelling.Full"


class ZoneType(Enum):
    """Zone type enumeration."""
    LOAD = "LOAD"
    DUMP = "DUMP"


class SegmentClass(Enum):
    """Segment class enumeration."""
    EMPTY = "EMPTY"
    FULL = "FULL"


# GPS time constants
GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)  # GPS time starts Jan 6, 1980
LEAP_SECONDS = timedelta(seconds=18)  # GPS-UTC offset (as of 2024)

# Aliases for backward compatibility
gps_epoch = GPS_EPOCH
leap_seconds = LEAP_SECONDS
