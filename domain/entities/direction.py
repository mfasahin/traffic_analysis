"""Direction enumeration for vehicle movement."""

from enum import Enum


class Direction(Enum):
    """Vehicle movement direction."""
    UP = "up"          # Moving upward (y decreasing)
    DOWN = "down"      # Moving downward (y increasing)
    LEFT = "left"      # Moving left (x decreasing)
    RIGHT = "right"    # Moving right (x increasing)
    UNKNOWN = "unknown"  # Direction not determined yet
    STATIONARY = "stationary"  # Vehicle not moving

