"""Traffic statistics entity."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
from domain.entities.vehicle import VehicleType
from domain.entities.direction import Direction
from domain.entities.vehicle_size import VehicleSize


@dataclass
class FrameStatistics:
    """Statistics for a single frame."""
    frame_number: int
    timestamp: float
    vehicle_count: int
    vehicles_by_type: Dict[VehicleType, int] = field(default_factory=dict)
    density: float = 0.0  # density as percentage: (∑(vehicle areas) / ROI area) × 100
    tracked_vehicles_count: int = 0  # Number of tracked vehicles in this frame
    speed_violations_count: int = 0  # Number of speed violations in this frame
    vehicles_by_direction: Dict[Direction, int] = field(default_factory=dict)  # Vehicles by direction
    vehicles_by_size: Dict[VehicleSize, int] = field(default_factory=dict)  # Vehicles by size
    vehicles_by_lane: Dict[str, int] = field(default_factory=dict)  # Vehicles by lane (for multi-ROI)


@dataclass
class TrafficStatistics:
    """Overall traffic statistics."""
    total_vehicles: int = 0
    vehicles_by_type: Dict[VehicleType, int] = field(default_factory=dict)
    max_vehicles_in_frame: int = 0
    average_vehicles_per_frame: float = 0.0
    peak_density: float = 0.0
    frame_statistics: List[FrameStatistics] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    # Speed and direction statistics
    total_tracked_vehicles: int = 0
    total_speed_violations: int = 0
    vehicles_by_direction: Dict[Direction, int] = field(default_factory=dict)
    average_speed_kmh: float = 0.0
    max_speed_kmh: float = 0.0
    speed_statistics_by_direction: Dict[Direction, Dict[str, float]] = field(default_factory=dict)
    # Advanced features
    vehicles_by_size: Dict[VehicleSize, int] = field(default_factory=dict)
    vehicles_by_lane: Dict[str, int] = field(default_factory=dict)  # For multi-ROI

