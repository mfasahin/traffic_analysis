"""Traffic statistics entity."""

from dataclasses import dataclass, field
from typing import Dict, Optional
from datetime import datetime
from domain.entities.vehicle import VehicleType


@dataclass
class FrameStatistics:
    """Statistics for a single frame."""
    frame_number: int
    timestamp: float
    vehicle_count: int
    vehicles_by_type: Dict[VehicleType, int] = field(default_factory=dict)
    density: float = 0.0  # density as percentage: (∑(vehicle areas) / ROI area) × 100


@dataclass
class TrafficStatistics:
    """Overall traffic statistics."""
    total_vehicles: int = 0
    vehicles_by_type: Dict[VehicleType, int] = field(default_factory=dict)
    max_vehicles_in_frame: int = 0
    average_vehicles_per_frame: float = 0.0
    peak_density: float = 0.0
    frame_statistics: list[FrameStatistics] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

