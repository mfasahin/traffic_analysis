"""Vehicle entity representing a detected vehicle."""

from dataclasses import dataclass
from typing import Optional
from enum import Enum


class VehicleType(Enum):
    """Vehicle type enumeration."""
    CAR = "car"
    TRUCK = "truck"
    BUS = "bus"
    MOTORCYCLE = "motorcycle"
    BICYCLE = "bicycle"
    UNKNOWN = "unknown"


@dataclass
class Vehicle:
    """Vehicle entity with detection information."""
    id: int
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    vehicle_type: VehicleType
    center: tuple[int, int]  # (x, y) center point
    frame_number: int
    
    def __post_init__(self):
        """Calculate center point from bounding box."""
        x1, y1, x2, y2 = self.bbox
        self.center = ((x1 + x2) // 2, (y1 + y2) // 2)
    
    def get_area(self) -> int:
        """
        Calculate the area of the vehicle bounding box in pixels.
        
        Formula: Area = Width(w) × Height(h)
        
        Returns:
            Area in pixels
        """
        x1, y1, x2, y2 = self.bbox
        width = x2 - x1
        height = y2 - y1
        return width * height

