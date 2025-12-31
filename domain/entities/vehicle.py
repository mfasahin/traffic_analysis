"""Vehicle entity representing a detected vehicle."""

from dataclasses import dataclass, field
from typing import Optional, Tuple, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from domain.entities.vehicle_size import VehicleSize, VehicleSizeClassifier
else:
    # Import at runtime to avoid circular dependency
    from domain.entities.vehicle_size import VehicleSize, VehicleSizeClassifier


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
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    vehicle_type: VehicleType
    center: Tuple[int, int]  # (x, y) center point
    frame_number: int
    size: Optional[VehicleSize] = None  # Vehicle size classification
    
    def __post_init__(self):
        """Calculate center point from bounding box and classify size."""
        x1, y1, x2, y2 = self.bbox
        self.center = ((x1 + x2) // 2, (y1 + y2) // 2)
        
        # Auto-classify size if not provided
        if self.size is None:
            classifier = VehicleSizeClassifier()
            self.size = classifier.classify(self)
    
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
    
    def get_dimensions(self) -> Tuple[int, int]:
        """
        Get width and height of bounding box.
        
        Returns:
            Tuple of (width, height) in pixels
        """
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1, y2 - y1)

