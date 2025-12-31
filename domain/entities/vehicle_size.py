"""Vehicle size classification based on bounding box dimensions."""

from dataclasses import dataclass
from typing import Tuple, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from domain.entities.vehicle import Vehicle


class VehicleSize(Enum):
    """Vehicle size categories."""
    SMALL = "small"      # Motorcycle, bicycle
    MEDIUM = "medium"    # Car, sedan
    LARGE = "large"      # Truck, bus
    UNKNOWN = "unknown"


@dataclass
class VehicleSizeClassifier:
    """Classifier for vehicle size based on bounding box."""
    
    # Size thresholds in pixels (can be adjusted based on video resolution)
    small_threshold: int = 3000      # Area < 3000 pixels = small
    medium_threshold: int = 15000    # Area < 15000 pixels = medium, else large
    
    def classify(self, vehicle: 'Vehicle') -> VehicleSize:
        """
        Classify vehicle size based on bounding box area.
        
        Args:
            vehicle: Vehicle to classify
            
        Returns:
            VehicleSize category
        """
        area = vehicle.get_area()
        
        if area < self.small_threshold:
            return VehicleSize.SMALL
        elif area < self.medium_threshold:
            return VehicleSize.MEDIUM
        else:
            return VehicleSize.LARGE
    
    def classify_by_dimensions(
        self,
        width: int,
        height: int
    ) -> VehicleSize:
        """
        Classify vehicle size by width and height.
        
        Args:
            width: Bounding box width in pixels
            height: Bounding box height in pixels
            
        Returns:
            VehicleSize category
        """
        area = width * height
        
        if area < self.small_threshold:
            return VehicleSize.SMALL
        elif area < self.medium_threshold:
            return VehicleSize.MEDIUM
        else:
            return VehicleSize.LARGE
    
    def get_size_from_vehicle_type(self, vehicle_type) -> VehicleSize:
        """
        Get expected size category from vehicle type.
        
        Args:
            vehicle_type: VehicleType enum
            
        Returns:
            VehicleSize category
        """
        from domain.entities.vehicle import VehicleType
        
        size_mapping = {
            VehicleType.MOTORCYCLE: VehicleSize.SMALL,
            VehicleType.BICYCLE: VehicleSize.SMALL,
            VehicleType.CAR: VehicleSize.MEDIUM,
            VehicleType.TRUCK: VehicleSize.LARGE,
            VehicleType.BUS: VehicleSize.LARGE,
            VehicleType.UNKNOWN: VehicleSize.UNKNOWN
        }
        
        return size_mapping.get(vehicle_type, VehicleSize.UNKNOWN)

