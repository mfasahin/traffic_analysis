"""Interface for vehicle detection."""

from abc import ABC, abstractmethod
from typing import List
from domain.entities.vehicle import Vehicle


class IVehicleDetector(ABC):
    """Interface for vehicle detection service."""
    
    @abstractmethod
    def detect_vehicles(self, frame) -> List[Vehicle]:
        """
        Detect vehicles in a frame.
        
        Args:
            frame: Input frame (numpy array or similar)
            
        Returns:
            List of detected Vehicle entities
        """
        pass

