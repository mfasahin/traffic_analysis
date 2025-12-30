"""Use case for detecting speed limit violations."""

from typing import List
from domain.entities.tracked_vehicle import TrackedVehicle


class SpeedViolationDetector:
    """Detector for speed limit violations."""
    
    def __init__(self, speed_limit_kmh: float = 50.0):
        """
        Initialize speed violation detector.
        
        Args:
            speed_limit_kmh: Speed limit in km/h (default: 50 km/h)
        """
        self.speed_limit_kmh = speed_limit_kmh
    
    def check_violation(self, tracked_vehicle: TrackedVehicle) -> bool:
        """
        Check if a tracked vehicle has violated the speed limit.
        
        Note: Speed violations can only be detected if calibration is available.
        Without calibration, speed is in pixels/second and cannot be compared to km/h limit.
        
        Args:
            tracked_vehicle: Tracked vehicle to check
            
        Returns:
            True if vehicle exceeded speed limit, False otherwise
        """
        if not tracked_vehicle.positions:
            return False
        
        # Only check violations if we have km/h values (calibration available)
        # Check current speed
        current_pos = tracked_vehicle.get_current_position()
        if current_pos and current_pos.speed_kmh > 0:
            if current_pos.speed_kmh > self.speed_limit_kmh:
                tracked_vehicle.speed_violations += 1
                return True
        
        # Check average speed (only if it's actually km/h, not pixels/second)
        # If average_speed_kmh is very high (>1000), it's likely pixels/second, not km/h
        if tracked_vehicle.average_speed_kmh > 0 and tracked_vehicle.average_speed_kmh < 1000:
            if tracked_vehicle.average_speed_kmh > self.speed_limit_kmh:
                tracked_vehicle.speed_violations += 1
                return True
        
        return False
    
    def get_violations(self, tracked_vehicles: List[TrackedVehicle]) -> List[TrackedVehicle]:
        """
        Get all vehicles that have violated speed limit.
        
        Args:
            tracked_vehicles: List of tracked vehicles
            
        Returns:
            List of vehicles with speed violations
        """
        violations = []
        for vehicle in tracked_vehicles:
            if self.check_violation(vehicle):
                violations.append(vehicle)
        return violations
    
    def count_violations(self, tracked_vehicles: List[TrackedVehicle]) -> int:
        """
        Count number of vehicles with speed violations.
        
        Args:
            tracked_vehicles: List of tracked vehicles
            
        Returns:
            Number of vehicles with violations
        """
        return len(self.get_violations(tracked_vehicles))

