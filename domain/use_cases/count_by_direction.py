"""Use case for counting vehicles by direction."""

from typing import Dict, List
from domain.entities.tracked_vehicle import TrackedVehicle
from domain.entities.direction import Direction


class DirectionCounter:
    """Counter for vehicles by movement direction."""
    
    @staticmethod
    def count_by_direction(tracked_vehicles: List[TrackedVehicle]) -> Dict[Direction, int]:
        """
        Count vehicles by their movement direction.
        
        Args:
            tracked_vehicles: List of tracked vehicles
            
        Returns:
            Dictionary mapping direction to count
        """
        counts = {direction: 0 for direction in Direction}
        
        for vehicle in tracked_vehicles:
            if vehicle.is_active and vehicle.current_direction != Direction.UNKNOWN:
                counts[vehicle.current_direction] = counts.get(vehicle.current_direction, 0) + 1
        
        return counts
    
    @staticmethod
    def get_vehicles_by_direction(
        tracked_vehicles: List[TrackedVehicle],
        direction: Direction
    ) -> List[TrackedVehicle]:
        """
        Get all vehicles moving in a specific direction.
        
        Args:
            tracked_vehicles: List of tracked vehicles
            direction: Direction to filter by
            
        Returns:
            List of vehicles moving in the specified direction
        """
        return [
            vehicle for vehicle in tracked_vehicles
            if vehicle.is_active and vehicle.current_direction == direction
        ]
    
    @staticmethod
    def get_direction_statistics(tracked_vehicles: List[TrackedVehicle]) -> Dict[str, any]:
        """
        Get comprehensive direction statistics.
        
        Args:
            tracked_vehicles: List of tracked vehicles
            
        Returns:
            Dictionary with direction statistics
        """
        counts = DirectionCounter.count_by_direction(tracked_vehicles)
        total = sum(counts.values())
        
        return {
            "counts": {direction.value: count for direction, count in counts.items()},
            "total": total,
            "percentages": {
                direction.value: (count / total * 100) if total > 0 else 0.0
                for direction, count in counts.items()
            }
        }

