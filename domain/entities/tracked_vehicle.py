"""Tracked vehicle entity with movement history."""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from datetime import datetime
from domain.entities.vehicle import Vehicle, VehicleType
from domain.entities.direction import Direction


@dataclass
class VehiclePosition:
    """Vehicle position at a specific frame."""
    frame_number: int
    timestamp: float
    center: Tuple[int, int]
    bbox: Tuple[int, int, int, int]
    speed_pixels_per_second: float = 0.0
    speed_kmh: float = 0.0
    direction: Direction = Direction.UNKNOWN


@dataclass
class TrackedVehicle:
    """Tracked vehicle with movement history."""
    track_id: int
    vehicle_type: VehicleType
    first_seen_frame: int
    last_seen_frame: int
    positions: List[VehiclePosition] = field(default_factory=list)
    current_direction: Direction = Direction.UNKNOWN
    average_speed_kmh: float = 0.0
    max_speed_kmh: float = 0.0
    total_distance_pixels: float = 0.0
    is_active: bool = True
    speed_violations: int = 0  # Number of speed limit violations
    
    def add_position(self, position: VehiclePosition):
        """Add a new position to the tracking history."""
        self.positions.append(position)
        self.last_seen_frame = position.frame_number
        
        # Update direction if we have at least 2 positions
        if len(self.positions) >= 2:
            self._update_direction()
            self._update_speed()
    
    def _update_direction(self):
        """Update vehicle direction based on recent positions."""
        if len(self.positions) < 2:
            return
        
        # Use last 2 positions to determine direction
        pos1 = self.positions[-2]
        pos2 = self.positions[-1]
        
        dx = pos2.center[0] - pos1.center[0]
        dy = pos2.center[1] - pos1.center[1]
        
        # Threshold to avoid noise
        threshold = 2  # pixels
        
        if abs(dx) < threshold and abs(dy) < threshold:
            self.current_direction = Direction.STATIONARY
        elif abs(dx) > abs(dy):
            # Horizontal movement
            if dx > threshold:
                self.current_direction = Direction.RIGHT
            elif dx < -threshold:
                self.current_direction = Direction.LEFT
        else:
            # Vertical movement
            if dy > threshold:
                self.current_direction = Direction.DOWN
            elif dy < -threshold:
                self.current_direction = Direction.UP
    
    def _update_speed(self):
        """Update speed statistics based on position history."""
        if len(self.positions) < 2:
            return
        
        speeds_kmh = []
        speeds_pixels_per_second = []
        total_distance = 0.0
        
        for i in range(1, len(self.positions)):
            pos1 = self.positions[i-1]
            pos2 = self.positions[i]
            
            # Calculate distance in pixels
            dx = pos2.center[0] - pos1.center[0]
            dy = pos2.center[1] - pos1.center[1]
            distance_pixels = (dx**2 + dy**2) ** 0.5
            
            # Calculate time difference
            time_diff = pos2.timestamp - pos1.timestamp
            if time_diff > 0:
                # speed_pixels_per_second is already calculated in tracker
                if pos2.speed_pixels_per_second > 0:
                    speeds_pixels_per_second.append(pos2.speed_pixels_per_second)
                
                # speed_kmh is already calculated by calibration if available
                if pos2.speed_kmh > 0:
                    speeds_kmh.append(pos2.speed_kmh)
                
                total_distance += distance_pixels
        
        self.total_distance_pixels = total_distance
        
        # Update statistics (use km/h if available, otherwise pixels/second)
        if speeds_kmh:
            self.average_speed_kmh = sum(speeds_kmh) / len(speeds_kmh)
            self.max_speed_kmh = max(speeds_kmh)
        elif speeds_pixels_per_second:
            # If no calibration, store average in kmh field as pixels/second for display
            # (This is a workaround - we'll handle display separately)
            avg_pixels_per_second = sum(speeds_pixels_per_second) / len(speeds_pixels_per_second)
            self.average_speed_kmh = avg_pixels_per_second  # Actually pixels/second
            self.max_speed_kmh = max(speeds_pixels_per_second)  # Actually pixels/second
    
    def get_current_position(self) -> Optional[VehiclePosition]:
        """Get the most recent position."""
        return self.positions[-1] if self.positions else None
    
    def get_distance_traveled_pixels(self) -> float:
        """Calculate total distance traveled in pixels."""
        if len(self.positions) < 2:
            return 0.0
        
        total = 0.0
        for i in range(1, len(self.positions)):
            pos1 = self.positions[i-1]
            pos2 = self.positions[i]
            dx = pos2.center[0] - pos1.center[0]
            dy = pos2.center[1] - pos1.center[1]
            total += (dx**2 + dy**2) ** 0.5
        
        return total

