"""Temporal smoothing service for filtering out sudden density drops."""

from typing import Optional, Deque
from collections import deque
from domain.entities.traffic_statistics import FrameStatistics


class TemporalSmoother:
    """Service for smoothing frame statistics to handle occlusion events."""
    
    def __init__(
        self,
        window_size: int = 5,
        drop_threshold: float = 0.5,
        smoothing_method: str = "moving_average"
    ):
        """
        Initialize temporal smoother.
        
        Args:
            window_size: Number of previous frames to consider for smoothing
            drop_threshold: Threshold for detecting sudden drops (0.5 = 50% drop)
                           If density drops more than this, use smoothed value
            smoothing_method: "moving_average" or "exponential"
        """
        self.window_size = window_size
        self.drop_threshold = drop_threshold
        self.smoothing_method = smoothing_method
        self.density_history: Deque[float] = deque(maxlen=window_size)
        self.vehicle_count_history: Deque[int] = deque(maxlen=window_size)
        self.last_smoothed_density: Optional[float] = None
        self.last_smoothed_count: Optional[int] = None
    
    def smooth_frame_statistics(self, frame_stats: FrameStatistics) -> FrameStatistics:
        """
        Apply temporal smoothing to frame statistics.
        
        This method filters out sudden drops in density/vehicle count that may
        occur due to occlusion (e.g., bird blocking camera view). It uses
        previous frame values to smooth out these anomalies.
        
        Args:
            frame_stats: Original frame statistics
            
        Returns:
            Smoothed frame statistics
        """
        current_density = frame_stats.density
        current_count = frame_stats.vehicle_count
        
        # Initialize history if empty
        if len(self.density_history) == 0:
            self.density_history.append(current_density)
            self.vehicle_count_history.append(current_count)
            self.last_smoothed_density = current_density
            self.last_smoothed_count = current_count
            return frame_stats
        
        # Calculate smoothed values based on history
        smoothed_density = self._calculate_smoothed_density(current_density)
        smoothed_count = self._calculate_smoothed_count(current_count)
        
        # Detect sudden drop compared to previous smoothed value
        final_density = current_density
        final_count = current_count
        
        if self.last_smoothed_density is not None and self.last_smoothed_density > 0:
            density_drop_ratio = current_density / self.last_smoothed_density
            
            # If sudden drop detected (e.g., bird blocking view), use smoothed value
            # This prevents artificial density drops from occlusion events
            if density_drop_ratio < (1.0 - self.drop_threshold):
                final_density = smoothed_density
                final_count = smoothed_count
        
        # Update history with final values (smoothed or original)
        self.density_history.append(final_density)
        self.vehicle_count_history.append(final_count)
        self.last_smoothed_density = final_density
        self.last_smoothed_count = final_count
        
        # Create smoothed frame statistics
        smoothed_stats = FrameStatistics(
            frame_number=frame_stats.frame_number,
            timestamp=frame_stats.timestamp,
            vehicle_count=final_count,
            vehicles_by_type=frame_stats.vehicles_by_type,
            density=final_density
        )
        
        return smoothed_stats
    
    def _calculate_smoothed_density(self, current_density: float) -> float:
        """Calculate smoothed density value."""
        if self.smoothing_method == "exponential":
            if self.last_smoothed_density is None:
                return current_density
            # Exponential moving average with alpha = 0.3
            alpha = 0.3
            return alpha * current_density + (1 - alpha) * self.last_smoothed_density
        else:
            # Simple moving average
            if len(self.density_history) == 0:
                return current_density
            history_list = list(self.density_history)
            history_list.append(current_density)
            return sum(history_list) / len(history_list)
    
    def _calculate_smoothed_count(self, current_count: int) -> int:
        """Calculate smoothed vehicle count value."""
        if self.smoothing_method == "exponential":
            if self.last_smoothed_count is None:
                return current_count
            # Exponential moving average with alpha = 0.3
            alpha = 0.3
            smoothed = alpha * current_count + (1 - alpha) * self.last_smoothed_count
            return int(round(smoothed))
        else:
            # Simple moving average
            if len(self.vehicle_count_history) == 0:
                return current_count
            history_list = list(self.vehicle_count_history)
            history_list.append(current_count)
            smoothed = sum(history_list) / len(history_list)
            return int(round(smoothed))
    
    def reset(self):
        """Reset smoothing history."""
        self.density_history.clear()
        self.vehicle_count_history.clear()
        self.last_smoothed_density = None
        self.last_smoothed_count = None

