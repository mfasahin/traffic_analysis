"""Speed calibration entity for converting pixels to real-world units."""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class SpeedCalibration:
    """
    Calibration data for converting pixel measurements to real-world speed.
    
    This is used to convert pixel-based movement to km/h. The calibration
    can be done by:
    1. Measuring a known distance in the video (e.g., road lane width)
    2. Providing reference points and real-world distances
    3. Using camera parameters if available
    """
    pixels_per_meter: float  # Conversion factor: pixels per meter
    fps: float  # Video frames per second
    
    def pixels_to_meters(self, pixels: float) -> float:
        """Convert pixels to meters."""
        return pixels / self.pixels_per_meter
    
    def pixels_per_second_to_kmh(self, pixels_per_second: float) -> float:
        """
        Convert speed from pixels per second to km/h.
        
        Formula: km/h = (pixels/s / pixels_per_meter) * 3.6
        
        Args:
            pixels_per_second: Speed in pixels per second
            
        Returns:
            Speed in km/h
        """
        if self.pixels_per_meter <= 0:
            return 0.0
        
        meters_per_second = pixels_per_second / self.pixels_per_meter
        kmh = meters_per_second * 3.6  # m/s to km/h conversion
        return kmh
    
    def calculate_speed_kmh(
        self,
        distance_pixels: float,
        time_seconds: float
    ) -> float:
        """
        Calculate speed in km/h from pixel distance and time.
        
        Args:
            distance_pixels: Distance traveled in pixels
            time_seconds: Time elapsed in seconds
            
        Returns:
            Speed in km/h
        """
        if time_seconds <= 0:
            return 0.0
        
        pixels_per_second = distance_pixels / time_seconds
        return self.pixels_per_second_to_kmh(pixels_per_second)
    
    @staticmethod
    def from_reference_distance(
        reference_distance_pixels: float,
        reference_distance_meters: float,
        fps: float
    ) -> 'SpeedCalibration':
        """
        Create calibration from a known reference distance.
        
        Example: If a 3.5m lane width is 100 pixels in the video:
        calibration = SpeedCalibration.from_reference_distance(100, 3.5, 30)
        
        Args:
            reference_distance_pixels: Known distance in pixels
            reference_distance_meters: Real-world distance in meters
            fps: Video frames per second
            
        Returns:
            SpeedCalibration object
        """
        pixels_per_meter = reference_distance_pixels / reference_distance_meters
        return SpeedCalibration(pixels_per_meter=pixels_per_meter, fps=fps)
    
    @staticmethod
    def from_roi_and_lane_width(
        roi_width_pixels: float,
        lane_width_meters: float,
        num_lanes: int,
        fps: float
    ) -> 'SpeedCalibration':
        """
        Create calibration from ROI width and lane information.
        
        Args:
            roi_width_pixels: Width of ROI in pixels
            lane_width_meters: Standard lane width in meters (typically 3.5m)
            num_lanes: Number of lanes in ROI
            fps: Video frames per second
            
        Returns:
            SpeedCalibration object
        """
        total_width_meters = lane_width_meters * num_lanes
        pixels_per_meter = roi_width_pixels / total_width_meters
        return SpeedCalibration(pixels_per_meter=pixels_per_meter, fps=fps)

