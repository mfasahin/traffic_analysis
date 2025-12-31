"""Day/night mode detection based on frame brightness."""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import cv2


class DayNightMode:
    """Day/night mode enumeration."""
    DAY = "day"
    NIGHT = "night"
    UNKNOWN = "unknown"


@dataclass
class DayNightDetector:
    """Detector for day/night mode based on frame brightness."""
    
    brightness_threshold: float = 50.0  # Threshold for day/night classification
    smoothing_window: int = 10  # Number of frames to consider for smoothing
    
    def __init__(self, brightness_threshold: float = 50.0, smoothing_window: int = 10):
        """
        Initialize day/night detector.
        
        Args:
            brightness_threshold: Average brightness threshold (0-255)
                                  Below this = night, above = day
            smoothing_window: Number of frames to smooth the detection
        """
        self.brightness_threshold = brightness_threshold
        self.smoothing_window = smoothing_window
        self.recent_detections: list[DayNightMode] = []
    
    def detect(self, frame: np.ndarray) -> DayNightMode:
        """
        Detect day/night mode from frame.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            DayNightMode (DAY, NIGHT, or UNKNOWN)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate average brightness
        avg_brightness = np.mean(gray)
        
        # Classify based on threshold
        if avg_brightness < self.brightness_threshold:
            mode = DayNightMode.NIGHT
        else:
            mode = DayNightMode.DAY
        
        # Add to recent detections for smoothing
        self.recent_detections.append(mode)
        if len(self.recent_detections) > self.smoothing_window:
            self.recent_detections.pop(0)
        
        # Return smoothed result
        return self.get_smoothed_mode()
    
    def get_smoothed_mode(self) -> DayNightMode:
        """Get smoothed day/night mode based on recent detections."""
        if not self.recent_detections:
            return DayNightMode.UNKNOWN
        
        # Count occurrences
        day_count = sum(1 for m in self.recent_detections if m == DayNightMode.DAY)
        night_count = sum(1 for m in self.recent_detections if m == DayNightMode.NIGHT)
        
        # Return majority
        if day_count > night_count:
            return DayNightMode.DAY
        elif night_count > day_count:
            return DayNightMode.NIGHT
        else:
            return DayNightMode.UNKNOWN
    
    def get_brightness(self, frame: np.ndarray) -> float:
        """Get average brightness of frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))
    
    def reset(self):
        """Reset detection history."""
        self.recent_detections.clear()

