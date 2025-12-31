"""Weather condition detection and analysis."""

from dataclasses import dataclass, field
from typing import Dict, Optional
from enum import Enum
import numpy as np
import cv2


class WeatherCondition(Enum):
    """Weather condition enumeration."""
    CLEAR = "clear"
    RAINY = "rainy"
    FOGGY = "foggy"
    SNOWY = "snowy"
    UNKNOWN = "unknown"


@dataclass
class WeatherDetector:
    """Detector for weather conditions based on frame analysis."""
    
    def detect(self, frame: np.ndarray, day_night_mode: str = "day") -> WeatherCondition:
        """
        Detect weather condition from frame.
        
        This is a simplified detector based on visual features:
        - Rain: High edge density, low contrast
        - Fog: Low contrast, high brightness variance
        - Snow: High brightness, high edge density
        - Clear: Normal contrast and brightness
        
        Args:
            frame: Input frame (BGR format)
            day_night_mode: "day" or "night" mode
            
        Returns:
            WeatherCondition
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate features
        avg_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        
        # Edge detection for rain/snow detection
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (frame.shape[0] * frame.shape[1])
        
        # Contrast (standard deviation of brightness)
        contrast = std_brightness
        
        # Simple heuristics (can be improved with ML)
        if day_night_mode == "night":
            # Harder to detect weather at night
            if contrast < 20:
                return WeatherCondition.FOGGY
            return WeatherCondition.CLEAR
        
        # Daytime detection
        if avg_brightness > 200 and edge_density > 0.15:
            return WeatherCondition.SNOWY
        elif edge_density > 0.12 and contrast < 30:
            return WeatherCondition.RAINY
        elif contrast < 25 and std_brightness > 40:
            return WeatherCondition.FOGGY
        else:
            return WeatherCondition.CLEAR
    
    def get_weather_impact_factor(self, condition: WeatherCondition) -> float:
        """
        Get impact factor of weather on traffic (0.0 to 1.0).
        
        Lower values indicate worse conditions (slower traffic expected).
        
        Args:
            condition: Detected weather condition
            
        Returns:
            Impact factor (1.0 = no impact, 0.5 = moderate impact, 0.0 = severe impact)
        """
        impact_factors = {
            WeatherCondition.CLEAR: 1.0,
            WeatherCondition.RAINY: 0.7,
            WeatherCondition.FOGGY: 0.5,
            WeatherCondition.SNOWY: 0.3,
            WeatherCondition.UNKNOWN: 1.0
        }
        return impact_factors.get(condition, 1.0)


@dataclass
class WeatherStatistics:
    """Statistics about weather conditions during analysis."""
    
    conditions_detected: Dict[WeatherCondition, int] = field(default_factory=dict)
    dominant_condition: Optional[WeatherCondition] = None
    average_impact_factor: float = 1.0
    
    def add_condition(self, condition: WeatherCondition):
        """Add a detected condition."""
        self.conditions_detected[condition] = self.conditions_detected.get(condition, 0) + 1
        self._update_dominant()
    
    def _update_dominant(self):
        """Update dominant condition."""
        if not self.conditions_detected:
            self.dominant_condition = None
            return
        
        self.dominant_condition = max(
            self.conditions_detected.items(),
            key=lambda x: x[1]
        )[0]

