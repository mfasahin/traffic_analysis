"""ROI (Region of Interest) entity for road area definition."""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import cv2


@dataclass
class ROI:
    """Region of Interest representing the road area as a polygon."""
    
    polygon_points: List[Tuple[int, int]]  # List of (x, y) points defining the polygon
    
    def __post_init__(self):
        """Validate polygon points."""
        if len(self.polygon_points) < 3:
            raise ValueError("ROI polygon must have at least 3 points")
    
    def get_area(self) -> int:
        """
        Calculate the area of the ROI polygon in pixels.
        
        Returns:
            Area in pixels
        """
        if len(self.polygon_points) < 3:
            return 0
        
        # Convert to numpy array for cv2.contourArea
        points = np.array(self.polygon_points, dtype=np.int32)
        return int(cv2.contourArea(points))
    
    def contains_point(self, x: int, y: int) -> bool:
        """
        Check if a point is inside the ROI polygon.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            True if point is inside ROI, False otherwise
        """
        if len(self.polygon_points) < 3:
            return False
        
        points = np.array(self.polygon_points, dtype=np.int32)
        return cv2.pointPolygonTest(points, (x, y), False) >= 0
    
    def get_bounding_box(self) -> Tuple[int, int, int, int]:
        """
        Get the bounding box of the ROI polygon.
        
        Returns:
            Tuple of (x_min, y_min, x_max, y_max)
        """
        if not self.polygon_points:
            return (0, 0, 0, 0)
        
        x_coords = [p[0] for p in self.polygon_points]
        y_coords = [p[1] for p in self.polygon_points]
        
        return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
    
    @staticmethod
    def create_full_frame(frame_width: int, frame_height: int) -> 'ROI':
        """
        Create a ROI that covers the entire frame (for backward compatibility).
        
        Args:
            frame_width: Width of the frame
            frame_height: Height of the frame
            
        Returns:
            ROI covering the entire frame
        """
        return ROI([
            (0, 0),
            (frame_width, 0),
            (frame_width, frame_height),
            (0, frame_height)
        ])

