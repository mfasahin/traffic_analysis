"""Video visualization with detection results."""

import cv2
import numpy as np
from typing import List, Optional
from domain.entities.vehicle import Vehicle, VehicleType
from domain.entities.traffic_statistics import FrameStatistics
from domain.entities.roi import ROI


class VideoVisualizer:
    """Visualizer for video with vehicle detection overlays."""
    
    # Colors for different vehicle types (BGR format for OpenCV)
    VEHICLE_COLORS = {
        VehicleType.CAR: (0, 255, 0),          # Green
        VehicleType.TRUCK: (255, 0, 0),        # Blue
        VehicleType.BUS: (0, 0, 255),          # Red
        VehicleType.MOTORCYCLE: (0, 255, 255), # Yellow
        VehicleType.BICYCLE: (255, 255, 0),    # Cyan
        VehicleType.UNKNOWN: (128, 128, 128),   # Gray
    }
    
    def __init__(self, show_confidence: bool = True, show_count: bool = True):
        """
        Initialize video visualizer.
        
        Args:
            show_confidence: Whether to show confidence scores
            show_count: Whether to show vehicle count on frame
        """
        self.show_confidence = show_confidence
        self.show_count = show_count
    
    def draw_vehicles(
        self,
        frame: np.ndarray,
        vehicles: List[Vehicle],
        frame_stats: Optional[FrameStatistics] = None,
        roi: Optional[ROI] = None
    ) -> np.ndarray:
        """
        Draw vehicle detections on frame.
        
        Args:
            frame: Input frame
            vehicles: List of detected vehicles
            frame_stats: Optional frame statistics
            roi: Optional Region of Interest to draw
            
        Returns:
            Frame with drawn detections
        """
        frame_copy = frame.copy()
        
        # Draw ROI polygon in red (if provided)
        if roi is not None:
            self._draw_roi(frame_copy, roi)
        
        # Draw bounding boxes
        for vehicle in vehicles:
            x1, y1, x2, y2 = vehicle.bbox
            color = self.VEHICLE_COLORS.get(vehicle.vehicle_type, (128, 128, 128))
            
            # Draw rectangle
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = vehicle.vehicle_type.value
            if self.show_confidence:
                label += f" {vehicle.confidence:.2f}"
            
            # Calculate text size
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # Draw label background
            cv2.rectangle(
                frame_copy,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                frame_copy,
                label,
                (x1, y1 - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        # Draw statistics overlay
        if frame_stats and self.show_count:
            self._draw_statistics_overlay(frame_copy, frame_stats)
        
        return frame_copy
    
    def _draw_statistics_overlay(self, frame: np.ndarray, stats: FrameStatistics):
        """Draw statistics overlay on frame."""
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        # Semi-transparent background
        cv2.rectangle(overlay, (10, 10), (300, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Statistics text
        y_offset = 30
        line_height = 25
        
        texts = [
            f"Frame: {stats.frame_number}",
            f"Vehicles: {stats.vehicle_count}",
            f"Density: {stats.density:.2f}",
        ]
        
        for i, text in enumerate(texts):
            cv2.putText(
                frame,
                text,
                (20, y_offset + i * line_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        # Vehicle type breakdown
        y_start = 110
        for vehicle_type, count in stats.vehicles_by_type.items():
            if count > 0:
                color = self.VEHICLE_COLORS.get(vehicle_type, (128, 128, 128))
                text = f"{vehicle_type.value}: {count}"
                cv2.putText(
                    frame,
                    text,
                    (20, y_start),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1
                )
                y_start += 20
    
    def _draw_roi(self, frame: np.ndarray, roi: ROI):
        """
        Draw ROI polygon on frame in red color.
        
        Args:
            frame: Frame to draw on
            roi: Region of Interest to draw
        """
        if len(roi.polygon_points) < 3:
            return
        
        # Convert points to numpy array
        points = np.array(roi.polygon_points, dtype=np.int32)
        
        # Draw filled polygon with transparency
        overlay = frame.copy()
        cv2.fillPoly(overlay, [points], (0, 0, 255))  # Red color in BGR
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Draw polygon outline in bright red
        cv2.polylines(frame, [points], isClosed=True, color=(0, 0, 255), thickness=2)

