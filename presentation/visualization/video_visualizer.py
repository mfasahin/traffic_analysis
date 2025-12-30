"""Video visualization with detection results."""

import cv2
import numpy as np
from typing import List, Optional
from domain.entities.vehicle import Vehicle, VehicleType
from domain.entities.traffic_statistics import FrameStatistics
from domain.entities.roi import ROI
from domain.entities.tracked_vehicle import TrackedVehicle
from domain.entities.direction import Direction


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
    
    def __init__(
        self,
        show_confidence: bool = True,
        show_count: bool = True,
        show_speed: bool = True,
        show_direction: bool = True,
        show_tracking: bool = True
    ):
        """
        Initialize video visualizer.
        
        Args:
            show_confidence: Whether to show confidence scores
            show_count: Whether to show vehicle count on frame
            show_speed: Whether to show vehicle speed
            show_direction: Whether to show movement direction
            show_tracking: Whether to show tracking IDs and trails
        """
        self.show_confidence = show_confidence
        self.show_count = show_count
        self.show_speed = show_speed
        self.show_direction = show_direction
        self.show_tracking = show_tracking
        
        # Direction arrow colors (BGR)
        self.DIRECTION_COLORS = {
            Direction.UP: (255, 0, 255),      # Magenta
            Direction.DOWN: (255, 255, 0),    # Cyan
            Direction.LEFT: (0, 165, 255),   # Orange
            Direction.RIGHT: (0, 255, 255),   # Yellow
            Direction.UNKNOWN: (128, 128, 128),  # Gray
            Direction.STATIONARY: (192, 192, 192)  # Light gray
        }
    
    def draw_vehicles(
        self,
        frame: np.ndarray,
        vehicles: List[Vehicle],
        frame_stats: Optional[FrameStatistics] = None,
        roi: Optional[ROI] = None,
        tracked_vehicles: Optional[List[TrackedVehicle]] = None
    ) -> np.ndarray:
        """
        Draw vehicle detections on frame.
        
        Args:
            frame: Input frame
            vehicles: List of detected vehicles
            frame_stats: Optional frame statistics
            roi: Optional Region of Interest to draw
            tracked_vehicles: Optional list of tracked vehicles for speed/direction display
            
        Returns:
            Frame with drawn detections
        """
        frame_copy = frame.copy()
        
        # Draw ROI polygon in red (if provided)
        if roi is not None:
            self._draw_roi(frame_copy, roi)
        
        # Draw tracking trails if enabled
        if tracked_vehicles and self.show_tracking:
            self._draw_tracking_trails(frame_copy, tracked_vehicles)
        
        # Create a mapping from vehicle center to tracked vehicle for speed/direction display
        tracked_map = {}
        if tracked_vehicles:
            for tracked in tracked_vehicles:
                if tracked.is_active and tracked.positions:
                    pos = tracked.get_current_position()
                    if pos:
                        tracked_map[pos.center] = tracked
        
        # Draw bounding boxes
        for vehicle in vehicles:
            x1, y1, x2, y2 = vehicle.bbox
            color = self.VEHICLE_COLORS.get(vehicle.vehicle_type, (128, 128, 128))
            
            # Check if this vehicle is tracked
            tracked = tracked_map.get(vehicle.center, None)
            if tracked:
                # Use direction color if available
                dir_color = self.DIRECTION_COLORS.get(tracked.current_direction, color)
                # Blend colors
                color = tuple(int((c1 + c2) / 2) for c1, c2 in zip(color, dir_color))
            
            # Draw rectangle (thicker for tracked vehicles)
            thickness = 3 if tracked else 2
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label with tracking info
            label_parts = []
            if tracked and self.show_tracking:
                label_parts.append(f"ID:{tracked.track_id}")
            
            label_parts.append(vehicle.vehicle_type.value)
            
            if self.show_confidence:
                label_parts.append(f"{vehicle.confidence:.2f}")
            
            if tracked and self.show_speed and tracked.get_current_position():
                pos = tracked.get_current_position()
                # Show km/h if calibration available, otherwise pixels/second
                if pos.speed_kmh > 0:
                    label_parts.append(f"{pos.speed_kmh:.0f}km/h")
                elif pos.speed_pixels_per_second > 0:
                    label_parts.append(f"{pos.speed_pixels_per_second:.1f}px/s")
            
            if tracked and self.show_direction:
                direction = tracked.current_direction
                if direction != Direction.UNKNOWN and direction != Direction.STATIONARY:
                    label_parts.append(direction.value.upper())
            
            label = " | ".join(label_parts)
            
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
            
            # Draw direction arrow for tracked vehicles
            if tracked and self.show_direction and tracked.current_direction != Direction.UNKNOWN:
                self._draw_direction_arrow(frame_copy, vehicle.center, tracked.current_direction)
            
            # Highlight speed violations
            if tracked and tracked.speed_violations > 0:
                # Draw red border for violations
                cv2.rectangle(frame_copy, (x1-2, y1-2), (x2+2, y2+2), (0, 0, 255), 2)
                cv2.putText(
                    frame_copy,
                    "SPEED VIOLATION",
                    (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )
        
        # Draw statistics overlay
        if frame_stats and self.show_count:
            self._draw_statistics_overlay(frame_copy, frame_stats, tracked_vehicles)
        
        return frame_copy
    
    def _draw_tracking_trails(self, frame: np.ndarray, tracked_vehicles: List[TrackedVehicle]):
        """Draw movement trails for tracked vehicles."""
        for tracked in tracked_vehicles:
            if not tracked.is_active or len(tracked.positions) < 2:
                continue
            
            # Draw trail (last 10 positions)
            positions_to_draw = tracked.positions[-10:]
            for i in range(1, len(positions_to_draw)):
                pos1 = positions_to_draw[i-1]
                pos2 = positions_to_draw[i]
                
                # Fade color based on age
                alpha = i / len(positions_to_draw)
                color = tuple(int(c * alpha) for c in (0, 255, 255))  # Cyan trail
                
                cv2.line(
                    frame,
                    pos1.center,
                    pos2.center,
                    color,
                    2
                )
    
    def _draw_direction_arrow(
        self,
        frame: np.ndarray,
        center: tuple,
        direction: Direction
    ):
        """Draw direction arrow on vehicle."""
        x, y = center
        arrow_length = 20
        
        if direction == Direction.UP:
            end_point = (x, y - arrow_length)
        elif direction == Direction.DOWN:
            end_point = (x, y + arrow_length)
        elif direction == Direction.LEFT:
            end_point = (x - arrow_length, y)
        elif direction == Direction.RIGHT:
            end_point = (x + arrow_length, y)
        else:
            return
        
        color = self.DIRECTION_COLORS.get(direction, (255, 255, 255))
        cv2.arrowedLine(frame, center, end_point, color, 2, tipLength=0.3)
    
    def _draw_statistics_overlay(
        self,
        frame: np.ndarray,
        stats: FrameStatistics,
        tracked_vehicles: Optional[List[TrackedVehicle]] = None
    ):
        """Draw statistics overlay on frame."""
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        # Calculate overlay height based on content
        overlay_height = 150
        if stats.vehicles_by_direction:
            overlay_height += len(stats.vehicles_by_direction) * 20
        if tracked_vehicles and self.show_speed:
            overlay_height += 40
        
        # Semi-transparent background
        cv2.rectangle(overlay, (10, 10), (350, overlay_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Statistics text
        y_offset = 30
        line_height = 25
        
        texts = [
            f"Frame: {stats.frame_number}",
            f"Vehicles: {stats.vehicle_count}",
            f"Density: {stats.density:.2f}%",
        ]
        
        if stats.tracked_vehicles_count > 0:
            texts.append(f"Tracked: {stats.tracked_vehicles_count}")
        
        if stats.speed_violations_count > 0:
            texts.append(f"Speed Violations: {stats.speed_violations_count}")
        
        for i, text in enumerate(texts):
            color = (0, 0, 255) if "Violations" in text else (255, 255, 255)
            cv2.putText(
                frame,
                text,
                (20, y_offset + i * line_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
        
        # Direction statistics
        y_start = y_offset + len(texts) * line_height + 10
        if stats.vehicles_by_direction:
            cv2.putText(
                frame,
                "Direction:",
                (20, y_start),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            y_start += 20
            for direction, count in stats.vehicles_by_direction.items():
                if count > 0:
                    color = self.DIRECTION_COLORS.get(direction, (128, 128, 128))
                    text = f"  {direction.value.upper()}: {count}"
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
        
        # Speed statistics
        if tracked_vehicles and self.show_speed:
            speeds_kmh = []
            speeds_px_per_s = []
            has_calibration = False
            
            for tracked in tracked_vehicles:
                if tracked.is_active and tracked.get_current_position():
                    pos = tracked.get_current_position()
                    if pos.speed_kmh > 0:
                        speeds_kmh.append(pos.speed_kmh)
                        has_calibration = True
                    elif pos.speed_pixels_per_second > 0:
                        speeds_px_per_s.append(pos.speed_pixels_per_second)
            
            if speeds_kmh:
                # Show km/h if calibration available
                avg_speed = sum(speeds_kmh) / len(speeds_kmh)
                max_speed = max(speeds_kmh)
                cv2.putText(
                    frame,
                    f"Avg Speed: {avg_speed:.0f} km/h",
                    (20, y_start),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )
                y_start += 20
                cv2.putText(
                    frame,
                    f"Max Speed: {max_speed:.0f} km/h",
                    (20, y_start),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1
                )
                y_start += 20
            elif speeds_px_per_s:
                # Show pixels/second if no calibration
                avg_speed = sum(speeds_px_per_s) / len(speeds_px_per_s)
                max_speed = max(speeds_px_per_s)
                cv2.putText(
                    frame,
                    f"Avg Speed: {avg_speed:.1f} px/s",
                    (20, y_start),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )
                y_start += 20
                cv2.putText(
                    frame,
                    f"Max Speed: {max_speed:.1f} px/s",
                    (20, y_start),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1
                )
                y_start += 20
                cv2.putText(
                    frame,
                    "(No calibration - use px/s)",
                    (20, y_start),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (128, 128, 128),
                    1
                )
                y_start += 20
        
        # Vehicle type breakdown
        if stats.vehicles_by_type:
            y_start += 10
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

