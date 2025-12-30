"""Use case for tracking vehicles across frames."""

from typing import List, Dict, Optional, Tuple
from domain.entities.vehicle import Vehicle
from domain.entities.tracked_vehicle import TrackedVehicle, VehiclePosition
from domain.entities.speed_calibration import SpeedCalibration
from domain.entities.direction import Direction


class VehicleTracker:
    """
    Simple IOU-based vehicle tracker.
    
    Tracks vehicles across frames by matching detections based on:
    - Intersection over Union (IOU) of bounding boxes
    - Centroid distance
    - Maximum distance threshold
    """
    
    def __init__(
        self,
        max_disappeared_frames: int = 5,
        max_distance: float = 100.0,
        iou_threshold: float = 0.3
    ):
        """
        Initialize vehicle tracker.
        
        Args:
            max_disappeared_frames: Maximum frames a vehicle can be missing before removal
            max_distance: Maximum pixel distance for matching (centroid)
            iou_threshold: Minimum IOU for bounding box matching
        """
        self.max_disappeared_frames = max_disappeared_frames
        self.max_distance = max_distance
        self.iou_threshold = iou_threshold
        self.tracked_vehicles: Dict[int, TrackedVehicle] = {}
        self.next_track_id = 1
        self.disappeared_count: Dict[int, int] = {}  # Track ID -> frames disappeared
    
    def update(
        self,
        vehicles: List[Vehicle],
        frame_number: int,
        timestamp: float,
        calibration: Optional[SpeedCalibration] = None
    ) -> List[TrackedVehicle]:
        """
        Update tracker with new vehicle detections.
        
        Args:
            vehicles: List of detected vehicles in current frame
            frame_number: Current frame number
            timestamp: Current timestamp in seconds
            calibration: Optional speed calibration for speed calculation
            
        Returns:
            List of tracked vehicles (active and matched)
        """
        # If no vehicles detected, increment disappeared count
        if len(vehicles) == 0:
            for track_id in list(self.disappeared_count.keys()):
                self.disappeared_count[track_id] += 1
                if self.disappeared_count[track_id] > self.max_disappeared_frames:
                    # Remove track
                    if track_id in self.tracked_vehicles:
                        self.tracked_vehicles[track_id].is_active = False
                    del self.disappeared_count[track_id]
                    if track_id in self.tracked_vehicles:
                        del self.tracked_vehicles[track_id]
            return list(self.tracked_vehicles.values())
        
        # If no existing tracks, create new ones
        if len(self.tracked_vehicles) == 0:
            for vehicle in vehicles:
                self._create_track(vehicle, frame_number, timestamp, calibration)
        else:
            # Match existing tracks with new detections
            matched, unmatched_detections, unmatched_tracks = self._match_detections(
                vehicles, frame_number, timestamp, calibration
            )
            
            # Update matched tracks
            for track_id, vehicle in matched.items():
                self._update_track(track_id, vehicle, frame_number, timestamp, calibration)
                if track_id in self.disappeared_count:
                    del self.disappeared_count[track_id]
            
            # Handle unmatched detections (new vehicles)
            for vehicle in unmatched_detections:
                self._create_track(vehicle, frame_number, timestamp, calibration)
            
            # Handle unmatched tracks (disappeared vehicles)
            for track_id in unmatched_tracks:
                if track_id not in self.disappeared_count:
                    self.disappeared_count[track_id] = 0
                self.disappeared_count[track_id] += 1
                
                if self.disappeared_count[track_id] > self.max_disappeared_frames:
                    if track_id in self.tracked_vehicles:
                        self.tracked_vehicles[track_id].is_active = False
                    del self.disappeared_count[track_id]
                    if track_id in self.tracked_vehicles:
                        del self.tracked_vehicles[track_id]
        
        return [tv for tv in self.tracked_vehicles.values() if tv.is_active]
    
    def _create_track(
        self,
        vehicle: Vehicle,
        frame_number: int,
        timestamp: float,
        calibration: Optional[SpeedCalibration]
    ):
        """Create a new track for a vehicle."""
        track_id = self.next_track_id
        self.next_track_id += 1
        
        position = VehiclePosition(
            frame_number=frame_number,
            timestamp=timestamp,
            center=vehicle.center,
            bbox=vehicle.bbox,
            speed_kmh=0.0,
            direction=Direction.UNKNOWN
        )
        
        tracked_vehicle = TrackedVehicle(
            track_id=track_id,
            vehicle_type=vehicle.vehicle_type,
            first_seen_frame=frame_number,
            last_seen_frame=frame_number,
            current_direction=Direction.UNKNOWN
        )
        tracked_vehicle.add_position(position)
        
        self.tracked_vehicles[track_id] = tracked_vehicle
    
    def _update_track(
        self,
        track_id: int,
        vehicle: Vehicle,
        frame_number: int,
        timestamp: float,
        calibration: Optional[SpeedCalibration]
    ):
        """Update an existing track with new detection."""
        tracked_vehicle = self.tracked_vehicles[track_id]
        
        # Calculate speed (always calculate pixels/second, km/h only if calibration available)
        speed_pixels_per_second = 0.0
        speed_kmh = 0.0
        
        if len(tracked_vehicle.positions) > 0:
            prev_pos = tracked_vehicle.positions[-1]
            dx = vehicle.center[0] - prev_pos.center[0]
            dy = vehicle.center[1] - prev_pos.center[1]
            distance_pixels = (dx**2 + dy**2) ** 0.5
            time_diff = timestamp - prev_pos.timestamp
            
            if time_diff > 0:
                # Always calculate pixels per second
                speed_pixels_per_second = distance_pixels / time_diff
                
                # Calculate km/h only if calibration is available
                if calibration:
                    speed_kmh = calibration.calculate_speed_kmh(distance_pixels, time_diff)
        
        position = VehiclePosition(
            frame_number=frame_number,
            timestamp=timestamp,
            center=vehicle.center,
            bbox=vehicle.bbox,
            speed_pixels_per_second=speed_pixels_per_second,
            speed_kmh=speed_kmh,
            direction=Direction.UNKNOWN
        )
        
        tracked_vehicle.add_position(position)
    
    def _match_detections(
        self,
        vehicles: List[Vehicle],
        frame_number: int,
        timestamp: float,
        calibration: Optional[SpeedCalibration]
    ) -> Tuple[Dict[int, Vehicle], List[Vehicle], List[int]]:
        """
        Match detections with existing tracks.
        
        Returns:
            Tuple of (matched: {track_id: vehicle}, unmatched_detections, unmatched_tracks)
        """
        if len(self.tracked_vehicles) == 0:
            return {}, vehicles, []
        
        # Get active tracks
        active_tracks = {
            tid: tv for tid, tv in self.tracked_vehicles.items()
            if tv.is_active and len(tv.positions) > 0
        }
        
        if len(active_tracks) == 0:
            return {}, vehicles, []
        
        # Calculate distance matrix (centroid distance)
        distances = {}
        for track_id, tracked_vehicle in active_tracks.items():
            last_pos = tracked_vehicle.get_current_position()
            if last_pos is None:
                continue
            
            for i, vehicle in enumerate(vehicles):
                dx = vehicle.center[0] - last_pos.center[0]
                dy = vehicle.center[1] - last_pos.center[1]
                distance = (dx**2 + dy**2) ** 0.5
                
                # Also check IOU
                iou = self._calculate_iou(vehicle.bbox, last_pos.bbox)
                
                # Combined score: prefer closer and higher IOU
                if distance <= self.max_distance and iou >= self.iou_threshold:
                    score = distance * (1.0 - iou)  # Lower is better
                    distances[(track_id, i)] = score
        
        # Greedy matching
        matched = {}
        used_detections = set()
        used_tracks = set()
        
        # Sort by score (best matches first)
        sorted_matches = sorted(distances.items(), key=lambda x: x[1])
        
        for (track_id, det_idx), score in sorted_matches:
            if track_id not in used_tracks and det_idx not in used_detections:
                matched[track_id] = vehicles[det_idx]
                used_tracks.add(track_id)
                used_detections.add(det_idx)
        
        # Find unmatched
        unmatched_detections = [
            vehicles[i] for i in range(len(vehicles))
            if i not in used_detections
        ]
        unmatched_tracks = [
            tid for tid in active_tracks.keys()
            if tid not in used_tracks
        ]
        
        return matched, unmatched_detections, unmatched_tracks
    
    def _calculate_iou(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union (IOU) of two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def reset(self):
        """Reset tracker state."""
        self.tracked_vehicles.clear()
        self.disappeared_count.clear()
        self.next_track_id = 1

