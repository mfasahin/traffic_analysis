"""Traffic analysis service."""

from typing import List, Optional, Callable
from domain.entities.vehicle import Vehicle
from domain.entities.traffic_statistics import TrafficStatistics, FrameStatistics
from domain.entities.roi import ROI
from domain.entities.speed_calibration import SpeedCalibration
from domain.entities.tracked_vehicle import TrackedVehicle
from domain.use_cases.count_vehicles import CountVehiclesUseCase
from domain.use_cases.track_vehicles import VehicleTracker
from domain.use_cases.detect_speed_violations import SpeedViolationDetector
from domain.use_cases.count_by_direction import DirectionCounter
from application.interfaces.vehicle_detector import IVehicleDetector
from application.interfaces.video_processor import IVideoProcessor
from application.services.temporal_smoother import TemporalSmoother
from datetime import datetime


class TrafficAnalyzerService:
    """Service for analyzing traffic in video."""
    
    def __init__(
        self,
        vehicle_detector: IVehicleDetector,
        video_processor: IVideoProcessor,
        roi: Optional[ROI] = None,
        enable_smoothing: bool = True,
        smoothing_window: int = 5,
        drop_threshold: float = 0.5,
        enable_tracking: bool = True,
        speed_calibration: Optional[SpeedCalibration] = None,
        speed_limit_kmh: float = 50.0
    ):
        """
        Initialize traffic analyzer service.
        
        Args:
            vehicle_detector: Vehicle detection service
            video_processor: Video processing service
            roi: Optional Region of Interest (road area polygon). 
                 If None, uses full frame area for backward compatibility.
            enable_smoothing: Enable temporal smoothing to filter occlusion events
            smoothing_window: Number of frames to consider for smoothing
            drop_threshold: Threshold for detecting sudden drops (0.5 = 50% drop)
            enable_tracking: Enable vehicle tracking for speed and direction analysis
            speed_calibration: Optional speed calibration for converting pixels to km/h
            speed_limit_kmh: Speed limit in km/h for violation detection
        """
        self.vehicle_detector = vehicle_detector
        self.video_processor = video_processor
        self.roi = roi
        self.statistics = TrafficStatistics()
        self.statistics.start_time = datetime.now()
        
        # Initialize temporal smoother if enabled
        self.smoother = None
        if enable_smoothing:
            self.smoother = TemporalSmoother(
                window_size=smoothing_window,
                drop_threshold=drop_threshold
            )
        
        # Initialize tracking if enabled
        self.enable_tracking = enable_tracking
        self.tracker = None
        self.speed_calibration = speed_calibration
        self.speed_violation_detector = None
        
        if enable_tracking:
            self.tracker = VehicleTracker()
            if speed_calibration:
                self.speed_violation_detector = SpeedViolationDetector(speed_limit_kmh=speed_limit_kmh)
    
    def analyze_video(
        self,
        video_path: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        frame_callback: Optional[Callable] = None,
        skip_frames: int = 0
    ) -> TrafficStatistics:
        """
        Analyze traffic in video.
        
        Args:
            video_path: Path to video file
            progress_callback: Optional callback function(frame_number, total_frames)
            frame_callback: Optional callback function(frame, vehicles, frame_stats) for visualization
            skip_frames: Number of frames to skip between processing (0 = process all)
            
        Returns:
            TrafficStatistics object with analysis results
        """
        if not self.video_processor.open_video(video_path):
            raise ValueError(f"Could not open video: {video_path}")
        
        frame_width, frame_height = self.video_processor.get_frame_size()
        total_frames = self.video_processor.get_frame_count()
        fps = self.video_processor.get_fps()
        
        frame_number = 0
        processed_frame_number = 0
        total_vehicles_counted = 0
        vehicles_by_type = {}
        
        while True:
            frame = self.video_processor.read_frame()
            if frame is None:
                break
            
            # Skip frames if requested
            if skip_frames > 0 and frame_number % (skip_frames + 1) != 0:
                frame_number += 1
                continue
            
            # Detect vehicles
            vehicles = self.vehicle_detector.detect_vehicles(frame)
            
            # Calculate frame statistics
            timestamp = frame_number / fps if fps > 0 else 0.0
            frame_stats = CountVehiclesUseCase.create_frame_statistics(
                frame_number=frame_number,
                timestamp=timestamp,
                vehicles=vehicles,
                roi=self.roi,
                frame_width=frame_width if self.roi is None else None,
                frame_height=frame_height if self.roi is None else None
            )
            
            # Track vehicles if enabled
            tracked_vehicles: List[TrackedVehicle] = []
            if self.enable_tracking and self.tracker:
                tracked_vehicles = self.tracker.update(
                    vehicles, frame_number, timestamp, self.speed_calibration
                )
                frame_stats.tracked_vehicles_count = len(tracked_vehicles)
                
                # Count by direction
                direction_counts = DirectionCounter.count_by_direction(tracked_vehicles)
                frame_stats.vehicles_by_direction = direction_counts
                
                # Check speed violations
                if self.speed_violation_detector:
                    violations = self.speed_violation_detector.get_violations(tracked_vehicles)
                    frame_stats.speed_violations_count = len(violations)
            
            # Apply temporal smoothing to filter occlusion events (e.g., bird blocking view)
            if self.smoother:
                frame_stats = self.smoother.smooth_frame_statistics(frame_stats)
            
            self.statistics.frame_statistics.append(frame_stats)
            
            # Update overall statistics
            total_vehicles_counted += len(vehicles)
            for vehicle_type, count in frame_stats.vehicles_by_type.items():
                vehicles_by_type[vehicle_type] = vehicles_by_type.get(vehicle_type, 0) + count
            
            if frame_stats.vehicle_count > self.statistics.max_vehicles_in_frame:
                self.statistics.max_vehicles_in_frame = frame_stats.vehicle_count
            
            if frame_stats.density > self.statistics.peak_density:
                self.statistics.peak_density = frame_stats.density
            
            # Frame callback for visualization
            if frame_callback:
                # Pass ROI and tracked vehicles to callback if it accepts them
                try:
                    should_stop = frame_callback(
                        frame, vehicles, frame_stats, frame_number,
                        roi=self.roi, tracked_vehicles=tracked_vehicles
                    )
                except TypeError:
                    try:
                        # Try with just ROI
                        should_stop = frame_callback(
                            frame, vehicles, frame_stats, frame_number, roi=self.roi
                        )
                    except TypeError:
                        # Backward compatibility: if callback doesn't accept new parameters
                        should_stop = frame_callback(frame, vehicles, frame_stats, frame_number)
                if should_stop is True:
                    break
            
            processed_frame_number += 1
            frame_number += 1
            
            # Progress callback
            if progress_callback:
                progress_callback(frame_number, total_frames)
        
        # Finalize statistics
        self.statistics.total_vehicles = total_vehicles_counted
        self.statistics.vehicles_by_type = vehicles_by_type
        self.statistics.average_vehicles_per_frame = (
            total_vehicles_counted / processed_frame_number if processed_frame_number > 0 else 0.0
        )
        
        # Calculate speed and direction statistics if tracking was enabled
        if self.enable_tracking and self.tracker:
            all_tracked = []
            for tracked in self.tracker.tracked_vehicles.values():
                if len(tracked.positions) >= 2:  # Only count vehicles with movement
                    all_tracked.append(tracked)
            
            self.statistics.total_tracked_vehicles = len(all_tracked)
            
            # Direction statistics
            direction_counts = DirectionCounter.count_by_direction(all_tracked)
            self.statistics.vehicles_by_direction = direction_counts
            
            # Speed statistics
            speeds = []
            speeds_by_direction = {}
            for tracked in all_tracked:
                if tracked.average_speed_kmh > 0:
                    speeds.append(tracked.average_speed_kmh)
                    direction = tracked.current_direction
                    if direction not in speeds_by_direction:
                        speeds_by_direction[direction] = []
                    speeds_by_direction[direction].append(tracked.average_speed_kmh)
            
            if speeds:
                self.statistics.average_speed_kmh = sum(speeds) / len(speeds)
                self.statistics.max_speed_kmh = max(speeds)
            
            # Speed statistics by direction
            for direction, dir_speeds in speeds_by_direction.items():
                if dir_speeds:
                    self.statistics.speed_statistics_by_direction[direction] = {
                        "average": sum(dir_speeds) / len(dir_speeds),
                        "max": max(dir_speeds),
                        "min": min(dir_speeds),
                        "count": len(dir_speeds)
                    }
            
            # Speed violations
            if self.speed_violation_detector:
                self.statistics.total_speed_violations = self.speed_violation_detector.count_violations(all_tracked)
        
        self.statistics.end_time = datetime.now()
        
        self.video_processor.release()
        
        return self.statistics

