"""Main entry point for traffic analysis application."""

import sys
import argparse
import json
from pathlib import Path
from typing import Optional

import cv2

from domain.entities.roi import ROI
from domain.entities.speed_calibration import SpeedCalibration
from infrastructure.detection.yolo_vehicle_detector import YOLOVehicleDetector
from infrastructure.video.opencv_video_processor import OpenCVVideoProcessor
from application.services.traffic_analyzer import TrafficAnalyzerService
from presentation.visualization.video_visualizer import VideoVisualizer
from presentation.reporting.statistics_reporter import StatisticsReporter


def progress_callback(current_frame: int, total_frames: int):
    """Progress callback for video analysis."""
    if total_frames > 0:
        progress = (current_frame / total_frames) * 100
        print(f"\rProcessing: {current_frame}/{total_frames} frames ({progress:.1f}%)", end='', flush=True)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Traffic Analysis - Vehicle Counting and Density Analysis')
    parser.add_argument(
        '--video',
        type=str,
        default='veriseti_2.mp4',
        help='Path to input video file (default: veriseti_2.mp4)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='yolov8n.pt',
        help='Path to YOLO model file (default: yolov8n.pt)'
    )
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.25,
        help='Confidence threshold for detection (default: 0.25)'
    )
    parser.add_argument(
        '--output-video',
        type=str,
        default=None,
        help='Path to output video with visualizations (optional)'
    )
    parser.add_argument(
        '--output-json',
        type=str,
        default='statistics.json',
        help='Path to output JSON statistics file (default: statistics.json)'
    )
    parser.add_argument(
        '--no-display',
        dest='display',
        action='store_false',
        default=True,
        help='Disable video display during processing'
    )
    parser.add_argument(
        '--display',
        dest='display',
        action='store_true',
        help='Display video during processing (default: enabled)'
    )
    parser.add_argument(
        '--skip-frames',
        type=int,
        default=0,
        help='Number of frames to skip between processing (0=process all, 1=process every 2nd frame, etc.)'
    )
    parser.add_argument(
        '--roi',
        type=str,
        default=None,
        help='ROI coordinates as comma-separated values: x1,y1,x2,y2,x3,y3,... (e.g., "100,200,800,200,900,600,50,600") or path to JSON file from select_roi.py. If not provided, uses full frame.'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode to see filtered non-vehicle objects (birds, people, etc.)'
    )
    parser.add_argument(
        '--no-smoothing',
        dest='enable_smoothing',
        action='store_false',
        default=True,
        help='Disable temporal smoothing (not recommended - smoothing filters occlusion events)'
    )
    parser.add_argument(
        '--smoothing-window',
        type=int,
        default=5,
        help='Number of frames to consider for smoothing (default: 5)'
    )
    parser.add_argument(
        '--drop-threshold',
        type=float,
        default=0.5,
        help='Threshold for detecting sudden drops (0.0-1.0, default: 0.5 = 50%% drop)'
    )
    parser.add_argument(
        '--no-tracking',
        dest='enable_tracking',
        action='store_false',
        default=True,
        help='Disable vehicle tracking (tracking enables speed and direction analysis)'
    )
    parser.add_argument(
        '--pixels-per-meter',
        type=float,
        default=None,
        help='Calibration: pixels per meter for speed calculation. If not provided, speed will be in pixels/second only.'
    )
    parser.add_argument(
        '--reference-distance',
        type=str,
        default=None,
        help='Calibration: reference distance as "pixels,meters" (e.g., "100,3.5" means 100 pixels = 3.5 meters)'
    )
    parser.add_argument(
        '--roi-width-lanes',
        type=str,
        default=None,
        help='Calibration: ROI width and lanes as "width_pixels,num_lanes,lane_width_meters" (e.g., "800,2,3.5")'
    )
    parser.add_argument(
        '--speed-limit',
        type=float,
        default=50.0,
        help='Speed limit in km/h for violation detection (default: 50.0)'
    )
    
    args = parser.parse_args()
    
    # Check if video file exists
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)
    
    print("Initializing components...")
    
    # Parse ROI if provided
    roi = None
    if args.roi:
        try:
            # Check if it's a JSON file path
            roi_path = Path(args.roi)
            if roi_path.exists() and roi_path.suffix == '.json':
                with open(roi_path, 'r', encoding='utf-8') as f:
                    roi_data = json.load(f)
                polygon_points = [tuple(p) for p in roi_data['polygon_points']]
                roi = ROI(polygon_points)
                print(f"ROI loaded from {args.roi} with {len(polygon_points)} points")
            else:
                # Parse as comma-separated coordinates
                coords = [int(x.strip()) for x in args.roi.split(',')]
                if len(coords) < 6 or len(coords) % 2 != 0:
                    raise ValueError("ROI must have at least 3 points (6 coordinates) and even number of coordinates")
                polygon_points = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
                roi = ROI(polygon_points)
                print(f"ROI defined with {len(polygon_points)} points")
        except Exception as e:
            print(f"Error parsing ROI: {e}")
            print("Using full frame as ROI")
    else:
        print("ROI not specified. Using full frame area for density calculation.")
        print("Tip: Use --roi parameter or run 'python select_roi.py <video>' to define road area.")
    
    # Smoothing status
    if args.enable_smoothing:
        print(f"Temporal smoothing enabled (window: {args.smoothing_window} frames, drop threshold: {args.drop_threshold*100:.0f}%)")
        print("This will filter out sudden density drops caused by occlusion (e.g., birds blocking view).")
    else:
        print("Temporal smoothing disabled.")
    
    # Speed calibration setup
    speed_calibration = None
    if args.enable_tracking:
        print(f"\nVehicle tracking enabled (speed limit: {args.speed_limit} km/h)")
        
        # Try to create calibration from various methods
        if args.pixels_per_meter:
            # Direct calibration
            video_processor_temp = OpenCVVideoProcessor()
            if video_processor_temp.open_video(str(video_path)):
                fps = video_processor_temp.get_fps()
                speed_calibration = SpeedCalibration(
                    pixels_per_meter=args.pixels_per_meter,
                    fps=fps
                )
                print(f"Speed calibration: {args.pixels_per_meter} pixels/meter")
                video_processor_temp.release()
        elif args.reference_distance:
            # Reference distance calibration
            try:
                parts = args.reference_distance.split(',')
                if len(parts) == 2:
                    pixels = float(parts[0].strip())
                    meters = float(parts[1].strip())
                    video_processor_temp = OpenCVVideoProcessor()
                    if video_processor_temp.open_video(str(video_path)):
                        fps = video_processor_temp.get_fps()
                        speed_calibration = SpeedCalibration.from_reference_distance(
                            pixels, meters, fps
                        )
                        print(f"Speed calibration: {pixels} pixels = {meters} meters")
                        video_processor_temp.release()
            except Exception as e:
                print(f"Warning: Could not parse reference distance: {e}")
        elif args.roi_width_lanes and roi:
            # ROI-based calibration
            try:
                parts = args.roi_width_lanes.split(',')
                if len(parts) == 3:
                    width_pixels = float(parts[0].strip())
                    num_lanes = int(parts[1].strip())
                    lane_width = float(parts[2].strip())
                    video_processor_temp = OpenCVVideoProcessor()
                    if video_processor_temp.open_video(str(video_path)):
                        fps = video_processor_temp.get_fps()
                        speed_calibration = SpeedCalibration.from_roi_and_lane_width(
                            width_pixels, lane_width, num_lanes, fps
                        )
                        print(f"Speed calibration: {width_pixels} pixels = {num_lanes} lanes × {lane_width}m")
                        video_processor_temp.release()
            except Exception as e:
                print(f"Warning: Could not parse ROI width/lanes: {e}")
        
        if not speed_calibration:
            print("Note: No speed calibration provided. Speed will be calculated in pixels/second.")
            print("      This is fine for relative speed comparison, but not for real-world km/h values.")
            print("      For km/h conversion, use --pixels-per-meter, --reference-distance, or --roi-width-lanes")
            print(f"      Speed limit checking ({args.speed_limit} km/h) will be disabled without calibration.")
        else:
            print(f"Speed limit: {args.speed_limit} km/h")
    else:
        print("Vehicle tracking disabled.")
    
    # Initialize components (Dependency Injection)
    vehicle_detector = YOLOVehicleDetector(
        model_path=args.model,
        confidence_threshold=args.confidence,
        debug=args.debug
    )
    video_processor = OpenCVVideoProcessor()
    traffic_analyzer = TrafficAnalyzerService(
        vehicle_detector=vehicle_detector,
        video_processor=video_processor,
        roi=roi,
        enable_smoothing=args.enable_smoothing,
        smoothing_window=args.smoothing_window,
        drop_threshold=args.drop_threshold,
        enable_tracking=args.enable_tracking,
        speed_calibration=speed_calibration,
        speed_limit_kmh=args.speed_limit
    )
    visualizer = VideoVisualizer(
        show_confidence=True,
        show_count=True,
        show_speed=args.enable_tracking,
        show_direction=args.enable_tracking,
        show_tracking=args.enable_tracking
    )
    reporter = StatisticsReporter()
    
    print(f"Starting analysis of: {args.video}")
    if args.skip_frames > 0:
        print(f"Frame skipping: Processing every {args.skip_frames + 1} frame(s)")
    print("Processing video with real-time visualization...\n")
    
    # Setup video writer if output path provided
    video_writer = None
    if args.output_video:
        video_processor_temp = OpenCVVideoProcessor()
        if video_processor_temp.open_video(str(video_path)):
            fps = video_processor_temp.get_fps()
            width, height = video_processor_temp.get_frame_size()
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(args.output_video, fourcc, fps, (width, height))
            print(f"Writing output video to: {args.output_video}")
            video_processor_temp.release()
    
    # Frame callback for real-time visualization
    def frame_callback(frame, vehicles, frame_stats, frame_number, roi=None, tracked_vehicles=None):
        """Callback for each processed frame."""
        # Draw visualizations (including ROI and tracked vehicles if provided)
        visualized_frame = visualizer.draw_vehicles(
            frame, vehicles, frame_stats, roi=roi, tracked_vehicles=tracked_vehicles
        )
        
        # Write to output video if specified
        if video_writer:
            video_writer.write(visualized_frame)
        
        # Display frame if requested
        if args.display:
            cv2.imshow('Traffic Analysis - Press Q to quit', visualized_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return True  # Signal to stop
        return False
    
    # Analyze video with real-time visualization
    try:
        statistics = traffic_analyzer.analyze_video(
            video_path=str(video_path),
            progress_callback=progress_callback,
            frame_callback=frame_callback,
            skip_frames=args.skip_frames
        )
        print("\n")  # New line after progress
        
        # Cleanup
        if video_writer:
            video_writer.release()
        if args.display:
            cv2.destroyAllWindows()
        
        # Print summary
        reporter.print_summary(statistics)
        
        # Export to JSON
        reporter.export_to_json(statistics, args.output_json)
        
        print("\nAnalysis completed successfully!")
        
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

