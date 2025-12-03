"""Main entry point for traffic analysis application."""

import sys
import argparse
from pathlib import Path
from typing import Optional
from infrastructure.detection.yolo_vehicle_detector import YOLOVehicleDetector
from infrastructure.video.opencv_video_processor import OpenCVVideoProcessor
from application.services.traffic_analyzer import TrafficAnalyzerService
from presentation.visualization.video_visualizer import VideoVisualizer
from presentation.reporting.statistics_reporter import StatisticsReporter
from domain.entities.roi import ROI
import cv2


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
                import json
                with open(roi_path, 'r') as f:
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
    
    # Initialize components (Dependency Injection)
    vehicle_detector = YOLOVehicleDetector(
        model_path=args.model,
        confidence_threshold=args.confidence
    )
    video_processor = OpenCVVideoProcessor()
    traffic_analyzer = TrafficAnalyzerService(
        vehicle_detector=vehicle_detector,
        video_processor=video_processor,
        roi=roi
    )
    visualizer = VideoVisualizer(show_confidence=True, show_count=True)
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
    def frame_callback(frame, vehicles, frame_stats, frame_number, roi=None):
        """Callback for each processed frame."""
        # Draw visualizations (including ROI if provided)
        visualized_frame = visualizer.draw_vehicles(frame, vehicles, frame_stats, roi=roi)
        
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

