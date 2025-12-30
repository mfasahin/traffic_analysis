"""Interactive ROI selector tool for defining road area."""

import cv2
import sys
import json
import numpy as np
from pathlib import Path
from domain.entities.roi import ROI


class ROISelector:
    """Interactive tool for selecting ROI polygon."""
    
    def __init__(self, video_path: str):
        """Initialize ROI selector with video."""
        self.video_path = video_path
        self.points = []
        self.drawing = False
        self.window_name = "ROI Selector - Click points to define polygon, Press 'q' when done"
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            print(f"Point {len(self.points)}: ({x}, {y})")
            
            # Draw point on frame
            cv2.circle(self.current_frame, (x, y), 5, (0, 0, 255), -1)
            
            # Draw lines between points
            if len(self.points) > 1:
                cv2.line(self.current_frame, self.points[-2], self.points[-1], (0, 0, 255), 2)
            
            cv2.imshow(self.window_name, self.current_frame)
    
    def select_roi(self):
        """Interactive ROI selection from video frame."""
        cap = cv2.VideoCapture(str(self.video_path))
        
        if not cap.isOpened():
            print(f"Error: Could not open video: {self.video_path}")
            return None
        
        # Read first frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from video")
            cap.release()
            return None
        
        self.current_frame = frame.copy()
        cap.release()
        
        # Create window and set mouse callback
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # Instructions
        print("\n" + "="*60)
        print("ROI SELECTOR")
        print("="*60)
        print("1. Click points on the frame to define the road polygon")
        print("2. Click at least 3 points")
        print("3. Press 'q' to finish and save ROI")
        print("4. Press 'r' to reset points")
        print("5. Press 'ESC' to cancel")
        print("="*60 + "\n")
        
        while True:
            display_frame = self.current_frame.copy()
            
            # Draw all points and lines
            for i, point in enumerate(self.points):
                cv2.circle(display_frame, point, 5, (0, 0, 255), -1)
                if i > 0:
                    cv2.line(display_frame, self.points[i-1], point, (0, 0, 255), 2)
            
            # Draw polygon if we have at least 3 points
            if len(self.points) >= 3:
                points_array = np.array(self.points, dtype=np.int32)
                overlay = display_frame.copy()
                cv2.fillPoly(overlay, [points_array], (0, 0, 255))
                cv2.addWeighted(overlay, 0.3, display_frame, 0.7, 0, display_frame)
                cv2.polylines(display_frame, [points_array], isClosed=True, color=(0, 0, 255), thickness=2)
            
            # Show instruction text
            cv2.putText(display_frame, f"Points: {len(self.points)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, "Press 'q' to finish, 'r' to reset", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow(self.window_name, display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                if len(self.points) >= 3:
                    break
                else:
                    print("Please select at least 3 points!")
            elif key == ord('r'):
                self.points = []
                self.current_frame = frame.copy()
                print("Points reset")
            elif key == 27:  # ESC
                print("Selection cancelled")
                cv2.destroyAllWindows()
                return None
        
        cv2.destroyAllWindows()
        
        if len(self.points) < 3:
            print("Error: Need at least 3 points for ROI")
            return None
        
        try:
            roi = ROI(self.points)
            print(f"\nROI created successfully!")
            print(f"Area: {roi.get_area()} pixels")
            print(f"Points: {self.points}")
            return roi
        except Exception as e:
            print(f"Error creating ROI: {e}")
            return None
    
    def save_roi_to_file(self, roi: ROI, output_file: str = "roi_coordinates.json"):
        """Save ROI coordinates to JSON file."""
        data = {
            "polygon_points": roi.polygon_points,
            "area": roi.get_area()
        }
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\nROI coordinates saved to: {output_file}")
        print(f"\nTo use this ROI, run:")
        coords_str = ','.join([f"{p[0]},{p[1]}" for p in roi.polygon_points])
        print(f'python main.py --roi "{coords_str}"')


def main():
    """Main function for ROI selector."""
    if len(sys.argv) < 2:
        print("Usage: python select_roi.py <video_path> [output_json]")
        print("Example: python select_roi.py veriseti_2.mp4 roi_coordinates.json")
        sys.exit(1)
    
    video_path = Path(sys.argv[1])
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    output_file = sys.argv[2] if len(sys.argv) > 2 else "roi_coordinates.json"
    
    selector = ROISelector(str(video_path))
    roi = selector.select_roi()
    
    if roi:
        selector.save_roi_to_file(roi, output_file)


if __name__ == "__main__":
    main()

