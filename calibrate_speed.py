"""Interactive speed calibration tool for traffic analysis."""

import cv2
import sys
from pathlib import Path
from domain.entities.speed_calibration import SpeedCalibration
import json


class SpeedCalibrator:
    """Interactive tool for calibrating speed measurements."""
    
    def __init__(self, video_path: str):
        """Initialize calibrator with video."""
        self.video_path = video_path
        self.points = []
        self.drawing = False
        self.window_name = "Speed Calibration - Click two points to measure distance"
        self.measurement_pixels = 0.0
        self.measurement_meters = 0.0
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events."""
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 2:
                self.points.append((x, y))
                print(f"Point {len(self.points)}: ({x}, {y})")
                
                # Draw point
                cv2.circle(self.current_frame, (x, y), 5, (0, 255, 0), -1)
                
                # Draw line if we have 2 points
                if len(self.points) == 2:
                    cv2.line(self.current_frame, self.points[0], self.points[1], (0, 255, 0), 2)
                    # Calculate distance
                    dx = self.points[1][0] - self.points[0][0]
                    dy = self.points[1][1] - self.points[0][1]
                    self.measurement_pixels = (dx**2 + dy**2) ** 0.5
                    print(f"\nMeasured distance: {self.measurement_pixels:.2f} pixels")
                    print("Enter the real-world distance in meters:")
                
                cv2.imshow(self.window_name, self.current_frame)
    
    def calibrate(self):
        """Interactive calibration from video frame."""
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
        print("\n" + "="*70)
        print("SPEED CALIBRATION TOOL")
        print("="*70)
        print("\nBu araç, video'daki piksel mesafesini gerçek dünya mesafesine çevirmek için kullanılır.")
        print("\nAdımlar:")
        print("1. Video'da bilinen bir mesafeyi seçin (örn: şerit genişliği, yol işareti arası)")
        print("2. İki noktaya tıklayarak bu mesafeyi ölçün")
        print("3. Gerçek dünya mesafesini (metre cinsinden) girin")
        print("\nÖrnekler:")
        print("  - Standart şerit genişliği: 3.5 metre")
        print("  - Yol işareti arası mesafe: genellikle 50 metre")
        print("  - Yol kenarı çizgisi uzunluğu: genellikle 3-6 metre")
        print("\nKontroller:")
        print("  - Sol tık: Nokta seç")
        print("  - 'r': Ölçümü sıfırla")
        print("  - 'q': Bitir ve kaydet")
        print("  - ESC: İptal")
        print("="*70 + "\n")
        
        while True:
            display_frame = self.current_frame.copy()
            
            # Draw points and line
            for i, point in enumerate(self.points):
                cv2.circle(display_frame, point, 5, (0, 255, 0), -1)
                if i > 0:
                    cv2.line(display_frame, self.points[i-1], point, (0, 255, 0), 2)
            
            # Show measurement if we have 2 points
            if len(self.points) == 2:
                mid_x = (self.points[0][0] + self.points[1][0]) // 2
                mid_y = (self.points[0][1] + self.points[1][1]) // 2
                text = f"{self.measurement_pixels:.1f} pixels"
                cv2.putText(display_frame, text, (mid_x - 50, mid_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show instructions
            cv2.putText(display_frame, f"Points: {len(self.points)}/2", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if len(self.points) < 2:
                cv2.putText(display_frame, "Click two points to measure distance", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                cv2.putText(display_frame, "Press 'q' to finish, 'r' to reset", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow(self.window_name, display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                if len(self.points) == 2:
                    break
                else:
                    print("Please select 2 points first!")
            elif key == ord('r'):
                self.points = []
                self.current_frame = frame.copy()
                self.measurement_pixels = 0.0
                print("Measurement reset")
            elif key == 27:  # ESC
                print("Calibration cancelled")
                cv2.destroyAllWindows()
                return None
        
        cv2.destroyAllWindows()
        
        # Get real-world distance from user
        if len(self.points) == 2:
            try:
                meters = float(input(f"\nEnter real-world distance in meters (measured: {self.measurement_pixels:.2f} pixels): "))
                if meters <= 0:
                    print("Error: Distance must be positive")
                    return None
                
                self.measurement_meters = meters
                
                # Get FPS from video
                cap = cv2.VideoCapture(str(self.video_path))
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                
                # Create calibration
                calibration = SpeedCalibration.from_reference_distance(
                    self.measurement_pixels,
                    self.measurement_meters,
                    fps
                )
                
                print(f"\n✓ Calibration successful!")
                print(f"  Pixels per meter: {calibration.pixels_per_meter:.2f}")
                print(f"  Video FPS: {fps:.2f}")
                print(f"\nUse this in main.py:")
                print(f"  --pixels-per-meter {calibration.pixels_per_meter:.2f}")
                print(f"\nOr use reference distance:")
                print(f"  --reference-distance \"{self.measurement_pixels:.0f},{self.measurement_meters}\"")
                
                return calibration
            except ValueError:
                print("Error: Invalid input. Please enter a number.")
                return None
        
        return None
    
    def save_calibration(self, calibration: SpeedCalibration, output_file: str = "calibration.json"):
        """Save calibration to JSON file."""
        data = {
            "pixels_per_meter": calibration.pixels_per_meter,
            "fps": calibration.fps,
            "measurement_pixels": self.measurement_pixels,
            "measurement_meters": self.measurement_meters
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"\nCalibration saved to: {output_file}")


def main():
    """Main function for calibration tool."""
    if len(sys.argv) < 2:
        print("Usage: python calibrate_speed.py <video_path> [output_json]")
        print("Example: python calibrate_speed.py veriseti_2.mp4 calibration.json")
        sys.exit(1)
    
    video_path = Path(sys.argv[1])
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    output_file = sys.argv[2] if len(sys.argv) > 2 else "calibration.json"
    
    calibrator = SpeedCalibrator(str(video_path))
    calibration = calibrator.calibrate()
    
    if calibration:
        calibrator.save_calibration(calibration, output_file)
        print("\n" + "="*70)
        print("KALİBRASYON TAMAMLANDI!")
        print("="*70)
        print(f"\nArtık main.py'yi şu şekilde kullanabilirsiniz:")
        print(f"python main.py --video {video_path} --pixels-per-meter {calibration.pixels_per_meter:.2f}")
        print("="*70)


if __name__ == "__main__":
    main()

