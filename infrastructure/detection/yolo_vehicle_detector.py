"""YOLOv8 vehicle detector implementation."""

from typing import List
import numpy as np
from ultralytics import YOLO
from domain.entities.vehicle import Vehicle, VehicleType
from application.interfaces.vehicle_detector import IVehicleDetector


class YOLOVehicleDetector(IVehicleDetector):
    """YOLOv8 based vehicle detector."""
    
    # COCO dataset class IDs for vehicles
    VEHICLE_CLASS_IDS = {
        2: VehicleType.CAR,      # car
        3: VehicleType.MOTORCYCLE,  # motorcycle
        5: VehicleType.BUS,      # bus
        7: VehicleType.TRUCK,    # truck
    }
    
    def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.25):
        """
        Initialize YOLO vehicle detector.
        
        Args:
            model_path: Path to YOLO model file (or model name for download)
            confidence_threshold: Minimum confidence for detection
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.vehicle_id_counter = 0
    
    def detect_vehicles(self, frame) -> List[Vehicle]:
        """
        Detect vehicles in frame using YOLOv8.
        
        Args:
            frame: Input frame (numpy array)
            
        Returns:
            List of detected Vehicle entities
        """
        # Optimize for speed: smaller image size, half precision
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            verbose=False,
            imgsz=640,  # Smaller image size for faster processing
            half=False  # Set to True if GPU available for faster inference
        )
        vehicles = []
        
        for result in results:
            boxes = result.boxes
            for i, box in enumerate(boxes):
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                # Check if detected object is a vehicle
                if class_id in self.VEHICLE_CLASS_IDS:
                    vehicle_type = self.VEHICLE_CLASS_IDS[class_id]
                    
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    bbox = (int(x1), int(y1), int(x2), int(y2))
                    
                    # Create vehicle entity
                    vehicle = Vehicle(
                        id=self.vehicle_id_counter,
                        bbox=bbox,
                        confidence=confidence,
                        vehicle_type=vehicle_type,
                        center=(0, 0),  # Will be calculated in __post_init__
                        frame_number=0  # Will be set by caller if needed
                    )
                    vehicles.append(vehicle)
                    self.vehicle_id_counter += 1
        
        return vehicles

