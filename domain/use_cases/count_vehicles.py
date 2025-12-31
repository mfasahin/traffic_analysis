"""Use case for counting vehicles in a frame."""

from typing import Dict, List, Optional
from domain.entities.vehicle import Vehicle, VehicleType
from domain.entities.traffic_statistics import FrameStatistics
from domain.entities.roi import ROI
from domain.entities.multi_roi import MultiROI
from domain.entities.vehicle_size import VehicleSize


class CountVehiclesUseCase:
    """Use case for counting vehicles."""
    
    @staticmethod
    def count_vehicles(vehicles: List[Vehicle]) -> int:
        """Count total number of vehicles."""
        return len(vehicles)
    
    @staticmethod
    def count_by_type(vehicles: List[Vehicle]) -> Dict[VehicleType, int]:
        """Count vehicles by type."""
        counts = {vt: 0 for vt in VehicleType}
        for vehicle in vehicles:
            counts[vehicle.vehicle_type] = counts.get(vehicle.vehicle_type, 0) + 1
        return counts
    
    @staticmethod
    def count_by_size(vehicles: List[Vehicle]) -> Dict[VehicleSize, int]:
        """Count vehicles by size."""
        counts = {vs: 0 for vs in VehicleSize}
        for vehicle in vehicles:
            if vehicle.size:
                counts[vehicle.size] = counts.get(vehicle.size, 0) + 1
        return counts
    
    @staticmethod
    def calculate_density(
        vehicles: List[Vehicle], 
        roi: Optional[ROI] = None,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None
    ) -> float:
        """
        Calculate real vehicle density based on area coverage.
        
        Formula: Density (%) = (∑(Vehicle Areas inside ROI) / ROI Area) × 100
        
        This method calculates the percentage of the road area (ROI) that is 
        occupied by vehicles, providing a more accurate density measurement
        than simple vehicle counting. A large truck will fill more area than
        a motorcycle, giving more precise density data.
        
        Only vehicles that are inside the ROI polygon are counted.
        
        Args:
            vehicles: List of detected vehicles
            roi: Region of Interest (road area polygon). If None, uses full frame.
            frame_width: Frame width (used if roi is None)
            frame_height: Frame height (used if roi is None)
            
        Returns:
            Density as percentage (0-100)
        """
        if not vehicles:
            return 0.0
        
        # Filter vehicles that are inside ROI
        if roi is not None:
            vehicles_inside_roi = [
                vehicle for vehicle in vehicles 
                if roi.contains_point(*vehicle.center)
            ]
            # Calculate total vehicle area (only for vehicles inside ROI)
            total_vehicle_area = sum(vehicle.get_area() for vehicle in vehicles_inside_roi)
            roi_area = roi.get_area()
        else:
            # Fallback to full frame area for backward compatibility
            # All vehicles are counted since ROI is the full frame
            total_vehicle_area = sum(vehicle.get_area() for vehicle in vehicles)
            if frame_width is None or frame_height is None:
                raise ValueError("Either roi or both frame_width and frame_height must be provided")
            roi_area = frame_width * frame_height
        
        if roi_area == 0:
            return 0.0
        
        # Calculate density percentage
        density_percentage = (total_vehicle_area / roi_area) * 100.0
        
        return density_percentage
    
    @staticmethod
    def create_frame_statistics(
        frame_number: int,
        timestamp: float,
        vehicles: List[Vehicle],
        roi: Optional[ROI] = None,
        multi_roi: Optional[MultiROI] = None,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None
    ) -> FrameStatistics:
        """
        Create frame statistics.
        
        Args:
            frame_number: Frame number
            timestamp: Timestamp of the frame
            vehicles: List of detected vehicles
            roi: Region of Interest (road area polygon). If None, uses full frame.
            multi_roi: Multiple ROI regions (for multi-lane analysis). Takes precedence over roi.
            frame_width: Frame width (used if roi is None)
            frame_height: Frame height (used if roi is None)
            
        Returns:
            FrameStatistics object
        """
        counts_by_type = CountVehiclesUseCase.count_by_type(vehicles)
        counts_by_size = CountVehiclesUseCase.count_by_size(vehicles)
        
        # Use multi_roi if provided, otherwise use single roi
        if multi_roi:
            # Calculate density for each lane and overall
            vehicles_by_lane = multi_roi.get_vehicles_by_lane(vehicles)
            lane_counts = {lane_id: len(veh_list) for lane_id, veh_list in vehicles_by_lane.items()}
            
            # Calculate overall density (sum of all lane areas)
            total_vehicle_area = sum(vehicle.get_area() for vehicle in vehicles)
            total_roi_area = multi_roi.get_total_area()
            density = (total_vehicle_area / total_roi_area * 100.0) if total_roi_area > 0 else 0.0
            
            return FrameStatistics(
                frame_number=frame_number,
                timestamp=timestamp,
                vehicle_count=len(vehicles),
                vehicles_by_type=counts_by_type,
                vehicles_by_size=counts_by_size,
                vehicles_by_lane=lane_counts,
                density=density
            )
        else:
            # Single ROI or full frame
            density = CountVehiclesUseCase.calculate_density(
                vehicles, 
                roi=roi,
                frame_width=frame_width,
                frame_height=frame_height
            )
            
            return FrameStatistics(
                frame_number=frame_number,
                timestamp=timestamp,
                vehicle_count=len(vehicles),
                vehicles_by_type=counts_by_type,
                vehicles_by_size=counts_by_size,
                density=density
            )

