"""Multiple ROI support for analyzing multiple lanes/regions."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from domain.entities.roi import ROI


@dataclass
class LaneROI:
    """ROI for a single lane with metadata."""
    roi: ROI
    lane_id: str  # Unique identifier for the lane (e.g., "lane_1", "lane_2")
    lane_name: Optional[str] = None  # Human-readable name (e.g., "Northbound", "Southbound")
    direction: Optional[str] = None  # Expected direction for this lane
    color: Tuple[int, int, int] = (0, 255, 0)  # BGR color for visualization


@dataclass
class MultiROI:
    """Container for multiple ROI regions (e.g., multiple lanes)."""
    
    lanes: List[LaneROI] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate multi-ROI."""
        if not self.lanes:
            raise ValueError("MultiROI must have at least one lane")
        
        # Ensure unique lane IDs
        lane_ids = [lane.lane_id for lane in self.lanes]
        if len(lane_ids) != len(set(lane_ids)):
            raise ValueError("Lane IDs must be unique")
    
    def add_lane(self, lane: LaneROI):
        """Add a lane to multi-ROI."""
        # Check for duplicate IDs
        if any(l.lane_id == lane.lane_id for l in self.lanes):
            raise ValueError(f"Lane ID '{lane.lane_id}' already exists")
        self.lanes.append(lane)
    
    def get_lane_by_id(self, lane_id: str) -> Optional[LaneROI]:
        """Get lane by ID."""
        for lane in self.lanes:
            if lane.lane_id == lane_id:
                return lane
        return None
    
    def get_total_area(self) -> int:
        """Get total area of all lanes combined."""
        return sum(lane.roi.get_area() for lane in self.lanes)
    
    def find_lane_for_point(self, x: int, y: int) -> Optional[LaneROI]:
        """Find which lane contains the given point."""
        for lane in self.lanes:
            if lane.roi.contains_point(x, y):
                return lane
        return None
    
    def get_vehicles_by_lane(
        self,
        vehicles: List  # List of Vehicle objects
    ) -> Dict[str, List]:
        """
        Group vehicles by which lane they are in.
        
        Args:
            vehicles: List of vehicles to group
            
        Returns:
            Dictionary mapping lane_id to list of vehicles in that lane
        """
        vehicles_by_lane: Dict[str, List] = {lane.lane_id: [] for lane in self.lanes}
        
        for vehicle in vehicles:
            lane = self.find_lane_for_point(*vehicle.center)
            if lane:
                vehicles_by_lane[lane.lane_id].append(vehicle)
        
        return vehicles_by_lane
    
    @staticmethod
    def from_single_roi(roi: ROI, lane_id: str = "lane_1") -> 'MultiROI':
        """Create MultiROI from a single ROI (for backward compatibility)."""
        lane = LaneROI(roi=roi, lane_id=lane_id)
        return MultiROI(lanes=[lane])

