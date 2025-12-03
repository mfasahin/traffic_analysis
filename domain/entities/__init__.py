"""Domain entities."""

from domain.entities.roi import ROI
from domain.entities.traffic_statistics import TrafficStatistics, FrameStatistics
from domain.entities.vehicle import Vehicle, VehicleType

__all__ = ['ROI', 'TrafficStatistics', 'FrameStatistics', 'Vehicle', 'VehicleType']
