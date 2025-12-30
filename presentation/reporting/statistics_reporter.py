"""Statistics reporting module."""

from typing import Optional
from domain.entities.traffic_statistics import TrafficStatistics
from domain.entities.vehicle import VehicleType
from domain.entities.direction import Direction
import json


class StatisticsReporter:
    """Reporter for traffic statistics."""
    
    def __init__(self):
        """Initialize statistics reporter."""
        pass
    
    def print_summary(self, statistics: TrafficStatistics):
        """
        Print summary statistics to console.
        
        Args:
            statistics: TrafficStatistics object
        """
        print("\n" + "="*60)
        print("TRAFFIC ANALYSIS SUMMARY")
        print("="*60)
        print(f"Total Vehicles Detected: {statistics.total_vehicles}")
        print(f"Max Vehicles in Single Frame: {statistics.max_vehicles_in_frame}")
        print(f"Average Vehicles per Frame: {statistics.average_vehicles_per_frame:.2f}")
        print(f"Peak Density: {statistics.peak_density:.2f}%")
        
        if statistics.total_tracked_vehicles > 0:
            print(f"\n--- TRACKING STATISTICS ---")
            print(f"Total Tracked Vehicles: {statistics.total_tracked_vehicles}")
            if statistics.average_speed_kmh > 0:
                print(f"Average Speed: {statistics.average_speed_kmh:.1f} km/h")
            if statistics.max_speed_kmh > 0:
                print(f"Maximum Speed: {statistics.max_speed_kmh:.1f} km/h")
            if statistics.total_speed_violations > 0:
                print(f"Speed Violations: {statistics.total_speed_violations}")
        
        print(f"\n--- VEHICLES BY TYPE ---")
        for vehicle_type, count in statistics.vehicles_by_type.items():
            if count > 0:
                print(f"  {vehicle_type.value.upper()}: {count}")
        
        if statistics.vehicles_by_direction:
            print(f"\n--- VEHICLES BY DIRECTION ---")
            for direction, count in statistics.vehicles_by_direction.items():
                if count > 0:
                    print(f"  {direction.value.upper()}: {count}")
        
        if statistics.speed_statistics_by_direction:
            print(f"\n--- SPEED BY DIRECTION ---")
            for direction, stats in statistics.speed_statistics_by_direction.items():
                print(f"  {direction.value.upper()}:")
                print(f"    Average: {stats.get('average', 0):.1f} km/h")
                print(f"    Max: {stats.get('max', 0):.1f} km/h")
                print(f"    Min: {stats.get('min', 0):.1f} km/h")
                print(f"    Count: {stats.get('count', 0)}")
        
        if statistics.start_time and statistics.end_time:
            duration = statistics.end_time - statistics.start_time
            print(f"\n--- ANALYSIS INFO ---")
            print(f"Analysis Duration: {duration.total_seconds():.2f} seconds")
            print(f"Total Frames Analyzed: {len(statistics.frame_statistics)}")
        
        print("="*60 + "\n")
    
    def export_to_json(self, statistics: TrafficStatistics, output_path: str):
        """
        Export statistics to JSON file.
        
        Args:
            statistics: TrafficStatistics object
            output_path: Path to output JSON file
        """
        data = {
            "total_vehicles": statistics.total_vehicles,
            "max_vehicles_in_frame": statistics.max_vehicles_in_frame,
            "average_vehicles_per_frame": statistics.average_vehicles_per_frame,
            "peak_density": statistics.peak_density,
            "vehicles_by_type": {
                vt.value: count for vt, count in statistics.vehicles_by_type.items()
            },
            "total_frames": len(statistics.frame_statistics),
            "start_time": statistics.start_time.isoformat() if statistics.start_time else None,
            "end_time": statistics.end_time.isoformat() if statistics.end_time else None,
        }
        
        # Add tracking statistics if available
        if statistics.total_tracked_vehicles > 0:
            data["tracking"] = {
                "total_tracked_vehicles": statistics.total_tracked_vehicles,
                "average_speed_kmh": statistics.average_speed_kmh,
                "max_speed_kmh": statistics.max_speed_kmh,
                "total_speed_violations": statistics.total_speed_violations,
                "vehicles_by_direction": {
                    d.value: count for d, count in statistics.vehicles_by_direction.items()
                },
                "speed_statistics_by_direction": {
                    d.value: stats for d, stats in statistics.speed_statistics_by_direction.items()
                }
            }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Statistics exported to: {output_path}")

