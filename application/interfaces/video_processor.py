"""Interface for video processing."""

from abc import ABC, abstractmethod
from typing import Optional, Callable


class IVideoProcessor(ABC):
    """Interface for video processing service."""
    
    @abstractmethod
    def open_video(self, video_path: str) -> bool:
        """Open video file."""
        pass
    
    @abstractmethod
    def read_frame(self):
        """Read next frame from video."""
        pass
    
    @abstractmethod
    def get_frame_count(self) -> int:
        """Get total number of frames."""
        pass
    
    @abstractmethod
    def get_fps(self) -> float:
        """Get frames per second."""
        pass
    
    @abstractmethod
    def get_frame_size(self) -> tuple[int, int]:
        """Get frame dimensions (width, height)."""
        pass
    
    @abstractmethod
    def release(self):
        """Release video resources."""
        pass

