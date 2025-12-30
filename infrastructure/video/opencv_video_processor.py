"""OpenCV video processor implementation."""

import cv2
from typing import Optional, Tuple
from application.interfaces.video_processor import IVideoProcessor


class OpenCVVideoProcessor(IVideoProcessor):
    """OpenCV based video processor."""
    
    def __init__(self):
        """Initialize OpenCV video processor."""
        self.cap: Optional[cv2.VideoCapture] = None
        self.video_path: Optional[str] = None
    
    def open_video(self, video_path: str) -> bool:
        """
        Open video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            True if video opened successfully, False otherwise
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        return self.cap.isOpened()
    
    def read_frame(self):
        """
        Read next frame from video.
        
        Returns:
            Frame as numpy array, or None if end of video
        """
        if self.cap is None:
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        return frame
    
    def get_frame_count(self) -> int:
        """
        Get total number of frames.
        
        Returns:
            Total frame count
        """
        if self.cap is None:
            return 0
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def get_fps(self) -> float:
        """
        Get frames per second.
        
        Returns:
            FPS value
        """
        if self.cap is None:
            return 0.0
        return self.cap.get(cv2.CAP_PROP_FPS)
    
    def get_frame_size(self) -> Tuple[int, int]:
        """
        Get frame dimensions.
        
        Returns:
            Tuple of (width, height)
        """
        if self.cap is None:
            return (0, 0)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)
    
    def release(self):
        """Release video resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

