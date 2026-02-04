import cv2
import numpy as np
from typing import Generator

import os

class Preprocessing:
    """
    Handles video file reading and frame extraction for object detection inference.

    This class reads a video from a file and preprocesses frames before passing them 
    to an object detection module for inference.
    """

    def __init__(self, filename: str, drop_rate: int = 10) -> None:
        """
        Initializes the Preprocessing class.

        :param filename: Path to the video file.
        :param drop_rate: The interval at which frames are selected. For example, 
                          `drop_rate=10` means every 10th frame is retained.
                          
        :ivar self.filename: Stores the video file path.
        :ivar self.drop_rate: Defines how frequently frames are extracted from the video.
        """
        self.filename = filename
        self.drop_rate = drop_rate

    def capture_video(self) -> Generator[np.ndarray, None, None]:
        """
        Captures frames from a video file and yields every nth frame.

        :return: A generator yielding frames as NumPy arrays.

        **Functionality:**
        - Opens a video file using OpenCV.
        - Iterates through each frame.
        - Yields every `drop_rate`-th frame.
        - Releases the video resource when finished.

        **Usage Example:**
        ```python
        video_processor = Preprocessing("video.mp4", drop_rate=10)
        for frame in video_processor.capture_video():
            process_frame(frame)  # Custom processing function (...think Detector Methods!)
        ```

        **Reference:**
        - OpenCV VideoCapture Documentation: 
          https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
        """
        # Case 1: directory of images
        if os.path.isdir(self.filename):
            frame_files = sorted(os.listdir(self.filename))
            frame_idx = 0

            for name in frame_files:
                # Skip non-image files
                if not name.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue

                # Drop frames based on drop_rate (e.g., drop_rate=2 keeps every 2nd frame)        
                if frame_idx % self.drop_rate != 0:
                    frame_idx += 1
                    continue

                frame_path = os.path.join(self.filename, name)
                frame = cv2.imread(frame_path)

                if frame is not None:
                    yield frame

                frame_idx += 1

            return

        # Case 2: video file
        cap = cv2.VideoCapture(self.filename)

        if not cap.isOpened():
            raise ValueError(f"Error: Unable to open video file '{self.filename}'.")

        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            
            # Yield frames at specified interval
            if frame_idx % self.drop_rate == 0:
                yield frame

            frame_idx += 1

        cap.release()
