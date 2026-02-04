import cv2
import numpy as np
from typing import List, Tuple


class Detector:
    """
    A class that represents an object detection model using OpenCV's DNN module
    with a YOLO-based architecture.
    """

    def __init__(self, weights_path: str, config_path: str, class_path: str, score_threshold: float=.5) -> None:
        """
        Initializes the YOLO model by loading the pre-trained network and class labels.

        :param weights_path: Path to the pre-trained YOLO weights file.
        :param config_path: Path to the YOLO configuration file.
        :param class_path: Path to the file containing class labels.

        :ivar self.net: The neural network model loaded from weights and config files.
        :ivar self.classes: A list of class labels loaded from the class_path file.
        :ivar self.img_height: Height of the input image/frame.
        :ivar self.img_width: Width of the input image/frame.
        """
        self.net = cv2.dnn.readNet(weights_path, config_path)

        # Load class labels
        with open(class_path, "r") as f:
            self.classes = [line.strip() for line in f if line.strip()]

        # No exception as empty file should be allowed 

        self.img_height: int = 0
        self.img_width: int = 0

        self.score_threshold = score_threshold

    def predict(self, preprocessed_frame: np.ndarray) -> List[np.ndarray]:
        """
        Runs the YOLO model on a single input frame and returns raw predictions.

        :param preprocessed_frame: A single image frame that has been preprocessed 
                                   for YOLO model inference (e.g., resized and normalized).

        :return: A list of NumPy arrays containing the raw output from the YOLO model.
                 Each output consists of multiple detections with bounding boxes, 
                 confidence scores, and class probabilities.

        :ivar self.img_height: The height of the input image/frame.
        :ivar self.img_width: The width of the input image/frame.

        **YOLO Output Format:**
        Each detection in the output contains:
        - First 4 values: Bounding box center x, center y, width, height.
        - 5th value: Confidence score.
        - Remaining values: Class probabilities for each detected object.

        **Reference:**
        - OpenCV YOLO Documentation: 
          https://opencv-tutorial.readthedocs.io/en/latest/yolo/yolo.html#create-a-blob
        """
        frame = np.asarray(preprocessed_frame) if preprocessed_frame is not None else None
        
        if frame is None or frame.size == 0 or frame.ndim < 2:
             raise ValueError("Input frame is empty")
        
        # Store frame dimensions for coordinate conversion later
        self.img_height, self.img_width = frame.shape[:2]
        
        # Convert frame to blob format (normalized, resized to 416x416)
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1 / 255.0,
            size=(416, 416),
            swapRB=True,
            crop=False,)
        
        self.net.setInput(blob)
        
        # Get output layer names for YOLO detection layers
        layer_names = self.net.getLayerNames()
        out_layer_ids = self.net.getUnconnectedOutLayers()

        # Flatten array (OpenCV returns inconsistent shapes)
        out_layer_ids = np.array(out_layer_ids).flatten()
        out_layer_names = [layer_names[i - 1] for i in out_layer_ids]

        # Run forward pass through the network
        outputs = self.net.forward(out_layer_names)
        return outputs

    def post_process(
        self, predict_output: List[np.ndarray]
    ) -> Tuple[List[List[int]], List[int], List[float], List[np.ndarray]]:
        """
        Processes the raw YOLO model predictions and filters out low-confidence detections.

        :param predict_output: A list of NumPy arrays containing raw predictions 
                               from the YOLO model.

        :return: A tuple containing:
            - **bboxes (List[List[int]])**: List of bounding boxes as `[x, y, width, height]`, 
              where (x, y) represents the top-left corner.
            - **class_ids (List[int])**: List of detected object class indices.
            - **confidence_scores (List[float])**: List of confidence scores for each detection.
            - **class_scores (List[np.ndarray])**: List of all class-specific confidence scores.

        **Post-processing steps:**
        1. Extract bounding box coordinates from YOLO output.
        2. Compute class probabilities and determine the most likely class.
        3. Filter out detections below the confidence threshold.
        4. Convert bounding box coordinates from center-based format to 
           top-left corner format.

        **Bounding Box Conversion:**
        YOLO outputs bounding box coordinates in the format:
        ```
        center_x, center_y, width, height
        ```
        This function converts them to:
        ```
        x, y, width, height
        ```
        where (x, y) is the top-left corner.

        **Reference:**
        - OpenCV YOLO Documentation: 
          https://opencv-tutorial.readthedocs.io/en/latest/yolo/yolo.html#create-a-blob
        """
        # Return empty lists if no predictions
        if not predict_output:
            return [], [], [], []
        
        bboxes: List[List[int]] = []
        class_ids: List[int] = []
        confidence_scores: List[float] = []
        class_scores: List[np.ndarray] = []

        # Process detections from each output layer
        for output in predict_output:
            for detection in output:
                # Extract class probabilities (everything after [cx, cy, w, h, objectness])
                scores = detection[5:]
                if scores.size == 0:
                    continue

                best_class_id = int(np.argmax(scores))
                best_score = float(scores[best_class_id])

                # Filter by class confidence threshold
                if float(detection[4]) <= self.score_threshold:
                    continue
                
                # Convert normalized coordinates to pixels
                cx = float(detection[0]) * self.img_width
                cy = float(detection[1]) * self.img_height
                w = float(detection[2]) * self.img_width
                h = float(detection[3]) * self.img_height

                # Convert center format to top-left corner format
                x = int(cx - (w / 2))
                y = int(cy - (h / 2))

                bboxes.append([x, y, int(w), int(h)])
                class_ids.append(best_class_id)
                confidence_scores.append(float(detection[4]))
                class_scores.append(scores)

        return bboxes, class_ids, confidence_scores, class_scores

"""
EXAMPLE USAGE:
model = Detector()

# Perform object detection on the current frame
predictions = self.detector.predict(frame)

# Extract bounding boxes, class IDs, confidence scores, and class-specific scores
bboxes, class_ids, confidence_scores, class_scores = self.detector.post_process(
    predictions
)
"""
