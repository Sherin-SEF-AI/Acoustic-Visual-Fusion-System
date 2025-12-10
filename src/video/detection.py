"""
Object Detection Module using YOLOv8.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import cv2
from loguru import logger

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("ultralytics not available, detection disabled")


@dataclass
class Detection:
    """Represents a detected object."""
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    camera_id: str = ""
    timestamp: float = 0.0
    features: Optional[np.ndarray] = None
    
    @property
    def center(self) -> np.ndarray:
        return np.array([(self.bbox[0]+self.bbox[2])/2, (self.bbox[1]+self.bbox[3])/2])
    
    @property
    def area(self) -> float:
        return float((self.bbox[2]-self.bbox[0]) * (self.bbox[3]-self.bbox[1]))
    
    @property
    def width(self) -> float:
        return float(self.bbox[2] - self.bbox[0])
    
    @property
    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])


class ObjectDetector:
    """YOLOv8-based object detector for people and objects."""
    
    def __init__(self, model_name: str = "yolov8m.pt", confidence_threshold: float = 0.5,
                 classes: Optional[list[int]] = None, device: str = "auto"):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.target_classes = classes  # None = all classes
        self.device = device
        self.model = None
        
        # COCO class names
        self.class_names = {
            0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
            24: "backpack", 26: "handbag", 27: "tie", 28: "suitcase",
            39: "bottle", 41: "cup", 56: "chair", 57: "couch",
            58: "potted_plant", 59: "bed", 60: "dining_table", 62: "tv",
            63: "laptop", 64: "mouse", 65: "remote", 66: "keyboard",
            67: "cell_phone", 73: "book"
        }
        
        self._load_model()
    
    def _load_model(self):
        """Load YOLO model."""
        if not YOLO_AVAILABLE:
            logger.error("YOLO not available")
            return
        
        try:
            self.model = YOLO(self.model_name)
            if self.device == "auto":
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"YOLO model loaded: {self.model_name} on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load YOLO: {e}")
    
    def detect(self, image: np.ndarray, camera_id: str = "",
               timestamp: float = 0.0) -> list[Detection]:
        """Detect objects in image."""
        if self.model is None:
            return []
        
        try:
            results = self.model.predict(
                image, conf=self.confidence_threshold, 
                device=self.device, verbose=False
            )
            
            detections = []
            for r in results:
                boxes = r.boxes
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i])
                    if self.target_classes and cls_id not in self.target_classes:
                        continue
                    
                    bbox = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i])
                    cls_name = self.class_names.get(cls_id, f"class_{cls_id}")
                    
                    detections.append(Detection(
                        bbox=bbox, confidence=conf, class_id=cls_id,
                        class_name=cls_name, camera_id=camera_id, timestamp=timestamp
                    ))
            
            return detections
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []
    
    def detect_persons(self, image: np.ndarray, camera_id: str = "",
                       timestamp: float = 0.0) -> list[Detection]:
        """Detect only persons."""
        all_dets = self.detect(image, camera_id, timestamp)
        return [d for d in all_dets if d.class_id == 0]


class FallbackDetector:
    """Simple HOG-based person detector as fallback."""
    
    def __init__(self):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        logger.info("FallbackDetector (HOG) initialized")
    
    def detect(self, image: np.ndarray, camera_id: str = "",
               timestamp: float = 0.0) -> list[Detection]:
        """Detect persons using HOG."""
        boxes, weights = self.hog.detectMultiScale(
            image, winStride=(8, 8), padding=(4, 4), scale=1.05
        )
        
        detections = []
        for (x, y, w, h), conf in zip(boxes, weights):
            detections.append(Detection(
                bbox=np.array([x, y, x+w, y+h]),
                confidence=float(conf), class_id=0, class_name="person",
                camera_id=camera_id, timestamp=timestamp
            ))
        return detections
