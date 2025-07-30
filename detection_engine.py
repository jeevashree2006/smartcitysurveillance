import cv2
import numpy as np
from datetime import datetime
import os

# Try to import ultralytics, provide fallback if not available
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Warning: ultralytics not available. Using fallback detection methods.")

# Try to import torch, provide fallback if not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: torch not available. Using CPU-only mode.")

class DetectionEngine:
    def __init__(self):
        """Initialize the detection engine"""
        self.model = None
        self.custom_model = None
        self.device = 'cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'
        self.load_models()
        
        # Define target classes for smart city surveillance
        self.target_classes = {
            'graffiti': [],  # Will be detected by custom model
            'posters': [],   # Will be detected by custom model
            'dustbins': [39],  # COCO class for bottles (closest to dustbins)
            'vehicles': [2, 3, 5, 7],  # car, motorcycle, bus, truck
            'people': [0],   # person
            'bicycles': [1], # bicycle
        }
        
        # COCO class names for reference
        self.coco_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
    
    def load_models(self):
        """Load detection models"""
        try:
            if ULTRALYTICS_AVAILABLE:
                # Load pre-trained YOLOv8 model
                self.model = YOLO('yolov8n.pt')  # Using nano version for faster processing
                print(f"YOLOv8 model loaded successfully on {self.device}")
                
                # For custom objects like graffiti and posters, we'll use the base model
                # and implement custom detection logic based on specific patterns
                self.custom_model = self.model
            else:
                print("Using fallback detection methods without YOLO")
                self.model = None
                self.custom_model = None
            
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Falling back to basic computer vision methods")
            self.model = None
            self.custom_model = None
    
    def detect_objects(self, frame, confidence_threshold=0.5, detection_targets=None):
        """
        Detect objects in the frame using YOLOv8
        
        Args:
            frame: Input image frame
            confidence_threshold: Minimum confidence score
            detection_targets: List of target objects to detect
            
        Returns:
            List of detection results
        """
        if self.model is None:
            return []
        
        if detection_targets is None:
            detection_targets = ['people', 'vehicles', 'dustbins']
        
        detections = []
        
        try:
            if self.model is not None and ULTRALYTICS_AVAILABLE:
                # Run YOLOv8 inference
                results = self.model(frame, conf=confidence_threshold, device=self.device)
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            # Extract detection information
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = box.conf[0].cpu().numpy()
                            class_id = int(box.cls[0].cpu().numpy())
                            class_name = self.coco_classes[class_id] if class_id < len(self.coco_classes) else "unknown"
                            
                            # Check if this detection matches our target classes
                            detected_category = self._classify_detection(class_id, detection_targets)
                            
                            if detected_category:
                                detection = {
                                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                    'confidence': float(confidence),
                                    'class_id': class_id,
                                    'class_name': class_name,
                                    'category': detected_category,
                                    'timestamp': datetime.now().isoformat(),
                                    'location': 'Camera_01'  # This could be made configurable
                                }
                                detections.append(detection)
            else:
                # Use fallback detection methods
                detections.extend(self._fallback_object_detection(frame, confidence_threshold, detection_targets))
            
            # Add custom detections for graffiti and posters if requested
            if 'Graffiti' in detection_targets or 'Posters' in detection_targets:
                custom_detections = self._detect_custom_objects(frame, confidence_threshold, detection_targets)
                detections.extend(custom_detections)
            
        except Exception as e:
            print(f"Error during detection: {e}")
        
        return detections
    
    def _classify_detection(self, class_id, detection_targets):
        """Classify detected object into surveillance categories"""
        category_mapping = {
            'People': 'people',
            'Vehicles': 'vehicles', 
            'Dustbins': 'dustbins',
            'Bicycles': 'bicycles'
        }
        
        for target in detection_targets:
            if target in category_mapping:
                target_key = category_mapping[target]
                if class_id in self.target_classes.get(target_key, []):
                    return target
        
        return None
    
    def _detect_custom_objects(self, frame, confidence_threshold, detection_targets):
        """
        Detect custom objects like graffiti and posters using image processing techniques
        This is a simplified implementation - in production, you'd use a trained custom model
        """
        custom_detections = []
        
        try:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect graffiti-like patterns (bright colors, irregular shapes)
            if 'Graffiti' in detection_targets:
                graffiti_detections = self._detect_graffiti_patterns(frame, hsv, gray)
                custom_detections.extend(graffiti_detections)
            
            # Detect poster-like patterns (rectangular shapes with text/images)
            if 'Posters' in detection_targets:
                poster_detections = self._detect_poster_patterns(frame, gray)
                custom_detections.extend(poster_detections)
                
        except Exception as e:
            print(f"Error in custom object detection: {e}")
        
        return custom_detections
    
    def _detect_graffiti_patterns(self, frame, hsv, gray):
        """Detect potential graffiti using color and edge detection"""
        detections = []
        
        try:
            # Define color ranges for typical graffiti colors (bright colors)
            color_ranges = [
                ([100, 100, 100], [130, 255, 255]),  # Blue range
                ([0, 100, 100], [10, 255, 255]),     # Red range
                ([170, 100, 100], [180, 255, 255]),  # Red range (wrap around)
                ([40, 100, 100], [80, 255, 255]),    # Green range
                ([15, 100, 100], [35, 255, 255]),    # Yellow range
            ]
            
            combined_mask = np.zeros(gray.shape, dtype=np.uint8)
            
            for lower, upper in color_ranges:
                lower = np.array(lower)
                upper = np.array(upper)
                mask = cv2.inRange(hsv, lower, upper)
                combined_mask = cv2.bitwise_or(combined_mask, mask)
            
            # Apply morphological operations to clean up the mask
            kernel = np.ones((5, 5), np.uint8)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Minimum area threshold
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Calculate a confidence score based on area and color intensity
                    roi_mask = combined_mask[y:y+h, x:x+w]
                    confidence = np.sum(roi_mask) / (w * h * 255) * 0.8  # Max confidence of 0.8
                    
                    if confidence > 0.3:  # Minimum confidence threshold
                        detection = {
                            'bbox': [x, y, x+w, y+h],
                            'confidence': float(confidence),
                            'class_id': -1,  # Custom class ID
                            'class_name': 'graffiti',
                            'category': 'Graffiti',
                            'timestamp': datetime.now().isoformat(),
                            'location': 'Camera_01'
                        }
                        detections.append(detection)
            
        except Exception as e:
            print(f"Error detecting graffiti: {e}")
        
        return detections
    
    def _detect_poster_patterns(self, frame, gray):
        """Detect potential posters using edge detection and geometric analysis"""
        detections = []
        
        try:
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Find lines using Hough transform
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None:
                # Look for rectangular patterns that could be posters
                # This is a simplified approach - in production, use more sophisticated methods
                
                # Apply contour detection on edges
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    # Approximate the contour
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # Look for rectangular shapes (4 corners)
                    if len(approx) == 4:
                        area = cv2.contourArea(contour)
                        if area > 1000:  # Minimum area for a poster
                            x, y, w, h = cv2.boundingRect(contour)
                            aspect_ratio = w / h
                            
                            # Posters typically have certain aspect ratios
                            if 0.5 < aspect_ratio < 2.0:
                                confidence = min(0.7, area / 10000)  # Scale confidence with area
                                
                                detection = {
                                    'bbox': [x, y, x+w, y+h],
                                    'confidence': float(confidence),
                                    'class_id': -2,  # Custom class ID
                                    'class_name': 'poster',
                                    'category': 'Posters',
                                    'timestamp': datetime.now().isoformat(),
                                    'location': 'Camera_01'
                                }
                                detections.append(detection)
            
        except Exception as e:
            print(f"Error detecting posters: {e}")
        
        return detections
    
    def draw_detections(self, frame, detections):
        """Draw detection bounding boxes and labels on the frame"""
        annotated_frame = frame.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            category = detection['category']
            class_name = detection['class_name']
            
            x1, y1, x2, y2 = bbox
            
            # Choose color based on category
            color_map = {
                'Graffiti': (0, 0, 255),      # Red
                'Posters': (255, 165, 0),     # Orange
                'Dustbins': (0, 255, 0),      # Green
                'Vehicles': (255, 0, 0),      # Blue
                'People': (255, 255, 0),      # Cyan
                'Bicycles': (128, 0, 128)     # Purple
            }
            
            color = color_map.get(category, (255, 255, 255))  # Default white
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{category}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Draw label background
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return annotated_frame
    
    def _fallback_object_detection(self, frame, confidence_threshold, detection_targets):
        """
        Fallback object detection using basic computer vision methods
        when YOLO is not available
        """
        detections = []
        
        try:
            # Convert to grayscale and HSV for analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Detect rectangular objects that could be dustbins or vehicles
            if 'Dustbins' in detection_targets or 'Vehicles' in detection_targets:
                detections.extend(self._detect_rectangular_objects(frame, gray, detection_targets))
            
            # Detect people using basic methods
            if 'People' in detection_targets:
                detections.extend(self._detect_people_fallback(frame, gray))
            
        except Exception as e:
            print(f"Error in fallback detection: {e}")
        
        return detections
    
    def _detect_rectangular_objects(self, frame, gray, detection_targets):
        """Detect rectangular objects using contour analysis"""
        detections = []
        
        try:
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # Minimum area threshold
                    # Approximate contour to polygon
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # Look for rectangular shapes
                    if len(approx) >= 4:
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = w / h
                        
                        # Classify based on aspect ratio and size
                        if 0.5 < aspect_ratio < 2.0 and area > 5000:
                            confidence = min(0.6, area / 20000)  # Scale confidence with area
                            
                            detection = {
                                'bbox': [x, y, x+w, y+h],
                                'confidence': float(confidence),
                                'class_id': -3,
                                'class_name': 'rectangular_object',
                                'category': 'Dustbins' if area < 15000 else 'Vehicles',
                                'timestamp': datetime.now().isoformat(),
                                'location': 'Camera_01'
                            }
                            detections.append(detection)
            
        except Exception as e:
            print(f"Error detecting rectangular objects: {e}")
        
        return detections
    
    def _detect_people_fallback(self, frame, gray):
        """Basic people detection using Haar cascades"""
        detections = []
        
        try:
            # Load face cascade as a proxy for people detection
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            if not face_cascade.empty():
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                
                for (x, y, w, h) in faces:
                    # Expand bounding box to include approximate body
                    body_x = max(0, x - w//2)
                    body_y = max(0, y)
                    body_w = min(frame.shape[1] - body_x, w * 2)
                    body_h = min(frame.shape[0] - body_y, h * 3)
                    
                    detection = {
                        'bbox': [body_x, body_y, body_x + body_w, body_y + body_h],
                        'confidence': 0.7,
                        'class_id': 0,
                        'class_name': 'person',
                        'category': 'People',
                        'timestamp': datetime.now().isoformat(),
                        'location': 'Camera_01'
                    }
                    detections.append(detection)
            
        except Exception as e:
            print(f"Error in people detection fallback: {e}")
        
        return detections
