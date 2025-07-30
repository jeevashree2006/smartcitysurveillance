import cv2
import numpy as np

# Try to import face_recognition, provide fallback if not available
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("Warning: face_recognition not available. Using OpenCV-only face detection.")

# Try to import torch, provide fallback if not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: torch not available.")

class PrivacyFilter:
    def __init__(self):
        """Initialize privacy filtering components"""
        self.face_model = None
        self.plate_cascade = None
        self.device = 'cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'
        self.load_models()
    
    def load_models(self):
        """Load models for face and license plate detection"""
        try:
            # Load Haar cascade for license plate detection (backup method)
            self.plate_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml'
            )
            
            # If the specific cascade is not available, use a general one
            if self.plate_cascade.empty():
                self.plate_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
            
            print("Privacy filter models loaded successfully")
            
        except Exception as e:
            print(f"Warning: Some privacy models could not be loaded: {e}")
    
    def apply_privacy_filters(self, frame, blur_faces=True, blur_plates=True, blur_intensity=10):
        """
        Apply privacy filters to blur faces and license plates
        
        Args:
            frame: Input image frame
            blur_faces: Whether to blur detected faces
            blur_plates: Whether to blur detected license plates
            blur_intensity: Intensity of blur effect
            
        Returns:
            Processed frame with privacy filters applied
        """
        filtered_frame = frame.copy()
        
        try:
            if blur_faces:
                filtered_frame = self._blur_faces(filtered_frame, blur_intensity)
            
            if blur_plates:
                filtered_frame = self._blur_license_plates(filtered_frame, blur_intensity)
                
        except Exception as e:
            print(f"Error applying privacy filters: {e}")
        
        return filtered_frame
    
    def _blur_faces(self, frame, blur_intensity):
        """Detect and blur faces in the frame"""
        try:
            if FACE_RECOGNITION_AVAILABLE:
                # Convert BGR to RGB for face_recognition library
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Find face locations using face_recognition library
                face_locations = face_recognition.face_locations(rgb_frame, model="hog")
                
                for (top, right, bottom, left) in face_locations:
                    # Extract face region
                    face_region = frame[top:bottom, left:right]
                    
                    # Apply Gaussian blur
                    blurred_face = cv2.GaussianBlur(face_region, (blur_intensity*2+1, blur_intensity*2+1), 0)
                    
                    # Replace the face region with blurred version
                    frame[top:bottom, left:right] = blurred_face
                
                # Fallback to Haar cascades if face_recognition fails or finds no faces
                if len(face_locations) == 0:
                    frame = self._blur_faces_cascade(frame, blur_intensity)
            else:
                # Use cascade method directly
                frame = self._blur_faces_cascade(frame, blur_intensity)
                
        except Exception as e:
            print(f"Error in face detection: {e}")
            # Fallback to cascade method
            frame = self._blur_faces_cascade(frame, blur_intensity)
        
        return frame
    
    def _blur_faces_cascade(self, frame, blur_intensity):
        """Fallback method using Haar cascades for face detection"""
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            for (x, y, w, h) in faces:
                # Extract face region
                face_region = frame[y:y+h, x:x+w]
                
                # Apply Gaussian blur
                blurred_face = cv2.GaussianBlur(face_region, (blur_intensity*2+1, blur_intensity*2+1), 0)
                
                # Replace the face region with blurred version
                frame[y:y+h, x:x+w] = blurred_face
                
        except Exception as e:
            print(f"Error in cascade face detection: {e}")
        
        return frame
    
    def _blur_license_plates(self, frame, blur_intensity):
        """Detect and blur license plates in the frame"""
        try:
            # Method 1: Use custom license plate detection
            plates = self._detect_license_plates_custom(frame)
            
            for (x, y, w, h) in plates:
                # Extract plate region
                plate_region = frame[y:y+h, x:x+w]
                
                # Apply Gaussian blur
                blurred_plate = cv2.GaussianBlur(plate_region, (blur_intensity*2+1, blur_intensity*2+1), 0)
                
                # Replace the plate region with blurred version
                frame[y:y+h, x:x+w] = blurred_plate
            
            # Method 2: Use text detection as backup
            if len(plates) == 0:
                frame = self._blur_text_regions(frame, blur_intensity)
                
        except Exception as e:
            print(f"Error in license plate detection: {e}")
        
        return frame
    
    def _detect_license_plates_custom(self, frame):
        """Custom license plate detection using image processing"""
        plates = []
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply bilateral filter to reduce noise while preserving edges
            filtered = cv2.bilateralFilter(gray, 11, 17, 17)
            
            # Find edges using Canny edge detector
            edges = cv2.Canny(filtered, 30, 200)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Sort contours by area (largest first)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
            
            for contour in contours:
                # Approximate the contour
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
                
                # License plates typically have 4 corners
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect_ratio = w / h
                    area = w * h
                    
                    # License plates have specific aspect ratios and minimum size
                    if 2.0 < aspect_ratio < 5.0 and area > 500:
                        plates.append((x, y, w, h))
            
            # Use Haar cascade as backup
            if self.plate_cascade and not self.plate_cascade.empty():
                cascade_plates = self.plate_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(50, 20)
                )
                plates.extend(cascade_plates)
                
        except Exception as e:
            print(f"Error in custom plate detection: {e}")
        
        return plates
    
    def _blur_text_regions(self, frame, blur_intensity):
        """Detect and blur text regions that might contain license plates"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Use MSER (Maximally Stable Extremal Regions) to detect text-like regions
            mser = cv2.MSER_create()
            regions, _ = mser.detectRegions(gray)
            
            for region in regions:
                # Get bounding box of the region
                x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))
                aspect_ratio = w / h
                area = w * h
                
                # Filter for license plate-like dimensions
                if 1.5 < aspect_ratio < 6.0 and 200 < area < 5000:
                    # Extract region
                    text_region = frame[y:y+h, x:x+w]
                    
                    # Apply blur
                    blurred_region = cv2.GaussianBlur(text_region, (blur_intensity*2+1, blur_intensity*2+1), 0)
                    
                    # Replace the region
                    frame[y:y+h, x:x+w] = blurred_region
                    
        except Exception as e:
            print(f"Error in text region detection: {e}")
        
        return frame
    
    def get_privacy_statistics(self, frame, blur_faces=True, blur_plates=True):
        """
        Get statistics about privacy elements detected in the frame
        
        Returns:
            Dictionary with counts of faces and license plates detected
        """
        stats = {
            'faces_detected': 0,
            'plates_detected': 0,
            'privacy_applied': blur_faces or blur_plates
        }
        
        try:
            if blur_faces:
                if FACE_RECOGNITION_AVAILABLE:
                    # Count faces using face_recognition
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    face_locations = face_recognition.face_locations(rgb_frame, model="hog")
                    stats['faces_detected'] = len(face_locations)
                    
                    # Fallback to cascade if no faces found
                    if stats['faces_detected'] == 0:
                        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                        stats['faces_detected'] = len(faces)
                else:
                    # Use cascade method directly
                    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                    stats['faces_detected'] = len(faces)
            
            if blur_plates:
                # Count license plates
                plates = self._detect_license_plates_custom(frame)
                stats['plates_detected'] = len(plates)
                
        except Exception as e:
            print(f"Error getting privacy statistics: {e}")
        
        return stats
