import cv2
import threading
import queue
import time
import numpy as np
from datetime import datetime
import logging

class VideoProcessor:
    def __init__(self, source, detection_engine, privacy_filter, alert_system,
                 confidence_threshold=0.5, detection_targets=None, 
                 blur_faces=True, blur_plates=True, blur_intensity=10,
                 enable_alerts=True, alert_threshold=3):
        """
        Initialize video processor
        
        Args:
            source: Video source (camera index, file path, or URL)
            detection_engine: Object detection engine
            privacy_filter: Privacy filtering component
            alert_system: Alert system for notifications
            confidence_threshold: Minimum confidence for detections
            detection_targets: List of objects to detect
            blur_faces: Whether to blur faces
            blur_plates: Whether to blur license plates
            blur_intensity: Blur effect intensity
            enable_alerts: Whether to enable real-time alerts
            alert_threshold: Number of detections before alert
        """
        self.source = source
        self.detection_engine = detection_engine
        self.privacy_filter = privacy_filter
        self.alert_system = alert_system
        
        # Configuration
        self.confidence_threshold = confidence_threshold
        self.detection_targets = detection_targets or ['Graffiti', 'Posters', 'Dustbins']
        self.blur_faces = blur_faces
        self.blur_plates = blur_plates
        self.blur_intensity = blur_intensity
        self.enable_alerts = enable_alerts
        self.alert_threshold = alert_threshold
        
        # Video capture
        self.cap = None
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=10)
        self.detection_queue = queue.Queue()
        
        # Current frame and detections
        self.current_frame = None
        self.current_detections = []
        self.new_detections = []
        
        # Threading
        self.capture_thread = None
        self.process_thread = None
        
        # Statistics
        self.frame_count = 0
        self.detection_count = 0
        self.start_time = None
        
        # Alert tracking
        self.consecutive_detections = 0
        self.last_alert_time = None
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize video capture
        self._initialize_capture()
    
    def _initialize_capture(self):
        """Initialize video capture from source"""
        try:
            self.cap = cv2.VideoCapture(self.source)
            
            if not self.cap.isOpened():
                raise ValueError(f"Could not open video source: {self.source}")
            
            # Set capture properties for better performance
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Get video properties
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            self.logger.info(f"Video capture initialized: {self.width}x{self.height} @ {self.fps}fps")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize video capture: {e}")
            raise
    
    def start(self):
        """Start video processing"""
        if self.is_running:
            return
        
        self.is_running = True
        self.start_time = time.time()
        
        # Start capture thread
        self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
        self.capture_thread.start()
        
        # Start processing thread
        self.process_thread = threading.Thread(target=self._process_frames, daemon=True)
        self.process_thread.start()
        
        self.logger.info("Video processing started")
    
    def stop(self):
        """Stop video processing"""
        self.is_running = False
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        
        if self.process_thread and self.process_thread.is_alive():
            self.process_thread.join(timeout=2.0)
        
        if self.cap:
            self.cap.release()
        
        self.logger.info("Video processing stopped")
    
    def _capture_frames(self):
        """Capture frames from video source in a separate thread"""
        while self.is_running:
            try:
                ret, frame = self.cap.read()
                
                if not ret:
                    if isinstance(self.source, str) and not self.source.isdigit():
                        # For video files, break when finished
                        self.logger.info("End of video file reached")
                        break
                    else:
                        # For cameras, continue trying
                        continue
                
                # Resize frame if too large for better performance
                if frame.shape[1] > 1280:
                    scale = 1280 / frame.shape[1]
                    new_width = int(frame.shape[1] * scale)
                    new_height = int(frame.shape[0] * scale)
                    frame = cv2.resize(frame, (new_width, new_height))
                
                # Add to queue (non-blocking, drop frame if queue is full)
                try:
                    self.frame_queue.put_nowait(frame)
                    self.frame_count += 1
                except queue.Full:
                    # Drop the oldest frame
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame)
                    except queue.Empty:
                        pass
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.01)
                
            except Exception as e:
                self.logger.error(f"Error in frame capture: {e}")
                time.sleep(0.1)
        
        self.is_running = False
    
    def _process_frames(self):
        """Process frames for detection and privacy filtering in a separate thread"""
        while self.is_running:
            try:
                # Get frame from queue
                try:
                    frame = self.frame_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Apply privacy filters first
                filtered_frame = self.privacy_filter.apply_privacy_filters(
                    frame,
                    blur_faces=self.blur_faces,
                    blur_plates=self.blur_plates,
                    blur_intensity=self.blur_intensity
                )
                
                # Perform object detection
                detections = self.detection_engine.detect_objects(
                    frame,  # Use original frame for detection (better accuracy)
                    confidence_threshold=self.confidence_threshold,
                    detection_targets=self.detection_targets
                )
                
                # Draw detections on filtered frame
                if detections:
                    filtered_frame = self.detection_engine.draw_detections(filtered_frame, detections)
                    self.detection_count += len(detections)
                    
                    # Add to new detections for dashboard
                    self.new_detections.extend(detections)
                    
                    # Check for alerts
                    if self.enable_alerts:
                        self._check_alerts(detections)
                
                # Update current frame and detections
                self.current_frame = filtered_frame
                self.current_detections = detections
                
                # Add processing timestamp
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(filtered_frame, f"Time: {timestamp}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Add performance info
                if self.start_time:
                    elapsed_time = time.time() - self.start_time
                    fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
                    cv2.putText(filtered_frame, f"FPS: {fps:.1f}", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(filtered_frame, f"Detections: {self.detection_count}", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
            except Exception as e:
                self.logger.error(f"Error in frame processing: {e}")
                time.sleep(0.1)
    
    def _check_alerts(self, detections):
        """Check if alerts should be triggered based on detections"""
        alert_categories = ['Graffiti', 'Posters']
        alert_detections = [d for d in detections if d.get('category') in alert_categories]
        
        if alert_detections:
            self.consecutive_detections += len(alert_detections)
            
            # Trigger alert if threshold is reached
            if self.consecutive_detections >= self.alert_threshold:
                current_time = time.time()
                
                # Don't spam alerts - minimum 30 seconds between alerts
                if (self.last_alert_time is None or 
                    current_time - self.last_alert_time > 30):
                    
                    alert_message = f"Alert: {len(alert_detections)} suspicious objects detected"
                    self.alert_system.trigger_alert(alert_message, alert_detections)
                    
                    self.last_alert_time = current_time
                    self.consecutive_detections = 0  # Reset counter
        else:
            # Decay consecutive detections if no alerts found
            self.consecutive_detections = max(0, self.consecutive_detections - 1)
    
    def process_frame(self):
        """Get the latest processed frame"""
        return self.current_frame
    
    def get_current_frame(self):
        """Get the current frame without processing info"""
        if self.current_frame is not None:
            return self.current_frame.copy()
        return None
    
    def get_new_detections(self):
        """Get and clear new detections since last call"""
        detections = self.new_detections.copy()
        self.new_detections.clear()
        return detections
    
    def get_current_detections(self):
        """Get detections from the current frame"""
        return self.current_detections.copy()
    
    def get_statistics(self):
        """Get processing statistics"""
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        return {
            'frames_processed': self.frame_count,
            'detections_made': self.detection_count,
            'elapsed_time': elapsed_time,
            'fps': fps,
            'is_running': self.is_running,
            'queue_size': self.frame_queue.qsize(),
            'resolution': f"{self.width}x{self.height}" if hasattr(self, 'width') else 'Unknown'
        }
    
    def set_detection_targets(self, targets):
        """Update detection targets"""
        self.detection_targets = targets
        self.logger.info(f"Detection targets updated: {targets}")
    
    def set_confidence_threshold(self, threshold):
        """Update confidence threshold"""
        self.confidence_threshold = threshold
        self.logger.info(f"Confidence threshold updated: {threshold}")
    
    def set_privacy_settings(self, blur_faces=None, blur_plates=None, blur_intensity=None):
        """Update privacy settings"""
        if blur_faces is not None:
            self.blur_faces = blur_faces
        if blur_plates is not None:
            self.blur_plates = blur_plates
        if blur_intensity is not None:
            self.blur_intensity = blur_intensity
        
        self.logger.info(f"Privacy settings updated: faces={self.blur_faces}, plates={self.blur_plates}, intensity={self.blur_intensity}")
