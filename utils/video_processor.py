import cv2
import time
import logging
from datetime import datetime


class VideoProcessor:
    """
    Reads frames from a video source and runs the full surveillance pipeline
    on each one:  privacy blurring  ->  object detection  ->  annotation  ->  alerts.

    Why synchronous (no background threads)?
    ----------------------------------------
    The original version launched two background threads from a ``start()``
    method -- but nothing ever called ``start()``, so the live feed stayed
    blank.  Even if it had, Streamlit re-runs the whole script top-to-bottom
    on every interaction, which fights a long-lived background thread.

    The reliable pattern for Streamlit + OpenCV is to process ONE frame per
    call and let the app drive a display loop.  That is exactly what
    ``read_and_process()`` does.
    """

    def __init__(self, source, detection_engine, privacy_filter, alert_system,
                 confidence_threshold=0.5, detection_targets=None,
                 blur_faces=True, blur_plates=True, blur_intensity=10,
                 enable_alerts=True, alert_threshold=3):
        """
        Args:
            source: Video source (camera index, file path, or stream URL)
            detection_engine: Object detection engine
            privacy_filter: Privacy filtering component
            alert_system: Alert system for notifications
            confidence_threshold: Minimum confidence for detections
            detection_targets: List of objects to detect
            blur_faces: Whether to blur faces
            blur_plates: Whether to blur license plates
            blur_intensity: Blur effect intensity
            enable_alerts: Whether to enable real-time alerts
            alert_threshold: Number of suspicious detections before an alert
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

        # Current frame and detections
        self.current_frame = None
        self.current_detections = []

        # Statistics
        self.frame_count = 0
        self.detection_count = 0
        self.start_time = None

        # Alert tracking (cooldown so we don't spam)
        self.consecutive_detections = 0
        self.last_alert_time = None

        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Open the video source immediately (raises if it can't be opened)
        self._initialize_capture()

    def _initialize_capture(self):
        """Open the video source and read its properties."""
        self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source: {self.source}")

        # Keep latency low for live sources
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.is_running = True
        self.start_time = time.time()

        self.logger.info(
            f"Video capture initialized: {self.width}x{self.height} @ {self.fps}fps"
        )

    def read_and_process(self):
        """
        Read ONE frame from the source and run the full pipeline on it.

        Returns:
            (annotated_frame_bgr, detections) on success, or
            (None, []) when the source has no more frames (end of file) or fails.
        """
        if self.cap is None or not self.cap.isOpened():
            return None, []

        ret, frame = self.cap.read()
        if not ret:
            # End of a video file, or a camera read failure.
            return None, []

        # Downscale large frames for smoother performance
        if frame.shape[1] > 1280:
            scale = 1280 / frame.shape[1]
            frame = cv2.resize(
                frame,
                (int(frame.shape[1] * scale), int(frame.shape[0] * scale)),
            )

        # 1) Privacy first -- blur faces/plates on a copy so identities never leak
        filtered_frame = self.privacy_filter.apply_privacy_filters(
            frame,
            blur_faces=self.blur_faces,
            blur_plates=self.blur_plates,
            blur_intensity=self.blur_intensity,
        )

        # 2) Detect on the ORIGINAL (unblurred) frame for better accuracy
        detections = self.detection_engine.detect_objects(
            frame,
            confidence_threshold=self.confidence_threshold,
            detection_targets=self.detection_targets,
        )

        # 3) Draw boxes on the privacy-filtered frame + check for alerts
        if detections:
            filtered_frame = self.detection_engine.draw_detections(filtered_frame, detections)
            self.detection_count += len(detections)

            if self.enable_alerts:
                self._check_alerts(detections)

        self.frame_count += 1

        # 4) Overlay timestamp / FPS / running detection count
        self._draw_overlay(filtered_frame)

        self.current_frame = filtered_frame
        self.current_detections = detections
        return filtered_frame, detections

    def _draw_overlay(self, frame):
        """Draw the live status text in the top-left corner of the frame."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"Time: {timestamp}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        elapsed = time.time() - self.start_time if self.start_time else 0
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Detections: {self.detection_count}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def _check_alerts(self, detections):
        """Trigger an alert when enough suspicious objects pile up."""
        alert_categories = ['Graffiti', 'Posters']
        alert_detections = [d for d in detections if d.get('category') in alert_categories]

        if alert_detections:
            self.consecutive_detections += len(alert_detections)

            if self.consecutive_detections >= self.alert_threshold:
                current_time = time.time()

                # Minimum 30 seconds between alerts to avoid spam
                if (self.last_alert_time is None or
                        current_time - self.last_alert_time > 30):
                    alert_message = (
                        f"Alert: {len(alert_detections)} suspicious objects detected"
                    )
                    self.alert_system.trigger_alert(alert_message, alert_detections)

                    self.last_alert_time = current_time
                    self.consecutive_detections = 0  # Reset counter
        else:
            # Decay the counter when nothing suspicious is seen
            self.consecutive_detections = max(0, self.consecutive_detections - 1)

    def get_current_frame(self):
        """Return a copy of the most recently processed frame (BGR), or None."""
        if self.current_frame is not None:
            return self.current_frame.copy()
        return None

    def get_current_detections(self):
        """Return the detections from the most recent frame."""
        return list(self.current_detections)

    def get_statistics(self):
        """Return processing statistics for display."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        fps = self.frame_count / elapsed if elapsed > 0 else 0

        return {
            'frames_processed': self.frame_count,
            'detections_made': self.detection_count,
            'elapsed_time': elapsed,
            'fps': fps,
            'is_running': self.is_running,
            'resolution': f"{self.width}x{self.height}" if hasattr(self, 'width') else 'Unknown',
        }

    def stop(self):
        """Stop processing and release the video source."""
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.logger.info("Video processing stopped")
