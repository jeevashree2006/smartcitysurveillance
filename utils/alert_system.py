import logging
import time
import queue
import threading
from datetime import datetime
from typing import List, Dict, Any
import json

class AlertSystem:
    def __init__(self):
        """Initialize the alert system"""
        self.alert_queue = queue.Queue()
        self.alert_history = []
        self.alert_callbacks = []
        self.is_running = False
        self.alert_thread = None
        
        # Alert configuration
        self.alert_cooldown = 30  # Minimum seconds between similar alerts
        self.max_history_size = 1000
        self.last_alert_times = {}
        
        # Alert levels
        self.alert_levels = {
            'LOW': 1,
            'MEDIUM': 2,
            'HIGH': 3,
            'CRITICAL': 4
        }
        
        # Category to alert level mapping
        self.category_alert_levels = {
            'Graffiti': 'HIGH',
            'Posters': 'MEDIUM',
            'Dustbins': 'LOW',
            'Vehicles': 'LOW',
            'People': 'LOW'
        }
        
        # Logging setup
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.start()
    
    def start(self):
        """Start the alert processing thread"""
        if not self.is_running:
            self.is_running = True
            self.alert_thread = threading.Thread(target=self._process_alerts, daemon=True)
            self.alert_thread.start()
            self.logger.info("Alert system started")
    
    def stop(self):
        """Stop the alert system"""
        self.is_running = False
        if self.alert_thread and self.alert_thread.is_alive():
            self.alert_thread.join(timeout=2.0)
        self.logger.info("Alert system stopped")
    
    def trigger_alert(self, message: str, detections: List[Dict[str, Any]] = None, level: str = None):
        """
        Trigger a new alert
        
        Args:
            message: Alert message
            detections: List of detection objects that triggered the alert
            level: Alert level (LOW, MEDIUM, HIGH, CRITICAL)
        """
        # Determine alert level if not specified
        if level is None:
            level = self._determine_alert_level(detections)
        
        # Create alert object
        alert = {
            'id': self._generate_alert_id(),
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'level': level,
            'detections': detections or [],
            'location': self._extract_location(detections),
            'confidence_scores': self._extract_confidence_scores(detections),
            'categories': self._extract_categories(detections)
        }
        
        # Check cooldown period
        if self._should_suppress_alert(alert):
            self.logger.debug(f"Alert suppressed due to cooldown: {message}")
            return
        
        # Add to queue for processing
        try:
            self.alert_queue.put_nowait(alert)
            self.logger.info(f"Alert triggered: {message} (Level: {level})")
        except queue.Full:
            self.logger.warning("Alert queue is full, dropping alert")
    
    def _process_alerts(self):
        """Process alerts in the background thread"""
        while self.is_running:
            try:
                # Get alert from queue with timeout
                alert = self.alert_queue.get(timeout=1.0)
                
                # Add to history
                self._add_to_history(alert)
                
                # Execute alert callbacks
                self._execute_callbacks(alert)
                
                # Log alert to console
                self._log_alert(alert)
                
                # Update last alert time for cooldown
                self._update_alert_cooldown(alert)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error processing alert: {e}")
    
    def _determine_alert_level(self, detections: List[Dict[str, Any]]) -> str:
        """Determine alert level based on detections"""
        if not detections:
            return 'LOW'
        
        # Get the highest alert level from all detections
        max_level = 'LOW'
        max_level_value = 0
        
        for detection in detections:
            category = detection.get('category', '')
            level = self.category_alert_levels.get(category, 'LOW')
            level_value = self.alert_levels.get(level, 1)
            
            if level_value > max_level_value:
                max_level = level
                max_level_value = level_value
        
        # Upgrade level based on number of detections
        if len(detections) >= 5:
            if max_level_value < 3:
                max_level = 'HIGH'
        elif len(detections) >= 3:
            if max_level_value < 2:
                max_level = 'MEDIUM'
        
        return max_level
    
    def _should_suppress_alert(self, alert: Dict[str, Any]) -> bool:
        """Check if alert should be suppressed due to cooldown"""
        # Create a key based on alert characteristics
        alert_key = f"{alert['level']}_{'-'.join(alert['categories'])}"
        
        current_time = time.time()
        last_time = self.last_alert_times.get(alert_key, 0)
        
        return (current_time - last_time) < self.alert_cooldown
    
    def _update_alert_cooldown(self, alert: Dict[str, Any]):
        """Update the last alert time for cooldown calculation"""
        alert_key = f"{alert['level']}_{'-'.join(alert['categories'])}"
        self.last_alert_times[alert_key] = time.time()
    
    def _add_to_history(self, alert: Dict[str, Any]):
        """Add alert to history with size limit"""
        self.alert_history.append(alert)
        
        # Trim history if it exceeds maximum size
        if len(self.alert_history) > self.max_history_size:
            self.alert_history = self.alert_history[-self.max_history_size:]
    
    def _execute_callbacks(self, alert: Dict[str, Any]):
        """Execute all registered alert callbacks"""
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Error executing alert callback: {e}")
    
    def _log_alert(self, alert: Dict[str, Any]):
        """Log alert to console with appropriate level"""
        level = alert['level']
        message = alert['message']
        timestamp = alert['timestamp']
        
        log_message = f"[{level}] {timestamp}: {message}"
        
        if level == 'CRITICAL':
            self.logger.critical(log_message)
        elif level == 'HIGH':
            self.logger.error(log_message)
        elif level == 'MEDIUM':
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
        
        # Print to console for immediate visibility
        print(f"🚨 ALERT: {log_message}")
    
    def _generate_alert_id(self) -> str:
        """Generate a unique alert ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"ALERT_{timestamp}_{len(self.alert_history)}"
    
    def _extract_location(self, detections: List[Dict[str, Any]]) -> str:
        """Extract location from detections"""
        if not detections:
            return "Unknown"
        
        locations = [d.get('location', 'Unknown') for d in detections]
        return locations[0] if locations else "Unknown"
    
    def _extract_confidence_scores(self, detections: List[Dict[str, Any]]) -> List[float]:
        """Extract confidence scores from detections"""
        if not detections:
            return []
        
        return [d.get('confidence', 0.0) for d in detections]
    
    def _extract_categories(self, detections: List[Dict[str, Any]]) -> List[str]:
        """Extract unique categories from detections"""
        if not detections:
            return []
        
        categories = list(set(d.get('category', 'Unknown') for d in detections))
        return categories
    
    def register_callback(self, callback):
        """
        Register a callback function to be called when alerts are triggered
        
        Args:
            callback: Function that takes an alert dictionary as parameter
        """
        self.alert_callbacks.append(callback)
        self.logger.info(f"Alert callback registered: {callback.__name__}")
    
    def unregister_callback(self, callback):
        """Unregister an alert callback"""
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)
            self.logger.info(f"Alert callback unregistered: {callback.__name__}")
    
    def get_recent_alerts(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get the most recent alerts"""
        return self.alert_history[-count:] if self.alert_history else []
    
    def get_alerts_by_level(self, level: str) -> List[Dict[str, Any]]:
        """Get all alerts of a specific level"""
        return [alert for alert in self.alert_history if alert['level'] == level]
    
    def get_alerts_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get all alerts containing a specific category"""
        return [alert for alert in self.alert_history 
                if category in alert.get('categories', [])]
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get statistics about alerts"""
        if not self.alert_history:
            return {
                'total_alerts': 0,
                'by_level': {},
                'by_category': {},
                'recent_count': 0
            }
        
        # Count by level
        level_counts = {}
        for alert in self.alert_history:
            level = alert['level']
            level_counts[level] = level_counts.get(level, 0) + 1
        
        # Count by category
        category_counts = {}
        for alert in self.alert_history:
            for category in alert.get('categories', []):
                category_counts[category] = category_counts.get(category, 0) + 1
        
        # Recent alerts (last hour)
        one_hour_ago = datetime.now().timestamp() - 3600
        recent_alerts = [
            alert for alert in self.alert_history
            if datetime.fromisoformat(alert['timestamp']).timestamp() > one_hour_ago
        ]
        
        return {
            'total_alerts': len(self.alert_history),
            'by_level': level_counts,
            'by_category': category_counts,
            'recent_count': len(recent_alerts),
            'alert_rate_per_hour': len(recent_alerts)
        }
    
    def clear_history(self):
        """Clear alert history"""
        self.alert_history.clear()
        self.last_alert_times.clear()
        self.logger.info("Alert history cleared")
    
    def export_alerts(self, format_type: str = 'json') -> str:
        """
        Export alert history in specified format
        
        Args:
            format_type: 'json' or 'csv'
            
        Returns:
            Formatted alert data as string
        """
        if format_type.lower() == 'json':
            return json.dumps(self.alert_history, indent=2, default=str)
        elif format_type.lower() == 'csv':
            import csv
            import io
            
            output = io.StringIO()
            if self.alert_history:
                fieldnames = self.alert_history[0].keys()
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                
                for alert in self.alert_history:
                    # Flatten complex fields for CSV
                    row = alert.copy()
                    row['detections'] = len(alert.get('detections', []))
                    row['categories'] = ', '.join(alert.get('categories', []))
                    row['confidence_scores'] = ', '.join(map(str, alert.get('confidence_scores', [])))
                    writer.writerow(row)
            
            return output.getvalue()
        else:
            raise ValueError("Unsupported format type. Use 'json' or 'csv'")
    
    def set_alert_level_mapping(self, category: str, level: str):
        """Set custom alert level for a category"""
        if level in self.alert_levels:
            self.category_alert_levels[category] = level
            self.logger.info(f"Alert level for {category} set to {level}")
        else:
            raise ValueError(f"Invalid alert level: {level}. Use one of {list(self.alert_levels.keys())}")
    
    def set_cooldown_period(self, seconds: int):
        """Set the cooldown period between similar alerts"""
        self.alert_cooldown = seconds
        self.logger.info(f"Alert cooldown period set to {seconds} seconds")
