import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import os
import threading
import queue
import time

from detection_engine import DetectionEngine
from privacy_filter import PrivacyFilter
from report_generator import ReportGenerator
from dashboard import Dashboard
from utils.video_processor import VideoProcessor
from utils.alert_system import AlertSystem

# Initialize session state
if 'detection_logs' not in st.session_state:
    st.session_state.detection_logs = []
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False
if 'video_processor' not in st.session_state:
    st.session_state.video_processor = None
if 'alert_queue' not in st.session_state:
    st.session_state.alert_queue = queue.Queue()

def main():
    st.set_page_config(
        page_title="Smart City Surveillance System",
        page_icon="🏙️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🏙️ Smart City Surveillance System")
    st.markdown("### Real-time Object Detection with Privacy Protection")
    
    # Initialize components
    detection_engine = DetectionEngine()
    privacy_filter = PrivacyFilter()
    report_generator = ReportGenerator()
    dashboard = Dashboard()
    alert_system = AlertSystem()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ System Configuration")
        
        # Video source selection
        video_source = st.selectbox(
            "Video Source",
            ["Webcam", "IP Camera", "Video File"],
            help="Select the source for video input"
        )
        
        if video_source == "IP Camera":
            ip_url = st.text_input(
                "IP Camera URL",
                placeholder="rtmp://192.168.1.100:1935/live/stream",
                help="Enter the RTMP/HTTP URL of your IP camera"
            )
        elif video_source == "Video File":
            uploaded_file = st.file_uploader(
                "Upload Video File",
                type=['mp4', 'avi', 'mov', 'mkv'],
                help="Upload a video file for processing"
            )
        
        # Detection settings
        st.subheader("🎯 Detection Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Minimum confidence score for detections"
        )
        
        detection_targets = st.multiselect(
            "Detection Targets",
            ["Graffiti", "Posters", "Dustbins", "Vehicles", "People"],
            default=["Graffiti", "Posters", "Dustbins"],
            help="Select objects to detect"
        )
        
        # Privacy settings
        st.subheader("🔒 Privacy Settings")
        blur_faces = st.checkbox("Blur Faces", value=True)
        blur_plates = st.checkbox("Blur License Plates", value=True)
        blur_intensity = st.slider("Blur Intensity", 1, 20, 10)
        
        # Alert settings
        st.subheader("🚨 Alert Settings")
        enable_alerts = st.checkbox("Enable Real-time Alerts", value=True)
        alert_threshold = st.slider("Alert Threshold", 1, 10, 3, 
                                  help="Number of detections before triggering alert")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Live Video Feed")
        video_placeholder = st.empty()
        
        # Control buttons
        col_start, col_stop, col_capture = st.columns(3)
        
        with col_start:
            if st.button("🎬 Start Surveillance", type="primary"):
                if not st.session_state.is_recording:
                    st.session_state.is_recording = True
                    
                    # Initialize video source
                    if video_source == "Webcam":
                        source = 0
                    elif video_source == "IP Camera" and ip_url:
                        source = ip_url
                    elif video_source == "Video File" and uploaded_file:
                        # Save uploaded file temporarily
                        with open("temp_video.mp4", "wb") as f:
                            f.write(uploaded_file.read())
                        source = "temp_video.mp4"
                    else:
                        st.error("Please configure video source properly")
                        st.session_state.is_recording = False
                        source = None
                    
                    if source is not None:
                        st.session_state.video_processor = VideoProcessor(
                            source=source,
                            detection_engine=detection_engine,
                            privacy_filter=privacy_filter,
                            alert_system=alert_system,
                            confidence_threshold=confidence_threshold,
                            detection_targets=detection_targets,
                            blur_faces=blur_faces,
                            blur_plates=blur_plates,
                            blur_intensity=blur_intensity,
                            enable_alerts=enable_alerts,
                            alert_threshold=alert_threshold
                        )
                        st.success("Surveillance started!")
                    
        with col_stop:
            if st.button("⏹️ Stop Surveillance"):
                if st.session_state.is_recording:
                    st.session_state.is_recording = False
                    if st.session_state.video_processor:
                        st.session_state.video_processor.stop()
                    st.success("Surveillance stopped!")
        
        with col_capture:
            if st.button("📸 Capture Frame"):
                if st.session_state.video_processor:
                    frame = st.session_state.video_processor.get_current_frame()
                    if frame is not None:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        cv2.imwrite(f"capture_{timestamp}.jpg", frame)
                        st.success(f"Frame captured as capture_{timestamp}.jpg")
    
    with col2:
        st.subheader("🚨 Real-time Alerts")
        alert_container = st.container()
        
        # Display recent alerts
        with alert_container:
            if st.session_state.alert_queue.qsize() > 0:
                alerts = []
                while not st.session_state.alert_queue.empty():
                    try:
                        alert = st.session_state.alert_queue.get_nowait()
                        alerts.append(alert)
                    except queue.Empty:
                        break
                
                for alert in alerts[-5:]:  # Show last 5 alerts
                    st.warning(f"⚠️ {alert['message']}")
                    st.caption(f"Time: {alert['timestamp']}")
            else:
                st.info("No alerts detected")
    
    # Video processing and display
    if st.session_state.is_recording and st.session_state.video_processor:
        frame = st.session_state.video_processor.process_frame()
        if frame is not None:
            # Convert BGR to RGB for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
        
        # Update detection logs
        new_detections = st.session_state.video_processor.get_new_detections()
        if new_detections:
            st.session_state.detection_logs.extend(new_detections)
    
    # Dashboard and Analytics
    st.markdown("---")
    dashboard.render(st.session_state.detection_logs)
    
    # Report Generation
    st.markdown("---")
    st.subheader("📊 Report Generation")
    
    col_export1, col_export2, col_export3 = st.columns(3)
    
    with col_export1:
        if st.button("📄 Generate CSV Report"):
            if st.session_state.detection_logs:
                csv_data = report_generator.generate_csv_report(st.session_state.detection_logs)
                st.download_button(
                    label="Download CSV Report",
                    data=csv_data,
                    file_name=f"surveillance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No detection data available for report generation")
    
    with col_export2:
        if st.button("📈 Generate Analytics Report"):
            if st.session_state.detection_logs:
                analytics_data = report_generator.generate_analytics_report(st.session_state.detection_logs)
                st.download_button(
                    label="Download Analytics Report",
                    data=analytics_data,
                    file_name=f"analytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            else:
                st.warning("No detection data available for analytics")
    
    with col_export3:
        if st.button("🗂️ Clear Detection Logs"):
            st.session_state.detection_logs = []
            st.success("Detection logs cleared!")

if __name__ == "__main__":
    main()
