import streamlit as st
import cv2
from datetime import datetime

from detection_engine import DetectionEngine
from privacy_filter import PrivacyFilter
from report_generator import ReportGenerator
from dashboard import Dashboard
from utils.video_processor import VideoProcessor
from utils.alert_system import AlertSystem


# ---------------------------------------------------------------------------
# Component initialisation
# ---------------------------------------------------------------------------
# @st.cache_resource makes Streamlit create these ONCE and reuse them on every
# rerun.  Without it, the YOLO model (and the alert background thread) would be
# rebuilt on every single button click -- slow and wasteful.
@st.cache_resource
def load_components():
    """Create the heavy, long-lived components a single time."""
    return {
        "detection_engine": DetectionEngine(),
        "privacy_filter": PrivacyFilter(),
        "report_generator": ReportGenerator(),
        "dashboard": Dashboard(),
        "alert_system": AlertSystem(),
    }


def init_session_state():
    """Seed the session-state keys the app relies on."""
    defaults = {
        "detection_logs": [],       # every detection collected this session
        "is_recording": False,      # is the live loop running?
        "video_processor": None,    # the active VideoProcessor (holds the camera)
        "last_frame": None,         # latest BGR frame, used by "Capture Frame"
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_alert_panel(placeholder, alert_system):
    """(Re)draw the recent-alerts panel inside a single placeholder."""
    with placeholder.container():
        recent = alert_system.get_recent_alerts(5)
        if recent:
            for alert in reversed(recent):  # newest first
                st.warning(f"⚠️ {alert['message']}")
                st.caption(f"{alert['level']} • {alert['timestamp']}")
        else:
            st.info("No alerts detected")


def main():
    st.set_page_config(
        page_title="Smart City Surveillance System",
        page_icon="🏙️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()

    components = load_components()
    detection_engine = components["detection_engine"]
    privacy_filter = components["privacy_filter"]
    report_generator = components["report_generator"]
    dashboard = components["dashboard"]
    alert_system = components["alert_system"]

    st.title("🏙️ Smart City Surveillance System")
    st.markdown("### Real-time Object Detection with Privacy Protection")

    # ----------------------------- Sidebar --------------------------------
    with st.sidebar:
        st.header("⚙️ System Configuration")

        video_source = st.selectbox(
            "Video Source",
            ["Webcam", "IP Camera", "Video File"],
            help="Select the source for video input",
        )

        ip_url = ""
        uploaded_file = None
        if video_source == "IP Camera":
            ip_url = st.text_input(
                "IP Camera URL",
                placeholder="rtmp://192.168.1.100:1935/live/stream",
                help="Enter the RTMP/HTTP URL of your IP camera",
            )
        elif video_source == "Video File":
            uploaded_file = st.file_uploader(
                "Upload Video File",
                type=["mp4", "avi", "mov", "mkv"],
                help="Upload a video file for processing",
            )

        st.subheader("🎯 Detection Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold", 0.1, 1.0, 0.5, 0.05,
            help="Minimum confidence score for detections",
        )
        detection_targets = st.multiselect(
            "Detection Targets",
            ["Graffiti", "Posters", "Dustbins", "Vehicles", "People"],
            default=["Graffiti", "Posters", "Dustbins"],
            help="Select objects to detect",
        )

        st.subheader("🔒 Privacy Settings")
        blur_faces = st.checkbox("Blur Faces", value=True)
        blur_plates = st.checkbox("Blur License Plates", value=True)
        blur_intensity = st.slider("Blur Intensity", 1, 20, 10)

        st.subheader("🚨 Alert Settings")
        enable_alerts = st.checkbox("Enable Real-time Alerts", value=True)
        alert_threshold = st.slider(
            "Alert Threshold", 1, 10, 3,
            help="Number of suspicious detections before triggering an alert",
        )

    # -------------------------- Main layout -------------------------------
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📹 Live Video Feed")
        video_placeholder = st.empty()
        stats_placeholder = st.empty()

        col_start, col_stop, col_capture = st.columns(3)
        start_clicked = col_start.button("🎬 Start Surveillance", type="primary")
        stop_clicked = col_stop.button("⏹️ Stop Surveillance")
        capture_clicked = col_capture.button("📸 Capture Frame")

    with col2:
        st.subheader("🚨 Real-time Alerts")
        alert_placeholder = st.empty()

    # ----------------------------- Start ----------------------------------
    if start_clicked and not st.session_state.is_recording:
        source = None

        if video_source == "Webcam":
            source = 0
        elif video_source == "IP Camera":
            if ip_url:
                source = ip_url
            else:
                st.error("Please enter an IP camera URL.")
        elif video_source == "Video File":
            if uploaded_file is not None:
                with open("temp_video.mp4", "wb") as f:
                    f.write(uploaded_file.read())
                source = "temp_video.mp4"
            else:
                st.error("Please upload a video file first.")

        if source is not None:
            try:
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
                    alert_threshold=alert_threshold,
                )
                st.session_state.is_recording = True
            except Exception as exc:
                st.session_state.is_recording = False
                st.session_state.video_processor = None
                st.error(f"Could not start surveillance: {exc}")

    # ------------------------------ Stop ----------------------------------
    if stop_clicked and st.session_state.is_recording:
        st.session_state.is_recording = False
        if st.session_state.video_processor is not None:
            st.session_state.video_processor.stop()
            st.session_state.video_processor = None
        st.success("Surveillance stopped.")

    # ---------------------------- Capture ---------------------------------
    if capture_clicked:
        if st.session_state.last_frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{timestamp}.jpg"
            cv2.imwrite(filename, st.session_state.last_frame)
            st.success(f"Frame captured as {filename}")
        else:
            st.warning("No frame available to capture yet. Start surveillance first.")

    # ----------------------- Live processing loop -------------------------
    # This blocking loop keeps pulling and displaying frames.  Clicking any
    # button (Stop/Capture) makes Streamlit interrupt the loop and rerun the
    # script from the top -- which is how the loop gets stopped cleanly.
    if st.session_state.is_recording and st.session_state.video_processor is not None:
        vp = st.session_state.video_processor

        while st.session_state.is_recording:
            frame, detections = vp.read_and_process()

            if frame is None:
                # Source ended (video file finished) or the camera failed
                st.session_state.is_recording = False
                vp.stop()
                st.session_state.video_processor = None
                video_placeholder.info("📷 Video source ended or is unavailable.")
                break

            # Remember the latest BGR frame for the Capture button
            st.session_state.last_frame = frame

            # Streamlit needs RGB; OpenCV gives us BGR
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            # Collect detections (cap the list so memory can't grow forever)
            if detections:
                st.session_state.detection_logs.extend(detections)
                if len(st.session_state.detection_logs) > 5000:
                    st.session_state.detection_logs = st.session_state.detection_logs[-5000:]

            # Live stats + alert panel
            stats = vp.get_statistics()
            stats_placeholder.caption(
                f"FPS: {stats['fps']:.1f}  •  Frames: {stats['frames_processed']}  •  "
                f"Total detections: {stats['detections_made']}  •  "
                f"Resolution: {stats['resolution']}"
            )
            render_alert_panel(alert_placeholder, alert_system)
    else:
        video_placeholder.info(
            "📷 Video feed will appear here once you click **Start Surveillance**."
        )
        render_alert_panel(alert_placeholder, alert_system)

    # ---------------------------- Dashboard -------------------------------
    # (Shown when idle/stopped. During a live webcam feed the loop above runs
    # continuously, so the dashboard refreshes each time you stop.)
    st.markdown("---")
    dashboard.render(st.session_state.detection_logs)

    # ------------------------- Report generation --------------------------
    st.markdown("---")
    st.subheader("📊 Report Generation")

    col_export1, col_export2, col_export3 = st.columns(3)

    with col_export1:
        if st.session_state.detection_logs:
            csv_data = report_generator.generate_csv_report(st.session_state.detection_logs)
            st.download_button(
                "📄 Download CSV Report",
                data=csv_data,
                file_name=f"surveillance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
        else:
            st.button("📄 Generate CSV Report", disabled=True,
                      help="No detection data available yet")

    with col_export2:
        if st.session_state.detection_logs:
            analytics_data = report_generator.generate_analytics_report(
                st.session_state.detection_logs
            )
            st.download_button(
                "📈 Download Analytics Report",
                data=analytics_data,
                file_name=f"analytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )
        else:
            st.button("📈 Generate Analytics Report", disabled=True,
                      help="No detection data available yet")

    with col_export3:
        if st.button("🗂️ Clear Detection Logs"):
            st.session_state.detection_logs = []
            st.success("Detection logs cleared!")


if __name__ == "__main__":
    main()
