# Smart City Surveillance System

A comprehensive real-time surveillance system built with Streamlit and OpenCV for smart city monitoring.

## Features

- **Real-time Video Processing**: Support for webcam, IP cameras, and video files
- **Object Detection**: Detects graffiti, posters, dustbins, vehicles, and people
- **Privacy Protection**: Automatic face and license plate blurring
- **Real-time Alerts**: Console alerts for suspicious activity detection
- **Analytics Dashboard**: Interactive charts and metrics visualization
- **Report Generation**: CSV export and analytics reports
- **Multi-threaded Architecture**: Efficient video processing and alert system

## Quick Start

1. **Run the application**:
   ```bash
   streamlit run app.py --server.port 5000
   ```

2. **Access the web interface**:
   - Open your browser to `http://localhost:5000`
   - The application will load with the surveillance dashboard

3. **Configure your surveillance**:
   - Select video source (Webcam recommended for testing)
   - Choose detection targets (Graffiti, Posters, etc.)
   - Configure privacy settings
   - Set alert thresholds

4. **Start monitoring**:
   - Click "Start Surveillance" to begin real-time processing
   - View live video feed with detection overlays
   - Monitor alerts and analytics in real-time

## System Architecture

### Core Components

- `app.py` - Main Streamlit application and UI
- `detection_engine.py` - Object detection with YOLOv8 and fallback methods
- `privacy_filter.py` - Face and license plate blurring
- `dashboard.py` - Real-time analytics and visualization
- `report_generator.py` - Data export and reporting
- `utils/video_processor.py` - Multi-threaded video processing
- `utils/alert_system.py` - Real-time alert management
- `models/yolo_models.py` - YOLO model management

### Data Flow

1. Video input (camera/file) → Frame capture
2. Object detection → Privacy filtering → Alert checking
3. Real-time display → Analytics update → Report generation

## Configuration Options

### Video Sources
- **Webcam**: Default camera (index 0)
- **IP Camera**: RTMP/HTTP streams
- **Video File**: MP4, AVI, MOV, MKV formats

### Detection Targets
- Graffiti (custom detection)
- Posters (custom detection)
- Dustbins (shape-based detection)
- Vehicles (contour analysis)
- People (face cascade detection)

### Privacy Settings
- Face blurring (with intensity control)
- License plate detection and blurring
- Configurable blur levels

### Alert System
- Real-time console alerts
- Configurable detection thresholds
- Alert cooldown periods
- Multiple alert levels (LOW, MEDIUM, HIGH, CRITICAL)

## Technical Requirements

### Dependencies
- Python 3.11+
- Streamlit
- OpenCV
- Plotly
- Pandas
- NumPy

### Optional Dependencies (Enhanced Features)
- ultralytics (YOLOv8 models)
- torch (GPU acceleration)
- face_recognition (Advanced face detection)

*Note: The system automatically falls back to basic computer vision methods when advanced packages are unavailable.*

## Usage Examples

### Basic Surveillance
```python
# Start with webcam
1. Select "Webcam" as video source
2. Choose detection targets
3. Click "Start Surveillance"
```

### IP Camera Integration
```python
# Configure IP camera
1. Select "IP Camera"
2. Enter RTMP URL: rtmp://192.168.1.100:1935/live/stream
3. Start surveillance
```

### Report Generation
```python
# Generate analytics reports
1. Run surveillance to collect data
2. Click "Generate CSV Report"
3. Download detection logs and analytics
```

## Deployment Options

### Local Development
```bash
streamlit run app.py --server.port 5000
```

### Production Deployment
- Configure server settings in `.streamlit/config.toml`
- Set up proper camera access permissions
- Configure alert notification systems
- Set up data backup for detection logs

## File Structure

```
├── app.py                    # Main application
├── detection_engine.py       # Object detection
├── privacy_filter.py         # Privacy protection
├── dashboard.py              # Analytics dashboard
├── report_generator.py       # Report generation
├── models/
│   └── yolo_models.py        # YOLO model management
├── utils/
│   ├── video_processor.py    # Video processing
│   └── alert_system.py       # Alert management
├── data/
│   └── detection_logs.csv    # Detection logs
└── .streamlit/
    └── config.toml           # Streamlit configuration
```

## Customization

### Adding New Detection Types
1. Modify `detection_engine.py`
2. Add detection logic in `_detect_custom_objects()`
3. Update category mappings
4. Add to UI selection options

### Custom Alert Actions
1. Extend `alert_system.py`
2. Register custom callback functions
3. Implement notification integrations (email, SMS, etc.)

### Enhanced Privacy Features
1. Modify `privacy_filter.py`
2. Add new detection methods
3. Implement selective blurring options

## Troubleshooting

### Common Issues
- **Camera access denied**: Check permissions and camera availability
- **High CPU usage**: Reduce video resolution or frame rate
- **Missing detections**: Adjust confidence thresholds
- **Alert spam**: Increase alert cooldown periods

### Performance Optimization
- Use smaller YOLO models for faster processing
- Reduce video resolution for real-time performance
- Implement frame skipping for high FPS sources
- Configure appropriate buffer sizes

## Contributing

This surveillance system is designed to be modular and extensible. Key areas for enhancement:
- Additional object detection models
- Enhanced privacy protection methods
- Advanced analytics and reporting
- Integration with external security systems
- Mobile application interface

## License

This project is built for educational and demonstration purposes. Ensure compliance with local privacy laws when deploying surveillance systems.