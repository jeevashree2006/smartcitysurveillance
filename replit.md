# Smart City Surveillance System

## Overview

This is a real-time computer vision surveillance system built with Streamlit that provides object detection capabilities for smart city monitoring. The system includes YOLOv8 integration with fallback detection methods, privacy protection features, and comprehensive analytics and reporting capabilities.

**Project Status**: Complete and functional - Ready for deployment
**Last Updated**: July 25, 2025

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

The application follows a modular, component-based architecture with clear separation of concerns:

- **Frontend**: Streamlit-based web interface for real-time monitoring and configuration
- **Computer Vision Engine**: YOLOv8-powered object detection system
- **Privacy Protection**: Face and license plate blurring capabilities
- **Analytics & Reporting**: Real-time dashboard with data visualization and export features
- **Alert System**: Threaded alert processing with configurable thresholds
- **Video Processing**: Multi-threaded video capture and processing pipeline

## Key Components

### Core Application (`app.py`)
- Main Streamlit application entry point
- Session state management for persistent data
- Component initialization and orchestration

### Detection Engine (`detection_engine.py`)
- YOLOv8 model integration for object detection
- Supports both pre-trained COCO models and custom detection logic
- Targets specific smart city objects: graffiti, posters, dustbins, vehicles, people, bicycles
- GPU acceleration support with CUDA fallback to CPU

### Privacy Filter (`privacy_filter.py`)
- Face detection and blurring using face_recognition library
- License plate detection and blurring using OpenCV Haar cascades
- Configurable blur intensity and selective filtering

### Dashboard (`dashboard.py`)
- Real-time analytics visualization using Plotly
- Metrics display and temporal analysis
- Detection logs table with filtering capabilities
- Responsive layout with multiple chart types

### Report Generator (`report_generator.py`)
- CSV export functionality for detection logs
- Data cleaning and formatting for reports
- Statistical analysis of detection patterns

### Alert System (`utils/alert_system.py`)
- Multi-threaded alert processing with queue-based architecture
- Configurable alert levels (LOW, MEDIUM, HIGH, CRITICAL)
- Alert cooldown mechanism to prevent spam
- Callback system for extensible alert handling

### Video Processor (`utils/video_processor.py`)
- Multi-threaded video capture and processing
- Queue-based frame management to prevent blocking
- Integration with detection engine and privacy filters
- Configurable detection parameters and thresholds

### Model Manager (`models/yolo_models.py`)
- Automated YOLO model downloading and management
- Support for multiple YOLOv8 variants (nano to extra-large)
- Device detection and model optimization

## Data Flow

1. **Video Input**: Capture frames from camera, file, or stream
2. **Frame Processing**: Apply object detection using YOLOv8
3. **Privacy Filtering**: Blur faces and license plates based on configuration
4. **Detection Logging**: Store detection results with timestamps and metadata
5. **Alert Processing**: Check detection patterns against alert thresholds
6. **Dashboard Update**: Real-time visualization of detection analytics
7. **Report Generation**: Export data in CSV format for analysis

## External Dependencies

### Computer Vision Libraries
- **OpenCV**: Video processing, image manipulation, and Haar cascades
- **Ultralytics YOLO**: State-of-the-art object detection
- **face_recognition**: Face detection and recognition capabilities
- **PyTorch**: Deep learning framework with GPU support

### Web Framework & UI
- **Streamlit**: Web application framework and UI components
- **Plotly**: Interactive data visualization and charts

### Data Processing
- **pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing and array operations

### Utilities
- **threading**: Multi-threaded processing for video and alerts
- **queue**: Thread-safe data sharing between components

## Deployment Strategy

The application is designed for local deployment with the following considerations:

### Hardware Requirements
- **GPU**: CUDA-compatible GPU recommended for optimal performance
- **CPU**: Fallback support for CPU-only environments
- **Memory**: Sufficient RAM for video processing and model loading
- **Storage**: Space for model weights and detection logs

### Configuration
- Environment-based configuration for model selection
- Runtime device detection (CUDA vs CPU)
- Configurable detection thresholds and privacy settings
- Modular component initialization with error handling

### Scalability
- Thread-based architecture for concurrent processing
- Queue-based frame management to handle varying processing speeds
- Configurable buffer sizes and processing parameters
- Memory-efficient detection logging with size limits

The system is architected to be easily extensible, allowing for additional detection models, new alert types, and enhanced privacy features without significant structural changes.