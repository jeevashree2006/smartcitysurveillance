#!/usr/bin/env python3
"""
Smart City Surveillance System Setup Script
Automatically sets up the project environment and dependencies
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required Python packages"""
    required_packages = [
        'streamlit>=1.47.0',
        'opencv-python>=4.11.0', 
        'plotly>=6.2.0',
        'pandas>=2.0.0',
        'numpy>=1.24.0'
    ]
    
    print("Installing required packages...")
    for package in required_packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✓ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {package}")

def check_directories():
    """Ensure all required directories exist"""
    directories = ['utils', 'models', '.streamlit', 'data']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✓ Created directory: {directory}")

def main():
    print("Smart City Surveillance System Setup")
    print("=" * 40)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher required")
        sys.exit(1)
    
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Setup directories
    check_directories()
    
    # Install packages
    install_requirements()
    
    print("\n" + "=" * 40)
    print("Setup complete! To run the application:")
    print("streamlit run app.py --server.port 5000")
    print("\nThen open http://localhost:5000 in your browser")

if __name__ == "__main__":
    main()