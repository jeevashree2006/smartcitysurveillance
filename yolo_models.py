import os
import urllib.request
import torch
from ultralytics import YOLO
import logging

class YOLOModelManager:
    def __init__(self):
        """Initialize YOLO model manager"""
        self.models = {}
        self.model_urls = {
            'yolov8n': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt',
            'yolov8s': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt',
            'yolov8m': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt',
            'yolov8l': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt',
            'yolov8x': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt'
        }
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.models_dir = 'models/weights'
        
        # Create models directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def download_model(self, model_name):
        """
        Download YOLO model weights if not already present
        
        Args:
            model_name: Name of the model (e.g., 'yolov8n')
            
        Returns:
            Path to the downloaded model file
        """
        if model_name not in self.model_urls:
            raise ValueError(f"Unknown model: {model_name}. Available models: {list(self.model_urls.keys())}")
        
        model_path = os.path.join(self.models_dir, f"{model_name}.pt")
        
        if os.path.exists(model_path):
            self.logger.info(f"Model {model_name} already exists at {model_path}")
            return model_path
        
        try:
            self.logger.info(f"Downloading {model_name} model...")
            urllib.request.urlretrieve(self.model_urls[model_name], model_path)
            self.logger.info(f"Model {model_name} downloaded successfully to {model_path}")
            return model_path
            
        except Exception as e:
            self.logger.error(f"Failed to download model {model_name}: {e}")
            # Fallback: try to use the model name directly (ultralytics will download automatically)
            return f"{model_name}.pt"
    
    def load_model(self, model_name='yolov8n'):
        """
        Load YOLO model
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Loaded YOLO model
        """
        if model_name in self.models:
            return self.models[model_name]
        
        try:
            # Try to use local model file first
            model_path = self.download_model(model_name)
            
            # Load the model
            model = YOLO(model_path)
            
            # Move to appropriate device
            model.to(self.device)
            
            # Cache the model
            self.models[model_name] = model
            
            self.logger.info(f"Model {model_name} loaded successfully on {self.device}")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {e}")
            # Fallback: try to load with just the model name
            try:
                model = YOLO(f"{model_name}.pt")
                model.to(self.device)
                self.models[model_name] = model
                self.logger.info(f"Model {model_name} loaded with fallback method")
                return model
            except Exception as fallback_error:
                self.logger.error(f"Fallback loading also failed: {fallback_error}")
                raise
    
    def get_model_info(self, model_name='yolov8n'):
        """
        Get information about a specific model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with model information
        """
        try:
            model = self.load_model(model_name)
            
            info = {
                'name': model_name,
                'device': str(model.device),
                'task': getattr(model, 'task', 'detect'),
                'names': getattr(model, 'names', {}),
                'num_classes': len(getattr(model, 'names', {})),
                'model_size': self._get_model_size(model_name),
                'speed': self._get_model_speed(model_name),
                'accuracy': self._get_model_accuracy(model_name)
            }
            
            return info
            
        except Exception as e:
            self.logger.error(f"Failed to get model info for {model_name}: {e}")
            return {}
    
    def _get_model_size(self, model_name):
        """Get approximate model size information"""
        size_map = {
            'yolov8n': {'params': '3.2M', 'size': '6.2MB'},
            'yolov8s': {'params': '11.2M', 'size': '21.5MB'},
            'yolov8m': {'params': '25.9M', 'size': '49.7MB'},
            'yolov8l': {'params': '43.7M', 'size': '83.7MB'},
            'yolov8x': {'params': '68.2M', 'size': '130.5MB'}
        }
        return size_map.get(model_name, {'params': 'Unknown', 'size': 'Unknown'})
    
    def _get_model_speed(self, model_name):
        """Get approximate model speed information (inference time)"""
        speed_map = {
            'yolov8n': {'cpu': '80.4ms', 'gpu': '0.99ms'},
            'yolov8s': {'cpu': '128.4ms', 'gpu': '1.20ms'},
            'yolov8m': {'cpu': '234.7ms', 'gpu': '1.83ms'},
            'yolov8l': {'cpu': '375.2ms', 'gpu': '2.39ms'},
            'yolov8x': {'cpu': '479.1ms', 'gpu': '3.53ms'}
        }
        return speed_map.get(model_name, {'cpu': 'Unknown', 'gpu': 'Unknown'})
    
    def _get_model_accuracy(self, model_name):
        """Get approximate model accuracy information (mAP50-95)"""
        accuracy_map = {
            'yolov8n': 37.3,
            'yolov8s': 44.9,
            'yolov8m': 50.2,
            'yolov8l': 52.9,
            'yolov8x': 53.9
        }
        return accuracy_map.get(model_name, 0.0)
    
    def list_available_models(self):
        """List all available YOLO models"""
        return list(self.model_urls.keys())
    
    def clear_cache(self):
        """Clear the model cache to free memory"""
        self.models.clear()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        self.logger.info("Model cache cleared")
    
    def get_optimal_model(self, priority='speed'):
        """
        Get the optimal model based on priority
        
        Args:
            priority: 'speed', 'accuracy', or 'balanced'
            
        Returns:
            Recommended model name
        """
        recommendations = {
            'speed': 'yolov8n',      # Fastest inference
            'accuracy': 'yolov8x',   # Highest accuracy
            'balanced': 'yolov8s'    # Good balance of speed and accuracy
        }
        
        return recommendations.get(priority, 'yolov8n')
    
    def benchmark_model(self, model_name, test_image_path=None):
        """
        Benchmark a model's performance
        
        Args:
            model_name: Name of the model to benchmark
            test_image_path: Optional path to test image
            
        Returns:
            Benchmark results dictionary
        """
        try:
            model = self.load_model(model_name)
            
            # Create a dummy image if no test image provided
            if test_image_path is None:
                import numpy as np
                test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            else:
                import cv2
                test_image = cv2.imread(test_image_path)
            
            # Warm up the model
            for _ in range(3):
                _ = model(test_image, verbose=False)
            
            # Benchmark inference time
            import time
            times = []
            num_runs = 10
            
            for _ in range(num_runs):
                start_time = time.time()
                results = model(test_image, verbose=False)
                end_time = time.time()
                times.append(end_time - start_time)
            
            avg_time = sum(times) / len(times)
            
            benchmark_results = {
                'model_name': model_name,
                'average_inference_time': f"{avg_time:.4f}s",
                'fps': f"{1/avg_time:.1f}",
                'device': str(model.device),
                'num_runs': num_runs
            }
            
            return benchmark_results
            
        except Exception as e:
            self.logger.error(f"Failed to benchmark model {model_name}: {e}")
            return {}
