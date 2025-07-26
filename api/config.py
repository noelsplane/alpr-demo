"""
Production Configuration Management for ALPR Surveillance System
"""
import os
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class CameraConfig:
    """Camera configuration for production deployment."""
    camera_id: str
    name: str
    device_path: str  # /dev/video0 or rtsp://ip/stream
    resolution: tuple = (1920, 1080)
    fps: int = 15
    enabled: bool = True
    location: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    direction: Optional[str] = None
    coverage_radius_meters: float = 50.0
    recording_enabled: bool = True
    detection_enabled: bool = True
    motion_detection: bool = True
    night_vision: bool = False
    auto_exposure: bool = True
    brightness: int = 50
    contrast: int = 50
    saturation: int = 50

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CameraConfig':
        return cls(**data)

@dataclass
class SystemConfig:
    """System-wide configuration."""
    max_cameras: int = 8
    default_fps: int = 15
    default_resolution: tuple = (1920, 1080)
    video_storage_path: str = "/var/lib/alpr-surveillance/videos"
    image_storage_path: str = "/var/lib/alpr-surveillance/images"
    max_storage_days: int = 30
    database_url: str = ""
    log_level: str = "INFO"
    api_workers: int = 4
    enable_gpu: bool = False
    gpu_device_id: int = 0
    
    # Detection settings
    plate_confidence_threshold: float = 0.7
    vehicle_confidence_threshold: float = 0.5
    detection_interval_seconds: int = 2
    
    # Alert settings
    enable_anomaly_detection: bool = True
    enable_alerts: bool = True
    alert_cooldown_seconds: int = 300
    
    # Performance settings
    max_frame_buffer_size: int = 5
    processing_thread_count: int = 2
    enable_frame_dropping: bool = True

class ProductionConfig:
    """Production configuration manager."""
    
    def __init__(self, config_dir: str = "/opt/alpr-surveillance/config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.cameras_file = self.config_dir / "cameras.json"
        self.system_file = self.config_dir / "system.json"
        
        self._cameras: Dict[str, CameraConfig] = {}
        self._system_config: SystemConfig = SystemConfig()
        
        self.load_configuration()
    
    def load_configuration(self):
        """Load configuration from files."""
        try:
            # Load system configuration
            if self.system_file.exists():
                with open(self.system_file, 'r') as f:
                    system_data = json.load(f)
                    self._system_config = SystemConfig(**system_data)
                    logger.info("Loaded system configuration")
            else:
                self.save_system_config()
                logger.info("Created default system configuration")
            
            # Load camera configurations
            if self.cameras_file.exists():
                with open(self.cameras_file, 'r') as f:
                    cameras_data = json.load(f)
                    self._cameras = {
                        cam_id: CameraConfig.from_dict(cam_data)
                        for cam_id, cam_data in cameras_data.items()
                    }
                    logger.info(f"Loaded {len(self._cameras)} camera configurations")
            else:
                self.detect_and_save_cameras()
                logger.info("Auto-detected and saved camera configurations")
                
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            self._create_default_config()
    
    def _create_default_config(self):
        """Create default configuration."""
        logger.info("Creating default configuration")
        
        # Default system config
        self._system_config = SystemConfig(
            database_url=os.getenv("DATABASE_URL", "sqlite:///alpr.db"),
            video_storage_path=os.getenv("VIDEO_STORAGE_PATH", "/var/lib/alpr-surveillance/videos"),
            image_storage_path=os.getenv("IMAGE_STORAGE_PATH", "/var/lib/alpr-surveillance/images"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            max_cameras=int(os.getenv("MAX_CAMERAS", "8")),
            api_workers=int(os.getenv("WORKERS", "4"))
        )
        
        # Detect cameras
        self.detect_and_save_cameras()
        
        # Save configuration
        self.save_configuration()
    
    def detect_and_save_cameras(self):
        """Auto-detect available cameras and create configurations."""
        try:
            from camera_detector import camera_detector
            
            detected_cameras = camera_detector.detect_all_cameras()
            self._cameras = {}
            
            for i, camera_info in enumerate(detected_cameras):
                if camera_info.is_working:
                    camera_config = CameraConfig(
                        camera_id=f"camera_{camera_info.index}",
                        name=camera_info.name or f"Camera {camera_info.index}",
                        device_path=camera_info.device_path or str(camera_info.index),
                        resolution=camera_info.resolution or (1920, 1080),
                        location=f"Location {i+1}",
                        enabled=True
                    )
                    self._cameras[camera_config.camera_id] = camera_config
            
            if not self._cameras:
                # Create a default camera config if none detected
                default_camera = CameraConfig(
                    camera_id="camera_0",
                    name="Default Camera",
                    device_path="0",
                    location="Default Location"
                )
                self._cameras["camera_0"] = default_camera
            
            logger.info(f"Detected {len(self._cameras)} cameras")
            
        except Exception as e:
            logger.error(f"Error detecting cameras: {e}")
            # Fallback to default camera
            default_camera = CameraConfig(
                camera_id="camera_0",
                name="Default Camera",
                device_path="0",
                location="Default Location"
            )
            self._cameras = {"camera_0": default_camera}
    
    def save_configuration(self):
        """Save configuration to files."""
        try:
            # Save system configuration
            self.save_system_config()
            
            # Save camera configurations
            cameras_data = {
                cam_id: cam_config.to_dict()
                for cam_id, cam_config in self._cameras.items()
            }
            
            with open(self.cameras_file, 'w') as f:
                json.dump(cameras_data, f, indent=2)
            
            logger.info("Configuration saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def save_system_config(self):
        """Save system configuration."""
        with open(self.system_file, 'w') as f:
            json.dump(asdict(self._system_config), f, indent=2)
    
    # Camera management
    def get_cameras(self) -> Dict[str, CameraConfig]:
        """Get all camera configurations."""
        return self._cameras.copy()
    
    def get_enabled_cameras(self) -> Dict[str, CameraConfig]:
        """Get only enabled camera configurations."""
        return {
            cam_id: cam_config 
            for cam_id, cam_config in self._cameras.items() 
            if cam_config.enabled
        }
    
    def get_camera(self, camera_id: str) -> Optional[CameraConfig]:
        """Get specific camera configuration."""
        return self._cameras.get(camera_id)
    
    def add_camera(self, camera_config: CameraConfig) -> bool:
        """Add new camera configuration."""
        try:
            if len(self._cameras) >= self._system_config.max_cameras:
                logger.warning(f"Maximum camera limit ({self._system_config.max_cameras}) reached")
                return False
            
            self._cameras[camera_config.camera_id] = camera_config
            self.save_configuration()
            logger.info(f"Added camera: {camera_config.camera_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding camera: {e}")
            return False
    
    def update_camera(self, camera_id: str, **kwargs) -> bool:
        """Update camera configuration."""
        try:
            if camera_id not in self._cameras:
                logger.warning(f"Camera {camera_id} not found")
                return False
            
            camera_config = self._cameras[camera_id]
            for key, value in kwargs.items():
                if hasattr(camera_config, key):
                    setattr(camera_config, key, value)
            
            self.save_configuration()
            logger.info(f"Updated camera: {camera_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating camera: {e}")
            return False
    
    def remove_camera(self, camera_id: str) -> bool:
        """Remove camera configuration."""
        try:
            if camera_id in self._cameras:
                del self._cameras[camera_id]
                self.save_configuration()
                logger.info(f"Removed camera: {camera_id}")
                return True
            else:
                logger.warning(f"Camera {camera_id} not found")
                return False
                
        except Exception as e:
            logger.error(f"Error removing camera: {e}")
            return False
    
    def enable_camera(self, camera_id: str) -> bool:
        """Enable camera."""
        return self.update_camera(camera_id, enabled=True)
    
    def disable_camera(self, camera_id: str) -> bool:
        """Disable camera."""
        return self.update_camera(camera_id, enabled=False)
    
    # System configuration
    def get_system_config(self) -> SystemConfig:
        """Get system configuration."""
        return self._system_config
    
    def update_system_config(self, **kwargs) -> bool:
        """Update system configuration."""
        try:
            for key, value in kwargs.items():
                if hasattr(self._system_config, key):
                    setattr(self._system_config, key, value)
            
            self.save_system_config()
            logger.info("Updated system configuration")
            return True
            
        except Exception as e:
            logger.error(f"Error updating system configuration: {e}")
            return False
    
    # Validation
    def validate_camera_config(self, camera_config: CameraConfig) -> List[str]:
        """Validate camera configuration and return any errors."""
        errors = []
        
        if not camera_config.camera_id:
            errors.append("Camera ID is required")
        
        if not camera_config.name:
            errors.append("Camera name is required")
        
        if not camera_config.device_path:
            errors.append("Device path is required")
        
        if camera_config.fps <= 0 or camera_config.fps > 60:
            errors.append("FPS must be between 1 and 60")
        
        if camera_config.resolution[0] <= 0 or camera_config.resolution[1] <= 0:
            errors.append("Resolution must be positive")
        
        if camera_config.latitude is not None:
            if not -90 <= camera_config.latitude <= 90:
                errors.append("Latitude must be between -90 and 90")
        
        if camera_config.longitude is not None:
            if not -180 <= camera_config.longitude <= 180:
                errors.append("Longitude must be between -180 and 180")
        
        return errors
    
    def export_config(self, filepath: str) -> bool:
        """Export configuration to file."""
        try:
            config_data = {
                "system": asdict(self._system_config),
                "cameras": {
                    cam_id: cam_config.to_dict()
                    for cam_id, cam_config in self._cameras.items()
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info(f"Configuration exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting configuration: {e}")
            return False
    
    def import_config(self, filepath: str) -> bool:
        """Import configuration from file."""
        try:
            with open(filepath, 'r') as f:
                config_data = json.load(f)
            
            # Import system config
            if "system" in config_data:
                self._system_config = SystemConfig(**config_data["system"])
            
            # Import camera configs
            if "cameras" in config_data:
                self._cameras = {
                    cam_id: CameraConfig.from_dict(cam_data)
                    for cam_id, cam_data in config_data["cameras"].items()
                }
            
            self.save_configuration()
            logger.info(f"Configuration imported from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error importing configuration: {e}")
            return False

# Global configuration instance
config = ProductionConfig()

def get_config() -> ProductionConfig:
    """Get the global configuration instance."""
    return config

def reload_config():
    """Reload configuration from files."""
    global config
    config.load_configuration()
    logger.info("Configuration reloaded")