import cv2
import logging
import platform
import subprocess
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import psutil

logger = logging.getLogger(__name__)

@dataclass
class CameraInfo:
    """Information about a detected camera."""
    index: int
    name: str
    device_path: Optional[str] = None
    resolution: Optional[Tuple[int, int]] = None
    fps: Optional[int] = None
    is_working: bool = True
    connection_type: str = "USB"

class CameraDetector:
    """Detects and manages available cameras including USB, built-in, and network cameras."""
    
    def __init__(self):
        self.detected_cameras: List[CameraInfo] = []
        self.platform = platform.system().lower()
        
    def detect_all_cameras(self) -> List[CameraInfo]:
        """Detect all available cameras using multiple methods."""
        self.detected_cameras = []
        
        # Check for WSL2 limitations
        self._check_wsl2_usb_support()
        
        # Method 1: OpenCV enumeration
        opencv_cameras = self._detect_opencv_cameras()
        self.detected_cameras.extend(opencv_cameras)
        
        # Method 2: Platform-specific detection
        if self.platform == "linux":
            linux_cameras = self._detect_linux_cameras()
            self._merge_camera_info(linux_cameras)
        elif self.platform == "windows":
            windows_cameras = self._detect_windows_cameras()
            self._merge_camera_info(windows_cameras)
        elif self.platform == "darwin":
            macos_cameras = self._detect_macos_cameras()
            self._merge_camera_info(macos_cameras)
        
        # Method 3: Network camera detection (common IP ranges)
        network_cameras = self._detect_network_cameras()
        self.detected_cameras.extend(network_cameras)
            
        # Test camera functionality
        self._test_cameras()
        
        logger.info(f"Detected {len(self.detected_cameras)} cameras")
        if len(self.detected_cameras) == 0:
            self._log_troubleshooting_info()
        return self.detected_cameras
    
    def _detect_opencv_cameras(self) -> List[CameraInfo]:
        """Detect cameras using OpenCV enumeration."""
        cameras = []
        
        # Check if we're in WSL2 environment
        try:
            with open('/proc/version', 'r') as f:
                if 'microsoft' in f.read().lower():
                    logger.warning("Running in WSL2 - USB camera access may be limited")
        except:
            pass
        
        # Test camera indices 0-10 (covers most common scenarios)
        for i in range(11):
            try:
                cap = cv2.VideoCapture(i)
                if cap is not None and cap.isOpened():
                    # Get camera properties
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    
                    # Test if we can actually read a frame
                    ret, frame = cap.read()
                    is_working = ret and frame is not None
                    
                    camera = CameraInfo(
                        index=i,
                        name=f"Camera {i}",
                        resolution=(width, height) if width > 0 and height > 0 else None,
                        fps=fps if fps > 0 else None,
                        is_working=is_working
                    )
                    cameras.append(camera)
                    logger.info(f"Found camera at index {i}: {width}x{height}@{fps}fps")
                    
                cap.release()
            except Exception as e:
                logger.debug(f"Error testing camera index {i}: {e}")
                continue
            
        return cameras
    
    def _detect_linux_cameras(self) -> List[CameraInfo]:
        """Detect cameras on Linux using /dev/video* devices."""
        cameras = []
        
        try:
            # List video devices
            result = subprocess.run(['ls', '/dev/video*'], 
                                  capture_output=True, text=True, check=False)
            
            if result.returncode == 0:
                video_devices = result.stdout.strip().split('\n')
                
                for device_path in video_devices:
                    if not device_path:
                        continue
                        
                    # Extract device number
                    match = re.search(r'video(\d+)', device_path)
                    if match:
                        device_num = int(match.group(1))
                        
                        # Get device info using v4l2-ctl if available
                        name = self._get_linux_camera_name(device_path)
                        
                        camera = CameraInfo(
                            index=device_num,
                            name=name or f"Video Device {device_num}",
                            device_path=device_path,
                            connection_type="USB/Built-in"
                        )
                        cameras.append(camera)
                        
        except Exception as e:
            logger.warning(f"Error detecting Linux cameras: {e}")
            
        return cameras
    
    def _get_linux_camera_name(self, device_path: str) -> Optional[str]:
        """Get camera name on Linux using v4l2-ctl."""
        try:
            result = subprocess.run(['v4l2-ctl', '--device', device_path, '--info'], 
                                  capture_output=True, text=True, check=False)
            
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'Card type' in line:
                        return line.split(':', 1)[1].strip()
                        
        except (subprocess.SubprocessError, FileNotFoundError):
            # v4l2-ctl not available
            pass
            
        return None
    
    def _detect_windows_cameras(self) -> List[CameraInfo]:
        """Detect cameras on Windows using PowerShell."""
        cameras = []
        
        try:
            # Use PowerShell to enumerate imaging devices
            ps_command = """
            Get-PnpDevice -Class Image | Where-Object {$_.Status -eq 'OK'} | 
            Select-Object FriendlyName, InstanceId
            """
            
            result = subprocess.run(['powershell', '-Command', ps_command], 
                                  capture_output=True, text=True, check=False)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                device_index = 0
                
                for line in lines[3:]:  # Skip headers
                    if line.strip():
                        parts = line.strip().split(None, 1)
                        if len(parts) >= 1:
                            name = parts[0] if len(parts) == 1 else ' '.join(parts[:-1])
                            
                            camera = CameraInfo(
                                index=device_index,
                                name=name,
                                connection_type="USB/Built-in"
                            )
                            cameras.append(camera)
                            device_index += 1
                            
        except Exception as e:
            logger.warning(f"Error detecting Windows cameras: {e}")
            
        return cameras
    
    def _detect_macos_cameras(self) -> List[CameraInfo]:
        """Detect cameras on macOS using system_profiler."""
        cameras = []
        
        try:
            result = subprocess.run(['system_profiler', 'SPCameraDataType'], 
                                  capture_output=True, text=True, check=False)
            
            if result.returncode == 0:
                camera_sections = result.stdout.split('\n\n')
                device_index = 0
                
                for section in camera_sections:
                    if 'Camera' in section or 'FaceTime' in section:
                        # Extract camera name
                        lines = section.split('\n')
                        name = "Unknown Camera"
                        
                        for line in lines:
                            if ':' not in line and line.strip():
                                name = line.strip()
                                break
                                
                        camera = CameraInfo(
                            index=device_index,
                            name=name,
                            connection_type="Built-in/USB"
                        )
                        cameras.append(camera)
                        device_index += 1
                        
        except Exception as e:
            logger.warning(f"Error detecting macOS cameras: {e}")
            
        return cameras
    
    def _detect_network_cameras(self) -> List[CameraInfo]:
        """Detect network cameras using common IP addresses and RTSP streams."""
        cameras = []
        
        # Common network camera IP addresses and RTSP URLs to try
        common_ips = [
            "192.168.1.100", "192.168.1.101", "192.168.1.102",
            "192.168.0.100", "192.168.0.101", "192.168.0.102",
            "10.0.0.100", "10.0.0.101", "10.0.0.102"
        ]
        
        # Common RTSP paths for different camera brands
        rtsp_paths = [
            "/stream1",
            "/live/ch00_0",
            "/cam/realmonitor?channel=1&subtype=0",
            "/videoMain",
            "/h264_ulaw.sdp",
            "/axis-media/media.amp"
        ]
        
        camera_index = 1000  # Start network cameras at index 1000+
        
        for ip in common_ips:
            for path in rtsp_paths:
                try:
                    rtsp_url = f"rtsp://{ip}{path}"
                    
                    # Quick test to see if camera responds (timeout after 2 seconds)
                    cap = cv2.VideoCapture(rtsp_url)
                    cap.set(cv2.CAP_PROP_TIMEOUT, 2000)  # 2 second timeout
                    
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            
                            camera = CameraInfo(
                                index=camera_index,
                                name=f"Network Camera {ip}",
                                device_path=rtsp_url,
                                resolution=(width, height) if width > 0 and height > 0 else None,
                                is_working=True,
                                connection_type="Network/RTSP"
                            )
                            cameras.append(camera)
                            logger.info(f"Found network camera at {rtsp_url}")
                            camera_index += 1
                            break  # Found working path for this IP, move to next IP
                    
                    cap.release()
                    
                except Exception as e:
                    # Network cameras may not be available, this is expected
                    pass
        
        return cameras
    
    def _merge_camera_info(self, platform_cameras: List[CameraInfo]):
        """Merge platform-specific camera info with OpenCV detected cameras."""
        for platform_cam in platform_cameras:
            # Try to match with existing OpenCV camera
            matched = False
            for opencv_cam in self.detected_cameras:
                if (opencv_cam.index == platform_cam.index or 
                    (platform_cam.device_path and 
                     str(platform_cam.index) in str(opencv_cam.index))):
                    # Update OpenCV camera with platform info
                    opencv_cam.name = platform_cam.name
                    opencv_cam.device_path = platform_cam.device_path
                    opencv_cam.connection_type = platform_cam.connection_type
                    matched = True
                    break
                    
            if not matched:
                # Add new camera that wasn't detected by OpenCV
                self.detected_cameras.append(platform_cam)
    
    def _test_cameras(self):
        """Test each detected camera to verify it's working."""
        for camera in self.detected_cameras:
            if camera.is_working is None or camera.is_working:
                try:
                    cap = cv2.VideoCapture(camera.index)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        camera.is_working = ret and frame is not None
                        
                        if camera.is_working and not camera.resolution:
                            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            if width > 0 and height > 0:
                                camera.resolution = (width, height)
                                
                        if camera.is_working and not camera.fps:
                            fps = int(cap.get(cv2.CAP_PROP_FPS))
                            if fps > 0:
                                camera.fps = fps
                    else:
                        camera.is_working = False
                        
                    cap.release()
                    
                except Exception as e:
                    logger.warning(f"Error testing camera {camera.index}: {e}")
                    camera.is_working = False
    
    def get_working_cameras(self) -> List[CameraInfo]:
        """Get only cameras that are confirmed to be working."""
        return [cam for cam in self.detected_cameras if cam.is_working]
    
    def get_best_camera(self) -> Optional[CameraInfo]:
        """Get the best available camera (highest resolution, working)."""
        working_cameras = self.get_working_cameras()
        
        if not working_cameras:
            return None
            
        # Sort by resolution (width * height) and prefer USB/external cameras
        def camera_score(cam: CameraInfo) -> tuple:
            resolution_score = 0
            if cam.resolution:
                resolution_score = cam.resolution[0] * cam.resolution[1]
                
            # Prefer USB cameras over built-in
            connection_priority = 0 if "USB" in cam.connection_type else 1
            
            return (resolution_score, -connection_priority, -cam.index)
        
        return max(working_cameras, key=camera_score)
    
    def refresh_cameras(self) -> List[CameraInfo]:
        """Refresh the camera list (useful for detecting newly connected cameras)."""
        return self.detect_all_cameras()
    
    def get_camera_by_index(self, index: int) -> Optional[CameraInfo]:
        """Get camera info by index."""
        for camera in self.detected_cameras:
            if camera.index == index:
                return camera
        return None
    
    def _check_wsl2_usb_support(self):
        """Check if we're in WSL2 and provide USB camera guidance."""
        try:
            with open('/proc/version', 'r') as f:
                version_info = f.read().lower()
                if 'microsoft' in version_info and 'wsl2' in version_info:
                    logger.warning("WSL2 detected: USB camera access requires USB passthrough setup")
                    logger.info("To use USB cameras in WSL2:")
                    logger.info("1. Install usbipd-win on Windows host")
                    logger.info("2. Run 'usbipd wsl list' in Windows PowerShell")
                    logger.info("3. Run 'usbipd wsl attach --busid <busid>' for your camera")
        except:
            pass
    
    def _log_troubleshooting_info(self):
        """Log troubleshooting information when no cameras are found."""
        logger.warning("No cameras detected. Troubleshooting:")
        
        # Check if running in WSL2
        try:
            with open('/proc/version', 'r') as f:
                if 'microsoft' in f.read().lower():
                    logger.info("WSL2 Environment:")
                    logger.info("- USB cameras need to be attached via usbipd-win")
                    logger.info("- Check Windows Device Manager for camera")
                    logger.info("- Ensure camera drivers are installed on Windows")
                    return
        except:
            pass
        
        # Check for video devices
        try:
            result = subprocess.run(['ls', '/dev/video*'], 
                                  capture_output=True, text=True, check=False)
            if result.returncode != 0:
                logger.info("No /dev/video* devices found")
                logger.info("- Check if camera is connected and recognized by system")
                logger.info("- Try: lsusb | grep -i camera")
                logger.info("- Check camera permissions")
            else:
                logger.info(f"Found video devices: {result.stdout.strip()}")
                logger.info("- Devices exist but OpenCV cannot access them")
                logger.info("- Check camera permissions: ls -la /dev/video*")
                logger.info("- Try adding user to video group: sudo usermod -a -G video $USER")
        except:
            pass

# Global instance for easy access
camera_detector = CameraDetector()

def get_available_cameras() -> List[CameraInfo]:
    """Convenience function to get available cameras."""
    return camera_detector.detect_all_cameras()

def get_working_cameras() -> List[CameraInfo]:
    """Convenience function to get only working cameras."""
    return camera_detector.get_working_cameras()