import cv2
from flask import Flask, Response
import socket

app = Flask(__name__)

def test_cameras():
    """Test all camera indices"""
    print("\nTesting available cameras...")
    available_cameras = []
    
    for i in range(5):  # Test indices 0-4
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Camera {i}: WORKING")
                available_cameras.append(i)
            else:
                print(f"Camera {i}: Opens but can't read frames")
            cap.release()
        else:
            print(f"Camera {i}: Not available")
    
    return available_cameras

def get_local_ip():
    """Get the local IP address"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]
    s.close()
    return ip

# Global camera variable
camera = None
camera_index = None

def generate_frames():
    """Generate frames from the camera"""
    global camera, camera_index
    
    if camera is None:
        return
        
    while True:
        success, frame = camera.read()
        if not success:
            print(f"Failed to read frame from camera {camera_index}")
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """Home page"""
    ip = get_local_ip()
    status = "Camera Active" if camera is not None else "No Camera Found"
    return f'''
    <h1>Webcam Stream Status: {status}</h1>
    <p>Camera Index: {camera_index if camera_index is not None else "None"}</p>
    <p>Stream URL for Linux ALPR: <strong>http://{ip}:5000/video_feed</strong></p>
    <hr>
    <h2>Live Feed:</h2>
    <img src="/video_feed" width="640" onerror="this.src=''; this.alt='No video feed available';">
    '''

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    if camera is None:
        return "No camera available", 503
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Test cameras first
    available = test_cameras()
    
    if available:
        # Use the first available camera
        camera_index = available[0]
        camera = cv2.VideoCapture(camera_index)
        print(f"\nUsing camera index: {camera_index}")
    else:
        print("\nERROR: No cameras detected!")
        print("\nPossible solutions:")
        print("1. Make sure your USB camera is plugged in")
        print("2. Check if another application is using the camera")
        print("3. Try closing Zoom, Teams, or other video apps")
        print("4. Check Windows Device Manager for camera issues")
    
    ip = get_local_ip()
    print(f"\n{'='*50}")
    print(f"Starting server anyway...")
    print(f"Your computer's IP: {ip}")
    print(f"Check: http://localhost:5000")
    print(f"{'='*50}\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)