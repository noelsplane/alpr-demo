// Camera Management System
// Complete implementation for ALPR multi-camera surveillance

class CameraManager {
    constructor() {
        this.cameras = new Map();
        this.activeStreams = new Map();
        this.websockets = new Map();
        this.callbacks = {
            onDetection: null,
            onAlert: null,
            onStatusChange: null
        };
    }
    
    // Initialize camera manager
    async init() {
        await this.loadCameras();
        this.setupGlobalWebSocket();
    }
    
    // Load cameras from API
    async loadCameras() {
        try {
            const response = await fetch('/api/v1/cameras');
            const data = await response.json();
            
            data.cameras.forEach(camera => {
                this.cameras.set(camera.id, camera);
            });
            
            return data.cameras;
        } catch (error) {
            console.error('Error loading cameras:', error);
            return [];
        }
    }
    
    // Setup global WebSocket for all cameras
    setupGlobalWebSocket() {
        const ws = new WebSocket('ws://localhost:8000/ws/surveillance');
        
        ws.onopen = () => {
            console.log('Connected to global surveillance WebSocket');
        };
        
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleGlobalMessage(data);
        };
        
        ws.onerror = (error) => {
            console.error('Global WebSocket error:', error);
        };
        
        ws.onclose = () => {
            console.log('Global WebSocket disconnected');
            // Retry connection after 3 seconds
            setTimeout(() => this.setupGlobalWebSocket(), 3000);
        };
        
        this.globalWs = ws;
    }
    
    // Connect to specific camera WebSocket
    connectToCamera(cameraId) {
        if (this.websockets.has(cameraId)) {
            return; // Already connected
        }
        
        const ws = new WebSocket(`ws://localhost:8000/ws/camera/${cameraId}`);
        
        ws.onopen = () => {
            console.log(`Connected to camera ${cameraId}`);
            this.updateCameraStatus(cameraId, 'connected');
        };
        
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleCameraMessage(cameraId, data);
        };
        
        ws.onerror = (error) => {
            console.error(`Camera ${cameraId} WebSocket error:`, error);
            this.updateCameraStatus(cameraId, 'error');
        };
        
        ws.onclose = () => {
            console.log(`Camera ${cameraId} WebSocket disconnected`);
            this.websockets.delete(cameraId);
            this.updateCameraStatus(cameraId, 'disconnected');
        };
        
        this.websockets.set(cameraId, ws);
    }
    
    // Start camera stream
    async startCameraStream(cameraId, source = '0') {
        try {
            const response = await fetch(`/api/v1/cameras/${cameraId}/stream`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ source })
            });
            
            if (response.ok) {
                const data = await response.json();
                this.activeStreams.set(cameraId, {
                    status: 'streaming',
                    startTime: new Date()
                });
                this.connectToCamera(cameraId);
                return data;
            } else {
                throw new Error('Failed to start camera stream');
            }
        } catch (error) {
            console.error(`Error starting camera ${cameraId}:`, error);
            throw error;
        }
    }
    
    // Stop camera stream
    async stopCameraStream(cameraId) {
        // Close WebSocket connection
        const ws = this.websockets.get(cameraId);
        if (ws) {
            ws.close();
            this.websockets.delete(cameraId);
        }
        
        // Remove from active streams
        this.activeStreams.delete(cameraId);
        this.updateCameraStatus(cameraId, 'stopped');
    }
    
    // Handle global WebSocket messages
    handleGlobalMessage(data) {
        if (data.type === 'detection') {
            this.handleDetection(data.data);
        } else if (data.type === 'anomaly_alert') {
            this.handleAlert(data.data);
        } else if (data.type === 'stats_update') {
            this.handleStatsUpdate(data.data);
        }
    }
    
    // Handle camera-specific messages
    handleCameraMessage(cameraId, data) {
        if (data.type === 'frame') {
            this.updateCameraFrame(cameraId, data.frame);
        } else if (data.type === 'detection') {
            this.handleCameraDetection(cameraId, data.detection);
        }
    }
    
    // Handle new detection
    handleDetection(detection) {
        // Update UI with new detection
        if (this.callbacks.onDetection) {
            this.callbacks.onDetection(detection);
        }
        
        // Update camera-specific UI if needed
        if (detection.camera_id) {
            this.updateCameraDetection(detection.camera_id, detection);
        }
    }
    
    // Handle anomaly alert
    handleAlert(alert) {
        if (this.callbacks.onAlert) {
            this.callbacks.onAlert(alert);
        }
        
        // Show notification
        this.showAlertNotification(alert);
    }
    
    // Handle stats update
    handleStatsUpdate(stats) {
        // Update UI statistics
        this.updateStatistics(stats);
    }
    
    // Update camera status
    updateCameraStatus(cameraId, status) {
        const camera = this.cameras.get(cameraId);
        if (camera) {
            camera.status = status;
            if (this.callbacks.onStatusChange) {
                this.callbacks.onStatusChange(cameraId, status);
            }
        }
    }
    
    // Update camera detection display
    updateCameraDetection(cameraId, detection) {
        // Find camera feed element
        const feedElement = document.querySelector(`[data-camera-id="${cameraId}"]`);
        if (feedElement) {
            const plateElement = feedElement.querySelector('.detection-plate');
            const timeElement = feedElement.querySelector('.detection-time');
            
            if (plateElement && timeElement) {
                plateElement.textContent = detection.plate_text || 'No plate';
                timeElement.textContent = 'Just now';
            }
        }
    }
    
    // Show alert notification
    showAlertNotification(alert) {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = 'alert-notification';
        notification.innerHTML = `
            <div class="alert-icon ${alert.severity}">⚠️</div>
            <div class="alert-content">
                <div class="alert-title">${alert.type}</div>
                <div class="alert-message">${alert.message}</div>
                <div class="alert-time">${new Date(alert.timestamp).toLocaleTimeString()}</div>
            </div>
            <button class="alert-close" onclick="this.parentElement.remove()">×</button>
        `;
        
        // Add to notification container
        let container = document.getElementById('notificationContainer');
        if (!container) {
            container = document.createElement('div');
            container.id = 'notificationContainer';
            container.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                z-index: 1000;
                display: flex;
                flex-direction: column;
                gap: 10px;
            `;
            document.body.appendChild(container);
        }
        
        container.appendChild(notification);
        
        // Auto-remove after 10 seconds
        setTimeout(() => {
            notification.remove();
        }, 10000);
    }
    
    // Update statistics display
    updateStatistics(stats) {
        if (stats.tracking_stats) {
            document.getElementById('totalDetections').textContent = 
                stats.tracking_stats.total_appearances || 0;
            document.getElementById('activeCameras').textContent = 
                `${this.activeStreams.size}/${this.cameras.size}`;
        }
    }
    
    // Get camera by ID
    getCamera(cameraId) {
        return this.cameras.get(cameraId);
    }
    
    // Get all cameras
    getAllCameras() {
        return Array.from(this.cameras.values());
    }
    
    // Get active cameras
    getActiveCameras() {
        return Array.from(this.cameras.values()).filter(cam => 
            this.activeStreams.has(cam.id)
        );
    }
    
    // Set callback functions
    on(event, callback) {
        if (this.callbacks.hasOwnProperty(event)) {
            this.callbacks[event] = callback;
        }
    }
}

// Vehicle search functionality
class VehicleSearch {
    constructor() {
        this.searchResults = [];
        this.filters = {
            plate: '',
            timeRange: '24h',
            vehicleTypes: [],
            alerts: [],
            cameras: []
        };
    }
    
    // Perform vehicle search
    async search(filters = {}) {
        this.filters = { ...this.filters, ...filters };
        
        const params = new URLSearchParams();
        
        if (this.filters.plate) {
            params.append('plate', this.filters.plate);
        }
        
        params.append('timeRange', this.filters.timeRange);
        
        if (this.filters.vehicleTypes.length > 0) {
            params.append('vehicleTypes', this.filters.vehicleTypes.join(','));
        }
        
        if (this.filters.alerts.length > 0) {
            params.append('alerts', this.filters.alerts.join(','));
        }
        
        if (this.filters.cameras.length > 0) {
            params.append('cameras', this.filters.cameras.join(','));
        }
        
        try {
            const response = await fetch(`/api/v1/search/vehicles?${params}`);
            const data = await response.json();
            
            this.searchResults = data.results;
            return data;
        } catch (error) {
            console.error('Search error:', error);
            throw error;
        }
    }
    
    // Get vehicle journey
    async getVehicleJourney(plateText) {
        try {
            const response = await fetch(`/api/v1/search/journey/${plateText}`);
            const data = await response.json();
            return data;
        } catch (error) {
            console.error('Error getting vehicle journey:', error);
            throw error;
        }
    }
    
    // Get frequent vehicles
    async getFrequentVehicles(timeWindowHours = 24, minAppearances = 5, cameraId = null) {
        const params = new URLSearchParams({
            time_window_hours: timeWindowHours,
            min_appearances: minAppearances
        });
        
        if (cameraId) {
            params.append('camera_id', cameraId);
        }
        
        try {
            const response = await fetch(`/api/v1/search/frequent-vehicles?${params}`);
            const data = await response.json();
            return data;
        } catch (error) {
            console.error('Error getting frequent vehicles:', error);
            throw error;
        }
    }
    
    // Display search results
    displayResults(containerId) {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        container.innerHTML = '';
        
        this.searchResults.forEach(result => {
            const resultElement = this.createResultElement(result);
            container.appendChild(resultElement);
        });
    }
    
    // Create result element
    createResultElement(result) {
        const element = document.createElement('div');
        element.className = 'search-result-item';
        
        const vehicleDesc = result.vehicle ? 
            `${result.vehicle.color || ''} ${result.vehicle.make || ''} ${result.vehicle.model || ''}`.trim() || 
            'Unknown Vehicle' : 
            'Unknown Vehicle';
        
        element.innerHTML = `
            <div class="result-header">
                <div class="result-plate">${result.plate_text}</div>
                <div class="result-confidence">${Math.round(result.confidence * 100)}%</div>
            </div>
            <div class="result-vehicle">${vehicleDesc}</div>
            <div class="result-details">
                <div class="result-time">${new Date(result.timestamp).toLocaleString()}</div>
                <div class="result-camera">Camera: ${result.camera_id || 'Unknown'}</div>
            </div>
            ${result.plate_image ? `
                <div class="result-image">
                    <img src="data:image/jpeg;base64,${result.plate_image}" alt="Plate">
                </div>
            ` : ''}
            ${result.alerts && result.alerts.length > 0 ? `
                <div class="result-alerts">
                    ${result.alerts.map(alert => `
                        <span class="alert-badge ${alert.severity}">${alert.type}</span>
                    `).join('')}
                </div>
            ` : ''}
        `;
        
        // Add click handler for journey view
        element.addEventListener('click', () => {
            this.showVehicleJourney(result.plate_text);
        });
        
        return element;
    }
    
    // Show vehicle journey
    async showVehicleJourney(plateText) {
        try {
            const journey = await this.getVehicleJourney(plateText);
            
            // Create journey modal
            const modal = document.createElement('div');
            modal.className = 'journey-modal';
            modal.innerHTML = `
                <div class="journey-content">
                    <div class="journey-header">
                        <h3>Vehicle Journey: ${plateText}</h3>
                        <button onclick="this.closest('.journey-modal').remove()">×</button>
                    </div>
                    <div class="journey-info">
                        <div class="vehicle-info">
                            <h4>Vehicle Details</h4>
                            <p>${journey.vehicle_info.color || ''} ${journey.vehicle_info.make || ''} ${journey.vehicle_info.model || ''}</p>
                        </div>
                        <div class="journey-stats">
                            <div class="stat">
                                <span class="stat-label">First Seen:</span>
                                <span class="stat-value">${new Date(journey.journey_stats.first_seen).toLocaleString()}</span>
                            </div>
                            <div class="stat">
                                <span class="stat-label">Last Seen:</span>
                                <span class="stat-value">${new Date(journey.journey_stats.last_seen).toLocaleString()}</span>
                            </div>
                            <div class="stat">
                                <span class="stat-label">Duration:</span>
                                <span class="stat-value">${Math.round(journey.journey_stats.duration_minutes)} minutes</span>
                            </div>
                            <div class="stat">
                                <span class="stat-label">Sightings:</span>
                                <span class="stat-value">${journey.journey_stats.total_sightings}</span>
                            </div>
                        </div>
                    </div>
                    <div class="journey-timeline">
                        <h4>Timeline</h4>
                        <div class="timeline-items">
                            ${journey.timeline.map(item => `
                                <div class="timeline-item">
                                    <div class="timeline-time">${new Date(item.timestamp).toLocaleTimeString()}</div>
                                    <div class="timeline-camera">${item.camera_name}</div>
                                    ${item.image ? `
                                        <div class="timeline-image">
                                            <img src="data:image/jpeg;base64,${item.image}" alt="Detection">
                                        </div>
                                    ` : ''}
                                </div>
                            `).join('')}
                        </div>
                    </div>
                </div>
            `;
            
            document.body.appendChild(modal);
        } catch (error) {
            console.error('Error showing journey:', error);
        }
    }
}

// Alert notification styles
const alertStyles = `
    .alert-notification {
        background: #2a2a2a;
        border: 1px solid #444;
        border-radius: 8px;
        padding: 1rem;
        display: flex;
        align-items: center;
        gap: 1rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        animation: slideIn 0.3s ease;
        max-width: 400px;
    }
    
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    .alert-icon {
        font-size: 1.5rem;
    }
    
    .alert-icon.low { color: #2196F3; }
    .alert-icon.medium { color: #ff9800; }
    .alert-icon.high { color: #f44336; }
    .alert-icon.critical { 
        color: #f44336; 
        animation: pulse 1s infinite;
    }
    
    .alert-content {
        flex: 1;
    }
    
    .alert-title {
        font-weight: 600;
        margin-bottom: 0.25rem;
    }
    
    .alert-message {
        font-size: 0.9rem;
        color: #ccc;
    }
    
    .alert-time {
        font-size: 0.8rem;
        color: #888;
        margin-top: 0.25rem;
    }
    
    .alert-close {
        background: none;
        border: none;
        color: #888;
        font-size: 1.5rem;
        cursor: pointer;
        padding: 0;
        width: 32px;
        height: 32px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 4px;
        transition: all 0.2s ease;
    }
    
    .alert-close:hover {
        background: rgba(255, 255, 255, 0.1);
        color: #fff;
    }
    
    .journey-modal {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0, 0, 0, 0.8);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 2000;
        padding: 2rem;
    }
    
    .journey-content {
        background: #1a1a1a;
        border-radius: 12px;
        max-width: 800px;
        width: 100%;
        max-height: 80vh;
        overflow: auto;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
    }
    
    .journey-header {
        padding: 1.5rem;
        border-bottom: 1px solid #333;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .journey-header h3 {
        margin: 0;
        color: #fff;
    }
    
    .journey-header button {
        background: none;
        border: none;
        color: #888;
        font-size: 2rem;
        cursor: pointer;
        padding: 0;
        width: 40px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 4px;
        transition: all 0.2s ease;
    }
    
    .journey-header button:hover {
        background: rgba(255, 255, 255, 0.1);
        color: #fff;
    }
    
    .journey-info {
        padding: 1.5rem;
        display: grid;
        grid-template-columns: 1fr 2fr;
        gap: 2rem;
    }
    
    .vehicle-info h4,
    .journey-stats h4,
    .journey-timeline h4 {
        margin: 0 0 1rem 0;
        color: #888;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .journey-stats {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 1rem;
    }
    
    .journey-stats .stat {
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
    }
    
    .journey-stats .stat-label {
        color: #888;
        font-size: 0.85rem;
    }
    
    .journey-stats .stat-value {
        color: #4CAF50;
        font-weight: 600;
    }
    
    .journey-timeline {
        padding: 0 1.5rem 1.5rem;
    }
    
    .timeline-items {
        display: flex;
        flex-direction: column;
        gap: 1rem;
    }
    
    .timeline-item {
        display: grid;
        grid-template-columns: 100px 1fr auto;
        gap: 1rem;
        padding: 1rem;
        background: #2a2a2a;
        border-radius: 8px;
        align-items: center;
    }
    
    .timeline-time {
        color: #888;
        font-size: 0.9rem;
    }
    
    .timeline-camera {
        font-weight: 500;
    }
    
    .timeline-image img {
        height: 60px;
        border-radius: 4px;
    }
`;

// Add styles to document
const styleSheet = document.createElement('style');
styleSheet.textContent = alertStyles;
document.head.appendChild(styleSheet);

// Export classes
window.CameraManager = CameraManager;
window.VehicleSearch = VehicleSearch;