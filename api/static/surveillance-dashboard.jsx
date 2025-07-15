// api/static/js/surveillance-dashboard.jsx
import React, { useState, useEffect } from 'react';

const SurveillanceDashboard = () => {
    const [vehicles, setVehicles] = useState([]);
    const [alerts, setAlerts] = useState([]);
    const [isConnected, setIsConnected] = useState(false);
    
    useEffect(() => {
        const ws = new WebSocket('ws://localhost:8000/ws');
        
        ws.onopen = () => setIsConnected(true);
        ws.onclose = () => setIsConnected(false);
        
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            if (data.type === 'detection') {
                updateVehicles(data.detections);
            }
            
            if (data.anomalies?.length > 0) {
                setAlerts(prev => [...data.anomalies, ...prev].slice(0, 50));
            }
        };
        
        return () => ws.close();
    }, []);
    
    const updateVehicles = (detections) => {
        // Update vehicle list with new detections
        const newVehicles = detections.plates_detected?.map(plate => ({
            plate: plate.text,
            state: plate.state,
            confidence: plate.confidence,
            lastSeen: new Date(),
            alertLevel: 'normal'
        })) || [];
        
        setVehicles(prev => {
            // Merge with existing, update timestamps
            const merged = [...newVehicles];
            prev.forEach(v => {
                if (!merged.find(n => n.plate === v.plate)) {
                    merged.push(v);
                }
            });
            return merged.slice(0, 100); // Keep last 100
        });
    };
    
    return (
        <div className="surveillance-dashboard">
            <div className="status-bar">
                <span className={`connection-status ${isConnected ? 'connected' : 'disconnected'}`}>
                    {isConnected ? '● Connected' : '○ Disconnected'}
                </span>
            </div>
            
            <div className="main-grid">
                <div className="video-section">
                    <video id="live-feed" autoPlay />
                    <canvas id="overlay-canvas" />
                </div>
                
                <div className="alerts-section">
                    <h3>Alerts ({alerts.length})</h3>
                    {alerts.map((alert, idx) => (
                        <Alert key={idx} alert={alert} />
                    ))}
                </div>
                
                <div className="vehicles-section">
                    <h3>Recent Vehicles ({vehicles.length})</h3>
                    {vehicles.map((vehicle, idx) => (
                        <VehicleCard key={idx} vehicle={vehicle} />
                    ))}
                </div>
            </div>
        </div>
    );
};