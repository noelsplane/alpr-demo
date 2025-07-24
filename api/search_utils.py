"""
Search utilities for vehicle detection queries.
"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from sqlalchemy import and_, or_, func
from sqlalchemy.orm import Session
from models import PlateDetection, SessionDetection, VehicleAnomaly, VehicleTrack, SessionAlert

class VehicleSearchEngine:
    """Advanced vehicle search with filtering and analytics."""
    
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def search_vehicles(
        self,
        plate_text: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        vehicle_types: Optional[List[str]] = None,
        vehicle_makes: Optional[List[str]] = None,
        vehicle_colors: Optional[List[str]] = None,
        states: Optional[List[str]] = None,
        cameras: Optional[List[str]] = None,
        has_anomalies: Optional[bool] = None,
        anomaly_types: Optional[List[str]] = None,
        confidence_threshold: float = 0.0,
        limit: int = 100,
        offset: int = 0,
        include_no_plate_vehicles: bool = False
    ) -> Tuple[List[Dict], int]:
        """
        Search vehicles with advanced filtering across all detection sources.
        Returns (results, total_count)
        """
        all_results = []
        
        # 1. Search regular upload detections (PlateDetection)
        upload_results = self._search_upload_detections(
            plate_text, start_time, end_time, vehicle_types, vehicle_makes,
            vehicle_colors, states, cameras, confidence_threshold
        )
        all_results.extend(upload_results)
        
        # 2. Search surveillance session detections (SessionDetection)
        session_results = self._search_session_detections(
            plate_text, start_time, end_time, vehicle_types, vehicle_makes,
            vehicle_colors, cameras, confidence_threshold
        )
        all_results.extend(session_results)
        
        # 3. Search vehicles without plates (VehicleAnomaly)
        if include_no_plate_vehicles or not plate_text:
            no_plate_results = self._search_no_plate_vehicles(
                start_time, end_time, vehicle_types, cameras
            )
            all_results.extend(no_plate_results)
        
        # 4. Apply anomaly filtering if requested
        if has_anomalies is not None or anomaly_types:
            all_results = self._filter_by_anomalies(
                all_results, has_anomalies, anomaly_types, start_time, end_time
            )
        
        # Sort by timestamp (most recent first)
        all_results.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        # Apply pagination
        total_count = len(all_results)
        paginated_results = all_results[offset:offset + limit]
        
        return paginated_results, total_count
    
    def _search_upload_detections(self, plate_text, start_time, end_time, 
                                 vehicle_types, vehicle_makes, vehicle_colors, 
                                 states, cameras, confidence_threshold):
        """Search upload detections from PlateDetection table."""
        query = self.db.query(PlateDetection)
        
        # Apply filters
        filters = []
        if plate_text:
            filters.append(PlateDetection.plate_text.ilike(f"%{plate_text}%"))
        if start_time:
            filters.append(PlateDetection.timestamp >= start_time)
        if end_time:
            filters.append(PlateDetection.timestamp <= end_time)
        if vehicle_types:
            filters.append(PlateDetection.vehicle_type.in_(vehicle_types))
        if vehicle_makes:
            filters.append(PlateDetection.vehicle_make.in_(vehicle_makes))
        if vehicle_colors:
            filters.append(PlateDetection.vehicle_color.in_(vehicle_colors))
        if states:
            filters.append(PlateDetection.state.in_(states))
        if cameras:
            filters.append(PlateDetection.camera_id.in_(cameras))
        if confidence_threshold > 0:
            filters.append(PlateDetection.confidence >= confidence_threshold)
        
        if filters:
            query = query.filter(and_(*filters))
        
        results = query.order_by(PlateDetection.timestamp.desc()).all()
        
        # Format results
        formatted_results = []
        for detection in results:
            result = {
                'id': f"upload_{detection.id}",
                'source': 'upload',
                'plate_text': detection.plate_text,
                'confidence': detection.confidence,
                'timestamp': detection.timestamp.isoformat(),
                'camera_id': detection.camera_id,
                'image_name': detection.image_name,
                'plate_image': detection.plate_image_base64,
                'vehicle': {
                    'type': detection.vehicle_type,
                    'make': detection.vehicle_make,
                    'model': detection.vehicle_model,
                    'color': detection.vehicle_color,
                    'year': detection.vehicle_year
                },
                'state': detection.state,
                'state_confidence': detection.state_confidence,
                'box': [detection.x1, detection.y1, detection.x2, detection.y2],
                'anomalies': []
            }
            formatted_results.append(result)
        
        return formatted_results
    
    def _search_session_detections(self, plate_text, start_time, end_time,
                                  vehicle_types, vehicle_makes, vehicle_colors,
                                  cameras, confidence_threshold):
        """Search surveillance session detections from SessionDetection table."""
        query = self.db.query(SessionDetection)
        
        # Apply filters
        filters = []
        if plate_text:
            filters.append(SessionDetection.plate_text.ilike(f"%{plate_text}%"))
        if start_time:
            filters.append(SessionDetection.detection_time >= start_time)
        if end_time:
            filters.append(SessionDetection.detection_time <= end_time)
        if vehicle_types:
            filters.append(SessionDetection.vehicle_type.in_(vehicle_types))
        if vehicle_makes:
            filters.append(SessionDetection.vehicle_make.in_(vehicle_makes))
        if vehicle_colors:
            filters.append(SessionDetection.vehicle_color.in_(vehicle_colors))
        if cameras:
            filters.append(SessionDetection.camera_id.in_(cameras))
        if confidence_threshold > 0:
            filters.append(SessionDetection.confidence >= confidence_threshold)
        
        if filters:
            query = query.filter(and_(*filters))
        
        results = query.order_by(SessionDetection.detection_time.desc()).all()
        
        # Format results
        formatted_results = []
        for detection in results:
            result = {
                'id': f"session_{detection.id}",
                'source': 'surveillance',
                'plate_text': detection.plate_text,
                'confidence': detection.confidence,
                'timestamp': detection.detection_time.isoformat(),
                'camera_id': detection.camera_id,
                'session_id': detection.session_id,
                'frame_id': detection.frame_id,
                'plate_image': detection.plate_image_base64,
                'vehicle': {
                    'type': detection.vehicle_type,
                    'make': detection.vehicle_make,
                    'model': detection.vehicle_model,
                    'color': detection.vehicle_color,
                    'year': None  # Session detections don't store year
                },
                'state': detection.state,
                'state_confidence': detection.state_confidence,
                'anomalies': []
            }
            formatted_results.append(result)
        
        return formatted_results
    
    def _search_no_plate_vehicles(self, start_time, end_time, vehicle_types, cameras):
        """Search vehicles without recognizable plates from VehicleAnomaly table."""
        query = self.db.query(VehicleAnomaly).filter(
            VehicleAnomaly.anomaly_type.in_(['NO_PLATE_VEHICLE', 'SUSPICIOUS_NO_PLATE'])
        )
        
        # Apply filters
        filters = []
        if start_time:
            filters.append(VehicleAnomaly.detected_time >= start_time)
        if end_time:
            filters.append(VehicleAnomaly.detected_time <= end_time)
        
        if filters:
            query = query.filter(and_(*filters))
        
        anomalies = query.order_by(VehicleAnomaly.detected_time.desc()).all()
        
        # Format results
        formatted_results = []
        for anomaly in anomalies:
            # Get associated vehicle track info
            track = self.db.query(VehicleTrack).filter(
                VehicleTrack.track_id == anomaly.track_id
            ).first()
            
            # Filter by vehicle type if specified
            if vehicle_types and track and track.vehicle_type not in vehicle_types:
                continue
            
            result = {
                'id': f"no_plate_{anomaly.id}",
                'source': 'no_plate_vehicle',
                'plate_text': None,  # No plate detected
                'confidence': 0,
                'timestamp': anomaly.detected_time.isoformat(),
                'camera_id': 'unknown',  # Track doesn't store camera info
                'track_id': anomaly.track_id,
                'plate_image': anomaly.image_data,
                'vehicle': {
                    'type': track.vehicle_type if track else 'Unknown',
                    'make': track.vehicle_make if track else None,
                    'model': track.vehicle_model if track else None,
                    'color': track.vehicle_color if track else None,
                    'year': track.vehicle_year if track else None
                },
                'state': None,
                'state_confidence': 0,
                'anomalies': [{
                    'type': anomaly.anomaly_type,
                    'severity': anomaly.severity,
                    'timestamp': anomaly.detected_time.isoformat(),
                    'message': anomaly.message
                }],
                'no_plate_reason': anomaly.message
            }
            formatted_results.append(result)
        
        return formatted_results
    
    def _filter_by_anomalies(self, results, has_anomalies, anomaly_types, start_time, end_time):
        """Filter results based on anomaly criteria."""
        filtered_results = []
        
        for result in results:
            # Get anomalies for this vehicle
            if result.get('plate_text'):
                anomalies = self._get_vehicle_anomalies(
                    result['plate_text'], start_time, end_time
                )
                result['anomalies'].extend(anomalies)
            
            # Apply anomaly filters
            has_vehicle_anomalies = len(result['anomalies']) > 0
            
            # Filter by has_anomalies
            if has_anomalies is not None:
                if has_anomalies and not has_vehicle_anomalies:
                    continue
                elif not has_anomalies and has_vehicle_anomalies:
                    continue
            
            # Filter by anomaly types
            if anomaly_types:
                anomaly_type_match = any(
                    anomaly['type'] in anomaly_types 
                    for anomaly in result['anomalies']
                )
                if not anomaly_type_match:
                    continue
            
            filtered_results.append(result)
        
        return filtered_results
    
    def _get_vehicle_anomalies(
        self, 
        plate_text: str, 
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict]:
        """Get anomalies for a specific vehicle."""
        if not plate_text:
            return []
        
        query = self.db.query(VehicleAnomaly).filter(
            VehicleAnomaly.plate_text == plate_text
        )
        
        if start_time:
            query = query.filter(VehicleAnomaly.detected_time >= start_time)
        
        if end_time:
            query = query.filter(VehicleAnomaly.detected_time <= end_time)
        
        anomalies = query.all()
        
        return [{
            'type': a.anomaly_type,
            'severity': a.severity,
            'timestamp': a.detected_time.isoformat(),
            'message': a.message
        } for a in anomalies]
    
    def get_search_analytics(
        self,
        time_range: str = "24h",
        vehicle_types: Optional[List[str]] = None,
        cameras: Optional[List[str]] = None
    ) -> Dict:
        """Get analytics for search dashboard."""
        # Parse time range
        end_time = datetime.now()
        if time_range == "1h":
            start_time = end_time - timedelta(hours=1)
        elif time_range == "24h":
            start_time = end_time - timedelta(hours=24)
        elif time_range == "7d":
            start_time = end_time - timedelta(days=7)
        elif time_range == "30d":
            start_time = end_time - timedelta(days=30)
        else:
            start_time = end_time - timedelta(hours=24)
        
        # Base filters
        filters = [PlateDetection.timestamp.between(start_time, end_time)]
        
        if vehicle_types:
            filters.append(PlateDetection.vehicle_type.in_(vehicle_types))
        
        if cameras:
            filters.append(PlateDetection.camera_id.in_(cameras))
        
        # Get statistics
        base_query = self.db.query(PlateDetection).filter(and_(*filters))
        
        # Total detections
        total_detections = base_query.count()
        
        # Unique vehicles
        unique_vehicles = base_query.distinct(PlateDetection.plate_text).count()
        
        # Recent detections (last hour)
        recent_time = datetime.now() - timedelta(hours=1)
        recent_detections = base_query.filter(
            PlateDetection.timestamp >= recent_time
        ).count()
        
        # Suspicious vehicles (with anomalies)
        suspicious_plates = self.db.query(VehicleAnomaly.plate_text)\
            .filter(VehicleAnomaly.detected_time.between(start_time, end_time))\
            .distinct()\
            .count()
        
        # Frequent visitors (5+ appearances)
        frequent_query = self.db.query(
            PlateDetection.plate_text,
            func.count(PlateDetection.id).label('count')
        ).filter(and_(*filters))\
         .group_by(PlateDetection.plate_text)\
         .having(func.count(PlateDetection.id) >= 5)
        
        frequent_visitors = frequent_query.count()
        
        # Active anomalies (last hour)
        active_anomalies = self.db.query(VehicleAnomaly).filter(
            VehicleAnomaly.detected_time >= recent_time
        ).count()
        
        # Vehicle type distribution
        type_distribution = {}
        type_query = self.db.query(
            PlateDetection.vehicle_type,
            func.count(PlateDetection.id).label('count')
        ).filter(
            and_(*filters),
            PlateDetection.vehicle_type.isnot(None)
        ).group_by(PlateDetection.vehicle_type).all()
        
        for vtype, count in type_query:
            type_distribution[vtype] = count
        
        # Top vehicles by appearance
        top_vehicles = []
        top_query = self.db.query(
            PlateDetection.plate_text,
            func.count(PlateDetection.id).label('count')
        ).filter(
            and_(*filters),
            PlateDetection.plate_text.isnot(None)
        ).group_by(PlateDetection.plate_text)\
         .order_by(func.count(PlateDetection.id).desc())\
         .limit(10)\
         .all()
        
        for plate, count in top_query:
            # Get vehicle details
            latest = base_query.filter(
                PlateDetection.plate_text == plate
            ).order_by(PlateDetection.timestamp.desc()).first()
            
            if latest:
                top_vehicles.append({
                    'plate_text': plate,
                    'count': count,
                    'vehicle_info': {
                        'type': latest.vehicle_type,
                        'make': latest.vehicle_make,
                        'model': latest.vehicle_model,
                        'color': latest.vehicle_color
                    }
                })
        
        return {
            'summary': {
                'total_detections': total_detections,
                'unique_vehicles': unique_vehicles,
                'recent_detections': recent_detections,
                'suspicious_vehicles': suspicious_plates,
                'frequent_visitors': frequent_visitors,
                'active_anomalies': active_anomalies
            },
            'vehicle_type_distribution': type_distribution,
            'top_vehicles': top_vehicles,
            'time_range': time_range,
            'filters': {
                'vehicle_types': vehicle_types,
                'cameras': cameras
            }
        }
    
    def get_vehicle_timeline(self, plate_text: str) -> List[Dict]:
        """Get timeline of all sightings for a specific vehicle."""
        detections = self.db.query(PlateDetection).filter(
            PlateDetection.plate_text == plate_text
        ).order_by(PlateDetection.timestamp).all()
        
        timeline = []
        for detection in detections:
            timeline.append({
                'timestamp': detection.timestamp.isoformat(),
                'camera_id': detection.camera_id,
                'confidence': detection.confidence,
                'state': detection.state,
                'image': detection.plate_image_base64,
                'vehicle_info': {
                    'type': detection.vehicle_type,
                    'make': detection.vehicle_make,
                    'model': detection.vehicle_model,
                    'color': detection.vehicle_color
                }
            })
        
        return timeline
