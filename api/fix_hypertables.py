#!/usr/bin/env python3
"""
Recreate tables with TimescaleDB-compatible structure
"""

import os
from dotenv import load_dotenv
from sqlalchemy import text
from database_config import db_config

load_dotenv()

def recreate_tables_for_timescale():
    """Drop and recreate tables with TimescaleDB-compatible structure."""
    
    engine = db_config.init_engine()
    
    with engine.connect() as conn:
        print("Recreating tables for TimescaleDB...")
        
        # First, drop existing tables
        print("\nDropping existing tables...")
        tables_to_drop = [
            'vehicle_anomalies',
            'track_plate_associations', 
            'vehicle_tracks',
            'session_alerts',
            'session_detections',
            'surveillance_sessions',
            'detections'
        ]
        
        for table in tables_to_drop:
            try:
                conn.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))
                conn.commit()
                print(f"  ✓ Dropped {table}")
            except Exception as e:
                print(f"  ✗ Error dropping {table}: {e}")
                conn.rollback()
        
        print("\nCreating tables with TimescaleDB-compatible structure...")
        
        # Create tables with composite primary keys including time columns
        table_definitions = [
            # Main detections table
            """
            CREATE TABLE detections (
                id BIGSERIAL,
                timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                plate_text VARCHAR(20),
                confidence FLOAT,
                image_name VARCHAR(255),
                camera_id VARCHAR(50),
                x1 INTEGER,
                y1 INTEGER,
                x2 INTEGER,
                y2 INTEGER,
                plate_image_base64 TEXT,
                state VARCHAR(2),
                state_confidence FLOAT DEFAULT 0.0,
                vehicle_type VARCHAR(50),
                vehicle_type_confidence FLOAT DEFAULT 0.0,
                vehicle_make VARCHAR(50),
                vehicle_make_confidence FLOAT DEFAULT 0.0,
                vehicle_model VARCHAR(50),
                vehicle_model_confidence FLOAT DEFAULT 0.0,
                vehicle_color VARCHAR(30),
                vehicle_color_confidence FLOAT DEFAULT 0.0,
                vehicle_year VARCHAR(4),
                vehicle_year_confidence FLOAT DEFAULT 0.0,
                PRIMARY KEY (id, timestamp)
            )
            """,
            
            # Surveillance sessions
            """
            CREATE TABLE surveillance_sessions (
                id SERIAL PRIMARY KEY,
                start_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                end_time TIMESTAMPTZ,
                status VARCHAR(20) DEFAULT 'active',
                total_detections INTEGER DEFAULT 0,
                total_vehicles INTEGER DEFAULT 0,
                total_alerts INTEGER DEFAULT 0,
                camera_ids TEXT,
                session_config TEXT,
                session_notes TEXT
            )
            """,
            
            # Session detections
            """
            CREATE TABLE session_detections (
                id BIGSERIAL,
                detection_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                session_id INTEGER REFERENCES surveillance_sessions(id),
                plate_text VARCHAR(20),
                confidence FLOAT,
                state VARCHAR(2),
                state_confidence FLOAT DEFAULT 0.0,
                frame_id VARCHAR(50),
                camera_id VARCHAR(50),
                plate_image_base64 TEXT,
                vehicle_type VARCHAR(50),
                vehicle_color VARCHAR(30),
                vehicle_make VARCHAR(50),
                vehicle_model VARCHAR(50),
                PRIMARY KEY (id, detection_time)
            )
            """,
            
            # Session alerts
            """
            CREATE TABLE session_alerts (
                id BIGSERIAL,
                alert_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                session_id INTEGER REFERENCES surveillance_sessions(id),
                alert_type VARCHAR(50) NOT NULL,
                severity VARCHAR(20) NOT NULL,
                plate_text VARCHAR(20),
                camera_id VARCHAR(50),
                message TEXT,
                details TEXT,
                PRIMARY KEY (id, alert_time)
            )
            """,
            
            # Vehicle tracks
            """
            CREATE TABLE vehicle_tracks (
                id SERIAL PRIMARY KEY,
                track_id VARCHAR(100) UNIQUE NOT NULL,
                first_seen TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                last_seen TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                vehicle_type VARCHAR(50),
                vehicle_make VARCHAR(50),
                vehicle_model VARCHAR(50),
                vehicle_color VARCHAR(30),
                vehicle_year VARCHAR(4),
                type_confidence FLOAT DEFAULT 0.0,
                make_confidence FLOAT DEFAULT 0.0,
                model_confidence FLOAT DEFAULT 0.0,
                color_confidence FLOAT DEFAULT 0.0,
                year_confidence FLOAT DEFAULT 0.0,
                total_appearances INTEGER DEFAULT 0,
                is_suspicious BOOLEAN DEFAULT FALSE,
                has_no_plate BOOLEAN DEFAULT FALSE,
                anomaly_count INTEGER DEFAULT 0
            )
            """,
            
            # Track plate associations
            """
            CREATE TABLE track_plate_associations (
                id SERIAL PRIMARY KEY,
                track_id VARCHAR(100) NOT NULL,
                plate_text VARCHAR(20) NOT NULL,
                first_seen TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                last_seen TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                appearance_count INTEGER DEFAULT 1
            )
            """,
            
            # Vehicle anomalies
            """
            CREATE TABLE vehicle_anomalies (
                id BIGSERIAL,
                detected_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                track_id VARCHAR(100) NOT NULL,
                anomaly_type VARCHAR(50) NOT NULL,
                severity VARCHAR(20) NOT NULL,
                plate_text VARCHAR(20),
                camera_id VARCHAR(50),
                message TEXT,
                details TEXT,
                image_data TEXT,
                session_id INTEGER,
                PRIMARY KEY (id, detected_time)
            )
            """
        ]
        
        # Create each table
        for i, table_sql in enumerate(table_definitions):
            try:
                conn.execute(text(table_sql))
                conn.commit()
                table_name = table_sql.split('CREATE TABLE ')[1].split(' (')[0]
                print(f"  ✓ Created {table_name}")
            except Exception as e:
                print(f"  ✗ Error creating table: {e}")
                conn.rollback()
        
        # Now create hypertables
        print("\nCreating hypertables...")
        hypertables = [
            ('detections', 'timestamp', '7 days'),
            ('session_detections', 'detection_time', '1 day'),
            ('session_alerts', 'alert_time', '1 day'),
            ('vehicle_anomalies', 'detected_time', '3 days')
        ]
        
        for table, time_col, interval in hypertables:
            try:
                conn.execute(text(f"""
                    SELECT create_hypertable('{table}', '{time_col}',
                        chunk_time_interval => INTERVAL '{interval}'
                    )
                """))
                conn.commit()
                print(f"  ✓ Created hypertable: {table}")
            except Exception as e:
                print(f"  ✗ Error creating hypertable {table}: {e}")
                conn.rollback()
        
        # Create indexes
        print("\nCreating indexes...")
        indexes = [
            "CREATE INDEX idx_detections_plate_time ON detections(plate_text, timestamp DESC)",
            "CREATE INDEX idx_detections_camera_time ON detections(camera_id, timestamp DESC)",
            "CREATE INDEX idx_detections_state ON detections(state) WHERE state IS NOT NULL",
            "CREATE INDEX idx_session_detections_session ON session_detections(session_id)",
            "CREATE INDEX idx_session_alerts_session ON session_alerts(session_id)",
            "CREATE INDEX idx_vehicle_tracks_suspicious ON vehicle_tracks(is_suspicious) WHERE is_suspicious = true",
            "CREATE INDEX idx_track_plate_assoc_track ON track_plate_associations(track_id)",
            "CREATE INDEX idx_track_plate_assoc_plate ON track_plate_associations(plate_text)",
            "CREATE INDEX idx_anomalies_track ON vehicle_anomalies(track_id)",
            "CREATE INDEX idx_anomalies_type ON vehicle_anomalies(anomaly_type, detected_time DESC)"
        ]
        
        for idx_sql in indexes:
            try:
                conn.execute(text(idx_sql))
                conn.commit()
                idx_name = idx_sql.split('INDEX ')[1].split(' ON')[0]
                print(f"  ✓ Created index: {idx_name}")
            except Exception as e:
                print(f"  ✗ Error creating index: {e}")
                conn.rollback()
        
        # Verify hypertables
        print("\n" + "="*50)
        print("Verifying setup...")
        
        result = conn.execute(text("""
            SELECT hypertable_schema, hypertable_name 
            FROM timescaledb_information.hypertables
            ORDER BY hypertable_name
        """))
        
        hypertables = result.fetchall()
        print(f"\nHypertables created ({len(hypertables)}):")
        for ht in hypertables:
            print(f"  - {ht.hypertable_name}")
        
        # Count all tables
        result = conn.execute(text("""
            SELECT COUNT(*) FROM information_schema.tables 
            WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
        """))
        table_count = result.scalar()
        print(f"\nTotal tables: {table_count}")
        
        print("\n✅ TimescaleDB setup complete!")

if __name__ == "__main__":
    confirm = input("This will DROP and RECREATE all tables. Continue? (yes/no): ")
    if confirm.lower() == 'yes':
        recreate_tables_for_timescale()
    else:
        print("Cancelled.")