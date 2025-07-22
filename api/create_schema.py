#!/usr/bin/env python3
"""
Create ALPR database schema with TimescaleDB hypertables
"""

import os
import sys
from dotenv import load_dotenv
from sqlalchemy import text

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database_config import db_config
from models import Base

def create_schema():
    """Create all tables and configure TimescaleDB."""
    
    print("Creating ALPR database schema...")
    
    try:
        # Initialize engine
        engine = db_config.init_engine()
        
        # Drop existing tables if needed (be careful with this in production!)
        response = input("\nDrop existing tables? (y/N): ").lower()
        if response == 'y':
            print("Dropping existing tables...")
            Base.metadata.drop_all(engine)
            print("✓ Tables dropped")
        
        # Create all tables
        print("\nCreating tables...")
        Base.metadata.create_all(engine)
        print("✓ Tables created successfully")
        
        # Create hypertables and add TimescaleDB optimizations
        with engine.connect() as conn:
            print("\nConfiguring TimescaleDB...")
            
            # Create hypertables
            hypertables = [
                {
                    'table': 'detections',
                    'time_column': 'timestamp',
                    'chunk_interval': '7 days',
                    'compress_after': '30 days'
                },
                {
                    'table': 'session_detections',
                    'time_column': 'detection_time',
                    'chunk_interval': '1 day',
                    'compress_after': '7 days'
                },
                {
                    'table': 'session_alerts',
                    'time_column': 'alert_time',
                    'chunk_interval': '1 day',
                    'compress_after': '7 days'
                },
                {
                    'table': 'vehicle_anomalies',
                    'time_column': 'detected_time',
                    'chunk_interval': '3 days',
                    'compress_after': '14 days'
                }
            ]
            
            for ht in hypertables:
                try:
                    # Create hypertable
                    result = conn.execute(text(f"""
                        SELECT create_hypertable(
                            '{ht['table']}',
                            '{ht['time_column']}',
                            chunk_time_interval => INTERVAL '{ht['chunk_interval']}',
                            if_not_exists => TRUE
                        )
                    """))
                    conn.commit()
                    print(f"  ✓ Created hypertable: {ht['table']}")
                    
                    # Add compression policy
                    conn.execute(text(f"""
                        ALTER TABLE {ht['table']} SET (
                            timescaledb.compress,
                            timescaledb.compress_orderby = '{ht['time_column']} DESC'
                        )
                    """))
                    
                    conn.execute(text(f"""
                        SELECT add_compression_policy('{ht['table']}',
                            INTERVAL '{ht['compress_after']}',
                            if_not_exists => TRUE)
                    """))
                    conn.commit()
                    print(f"    ✓ Added compression policy: compress after {ht['compress_after']}")
                    
                except Exception as e:
                    if "already a hypertable" in str(e):
                        print(f"  ℹ {ht['table']} is already a hypertable")
                    else:
                        print(f"  ⚠ Warning for {ht['table']}: {e}")
            
            # Create additional indexes for performance
            print("\nCreating performance indexes...")
            indexes = [
                # For plate searches across time
                "CREATE INDEX IF NOT EXISTS idx_plate_time ON detections(plate_text, timestamp DESC) WHERE plate_text IS NOT NULL",
                
                # For camera-based queries
                "CREATE INDEX IF NOT EXISTS idx_camera_time ON detections(camera_id, timestamp DESC) WHERE camera_id IS NOT NULL",
                
                # For vehicle attribute searches
                "CREATE INDEX IF NOT EXISTS idx_vehicle_make_model ON detections(vehicle_make, vehicle_model) WHERE vehicle_make IS NOT NULL",
                
                # For state analysis
                "CREATE INDEX IF NOT EXISTS idx_state_conf ON detections(state, state_confidence DESC) WHERE state IS NOT NULL",
                
                # For anomaly tracking
                "CREATE INDEX IF NOT EXISTS idx_anomaly_severity_time ON vehicle_anomalies(severity, detected_time DESC)",
                
                # For active sessions
                "CREATE INDEX IF NOT EXISTS idx_session_status ON surveillance_sessions(status, start_time DESC)",
                
                # For vehicle tracking
                "CREATE INDEX IF NOT EXISTS idx_track_suspicious ON vehicle_tracks(is_suspicious, last_seen DESC) WHERE is_suspicious = true",
                
                # For plate associations
                "CREATE INDEX IF NOT EXISTS idx_plate_assoc ON track_plate_associations(plate_text, last_seen DESC)"
            ]
            
            for idx_sql in indexes:
                try:
                    conn.execute(text(idx_sql))
                    conn.commit()
                    # Extract index name for display
                    idx_name = idx_sql.split("INDEX IF NOT EXISTS ")[1].split(" ON")[0]
                    print(f"  ✓ Created index: {idx_name}")
                except Exception as e:
                    print(f"  ⚠ Index creation warning: {e}")
            
            # Create continuous aggregates for analytics
            print("\nCreating continuous aggregates...")
            aggregates = [
                {
                    'name': 'hourly_stats',
                    'query': """
                        CREATE MATERIALIZED VIEW IF NOT EXISTS hourly_stats
                        WITH (timescaledb.continuous) AS
                        SELECT 
                            time_bucket('1 hour', timestamp) AS hour,
                            COUNT(*) as detection_count,
                            COUNT(DISTINCT plate_text) as unique_plates,
                            COUNT(DISTINCT state) as unique_states,
                            AVG(confidence)::numeric(4,2) as avg_confidence
                        FROM detections
                        GROUP BY hour
                        WITH NO DATA
                    """
                },
                {
                    'name': 'daily_vehicle_stats',
                    'query': """
                        CREATE MATERIALIZED VIEW IF NOT EXISTS daily_vehicle_stats
                        WITH (timescaledb.continuous) AS
                        SELECT 
                            time_bucket('1 day', timestamp) AS day,
                            plate_text,
                            COUNT(*) as sighting_count,
                            MIN(timestamp) as first_seen,
                            MAX(timestamp) as last_seen
                        FROM detections
                        WHERE plate_text IS NOT NULL
                        GROUP BY day, plate_text
                        WITH NO DATA
                    """
                }
            ]
            
            for agg in aggregates:
                try:
                    conn.execute(text(agg['query']))
                    conn.commit()
                    print(f"  ✓ Created aggregate: {agg['name']}")
                except Exception as e:
                    if "already exists" in str(e):
                        print(f"  ℹ {agg['name']} already exists")
                    else:
                        print(f"  ⚠ Warning for {agg['name']}: {e}")
        
        print("\n✅ Database schema creation complete!")
        
        # Show summary
        with engine.connect() as conn:
            # Count tables
            result = conn.execute(text("""
                SELECT COUNT(*) FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
            """))
            table_count = result.scalar()
            
            # Count hypertables
            result = conn.execute(text("""
                SELECT COUNT(*) FROM timescaledb_information.hypertables
            """))
            hypertable_count = result.scalar()
            
            # Count indexes
            result = conn.execute(text("""
                SELECT COUNT(*) FROM pg_indexes 
                WHERE schemaname = 'public'
            """))
            index_count = result.scalar()
            
            print(f"\nDatabase Summary:")
            print(f"  Tables: {table_count}")
            print(f"  Hypertables: {hypertable_count}")
            print(f"  Indexes: {index_count}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    print("ALPR TimescaleDB Schema Setup")
    print("=" * 40)
    
    if not os.path.exists('.env'):
        print("Error: .env file not found")
        sys.exit(1)
    
    if create_schema():
        print("\n✅ Setup complete! Your database is ready.")
        print("\nNext steps:")
        print("1. If you have SQLite data to migrate, we can do that next")
        print("2. Otherwise, start the API server: uvicorn main:app --reload")
    else:
        sys.exit(1)