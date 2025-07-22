#!/usr/bin/env python3
"""
Test database connection and create tables
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Import database config and models
from database_config import db_config
from models import Base
from sqlalchemy import text

def test_connection():
    """Test database connection and create tables."""
    
    print("Testing database connection...")
    
    try:
        # Initialize engine
        engine = db_config.init_engine()
        
        # Test connection
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print("✓ Database connection successful")
            
            # Check TimescaleDB
            result = conn.execute(text("SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'"))
            ts_version = result.fetchone()
            if ts_version:
                print(f"✓ TimescaleDB version: {ts_version[0]}")
            else:
                print("✗ TimescaleDB not found")
                return False
        
        # Create all tables
        print("\nCreating tables...")
        Base.metadata.create_all(engine)
        print("✓ Tables created successfully")
        
        # Create hypertables
        print("\nCreating hypertables...")
        with engine.connect() as conn:
            # Create hypertables for time-series data
            hypertables = [
                ('detections', 'timestamp'),
                ('session_detections', 'detection_time'),
                ('session_alerts', 'alert_time'),
                ('vehicle_anomalies', 'detected_time')
            ]
            
            for table, time_column in hypertables:
                try:
                    conn.execute(text(f"""
                        SELECT create_hypertable('{table}', '{time_column}',
                            chunk_time_interval => INTERVAL '7 days',
                            if_not_exists => TRUE)
                    """))
                    conn.commit()
                    print(f"  ✓ Created hypertable: {table}")
                except Exception as e:
                    print(f"  ! Note for {table}: {e}")
        
        print("\n✅ Database setup complete!")
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check your .env file has the correct credentials")
        print("2. Ensure PostgreSQL is running: sudo systemctl status postgresql")
        print("3. Verify the database exists: sudo -u postgres psql -l")
        return False


if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    print("ALPR TimescaleDB Setup Test")
    print("=" * 40)
    
    if not os.path.exists('.env'):
        print("Error: .env file not found")
        print("Please create .env file in the api directory")
        sys.exit(1)
    
    # Check if password is set
    if not os.getenv('DB_PASSWORD'):
        print("Error: DB_PASSWORD not set in .env file")
        sys.exit(1)
    
    if test_connection():
        print("\nNext steps:")
        print("1. If you have existing SQLite data, run the migration script")
        print("2. Start the API server: uvicorn main:app --reload")
    else:
        sys.exit(1)