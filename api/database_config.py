"""
Database configuration for TimescaleDB
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)

class DatabaseConfig:
    """Centralized database configuration for TimescaleDB"""
    
    def __init__(self):
        self.connection_string = self._get_connection_string()
        self.engine = None
        self.SessionLocal = None
        
    def _get_connection_string(self):
        """Get PostgreSQL/TimescaleDB connection string"""
        # PostgreSQL/TimescaleDB connection
        db_user = os.getenv('DB_USER', 'alpr_user')
        db_password = os.getenv('DB_PASSWORD', 'alpr_password')
        db_host = os.getenv('DB_HOST', 'localhost')
        db_port = os.getenv('DB_PORT', '5432')
        db_name = os.getenv('DB_NAME', 'alpr_surveillance')
        
        return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    
    def init_engine(self):
        """Initialize TimescaleDB engine with optimized settings"""
        # PostgreSQL/TimescaleDB optimized settings
        self.engine = create_engine(
            self.connection_string,
            pool_size=20,
            max_overflow=30,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=False,
            poolclass=QueuePool,
            connect_args={
                "application_name": "alpr_surveillance",
                "options": "-c timezone=utc"
            }
        )
        
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        logger.info(f"TimescaleDB engine initialized: {self.connection_string.split('@')[1]}")
        
        return self.engine
    
    def get_session(self):
        """Get a new database session"""
        if not self.SessionLocal:
            self.init_engine()
        return self.SessionLocal()
    
    def test_connection(self):
        """Test the database connection"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute("SELECT 1")
                logger.info("Database connection successful")
                return True
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False

# Global instance
db_config = DatabaseConfig()