"""
Database configuration with support for SQLite and TimescaleDB
"""

import os
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)

class DatabaseConfig:
    """Centralized database configuration"""
    
    def __init__(self):
        self.db_type = os.getenv('DB_TYPE', 'sqlite').lower()
        self.connection_string = self._get_connection_string()
        self.engine = None
        self.SessionLocal = None
        
    def _get_connection_string(self):
        """Get appropriate connection string based on DB_TYPE"""
        if self.db_type == 'timescale' or self.db_type == 'postgresql':
            # PostgreSQL/TimescaleDB connection
            db_user = os.getenv('DB_USER', 'alpr_user')
            db_password = os.getenv('DB_PASSWORD', 'alpr_password')
            db_host = os.getenv('DB_HOST', 'localhost')
            db_port = os.getenv('DB_PORT', '5432')
            db_name = os.getenv('DB_NAME', 'alpr_surveillance')
            
            return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        else:
            # SQLite connection (default)
            db_path = os.getenv('SQLITE_PATH', 'detections.db')
            return f"sqlite:///{db_path}"
    
    def init_engine(self):
        """Initialize database engine with appropriate settings"""
        if self.db_type in ['timescale', 'postgresql']:
            # PostgreSQL/TimescaleDB settings
            self.engine = create_engine(
                self.connection_string,
                pool_size=20,
                max_overflow=30,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=False
            )
        else:
            # SQLite settings
            self.engine = create_engine(
                self.connection_string,
                connect_args={'check_same_thread': False},
                poolclass=NullPool
            )
            
            # Enable foreign keys for SQLite
            @event.listens_for(self.engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()
        
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        logger.info(f"Database engine initialized: {self.db_type}")
        
        return self.engine
    
    def get_session(self):
        """Get a new database session"""
        if not self.SessionLocal:
            self.init_engine()
        return self.SessionLocal()
    
    def is_timescale(self):
        """Check if using TimescaleDB"""
        return self.db_type == 'timescale'

# Global instance
db_config = DatabaseConfig()