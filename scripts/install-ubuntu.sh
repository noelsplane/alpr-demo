#!/bin/bash
# ALPR Surveillance System - Ubuntu Installation Script
# For Ubuntu 20.04+ / Debian 11+

set -e

echo "🚀 ALPR Surveillance System Installation"
echo "========================================"

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo "❌ This script should not be run as root (use your regular user account)"
   exit 1
fi

# Variables
INSTALL_DIR="/opt/alpr-surveillance"
SERVICE_USER="alpr"
DB_NAME="alpr_surveillance"
DB_USER="alpr_user"

echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

echo "🔧 Installing system dependencies..."
sudo apt install -y \
    python3.10 python3.10-venv python3.10-dev \
    build-essential cmake pkg-config \
    libopencv-dev python3-opencv \
    v4l-utils uvcdynctrl \
    postgresql postgresql-contrib libpq-dev \
    nginx supervisor \
    git curl wget unzip \
    linux-modules-extra-$(uname -r) \
    usb-modeswitch usb-modeswitch-data \
    htop iotop nethogs \
    logrotate \
    ufw

echo "📹 Setting up camera support..."
# Add user to video group for camera access
sudo usermod -a -G video $USER

echo "🗄️  Setting up PostgreSQL database..."
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Generate secure password
DB_PASSWORD=$(openssl rand -base64 32)

# Create database and user
sudo -u postgres psql << EOF
CREATE USER $DB_USER WITH PASSWORD '$DB_PASSWORD';
CREATE DATABASE $DB_NAME OWNER $DB_USER;
GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;
ALTER USER $DB_USER CREATEDB;
\q
EOF

echo "✅ Database created with user: $DB_USER"
echo "🔑 Database password: $DB_PASSWORD"
echo "💾 Save this password! You'll need it for configuration."

echo "👤 Creating service user..."
sudo useradd -r -m -s /bin/bash $SERVICE_USER || true
sudo usermod -a -G video $SERVICE_USER

echo "📂 Setting up application directory..."
sudo mkdir -p $INSTALL_DIR
sudo chown $SERVICE_USER:$SERVICE_USER $INSTALL_DIR

# Create required directories
sudo mkdir -p /var/lib/alpr-surveillance/{videos,images}
sudo mkdir -p /var/log/alpr-surveillance
sudo mkdir -p /var/backups/alpr-surveillance

sudo chown -R $SERVICE_USER:$SERVICE_USER /var/lib/alpr-surveillance
sudo chown -R $SERVICE_USER:$SERVICE_USER /var/log/alpr-surveillance
sudo chown -R $SERVICE_USER:$SERVICE_USER /var/backups/alpr-surveillance

echo "🐍 Setting up Python environment..."
sudo -u $SERVICE_USER bash << EOF
cd $INSTALL_DIR
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# Install Python packages
pip install fastapi uvicorn gunicorn
pip install opencv-python opencv-contrib-python
pip install easyocr ultralytics
pip install sqlalchemy psycopg2-binary alembic
pip install pillow numpy scipy
pip install python-multipart python-dotenv
pip install websockets
pip install requests
pip install difflib pathlib
EOF

echo "🔧 Creating environment configuration..."
sudo -u $SERVICE_USER tee $INSTALL_DIR/.env > /dev/null << EOF
# Database Configuration
DATABASE_URL=postgresql://$DB_USER:$DB_PASSWORD@localhost/$DB_NAME

# API Keys (add your actual tokens)
PLATERECOGNIZER_TOKEN=your_token_here

# Logging
LOG_LEVEL=INFO
LOG_FILE=/var/log/alpr-surveillance/app.log

# Camera Settings
DEFAULT_CAMERA_RESOLUTION=1920x1080
MAX_CAMERAS=8
FRAME_RATE=15

# Storage
VIDEO_STORAGE_PATH=/var/lib/alpr-surveillance/videos
IMAGE_STORAGE_PATH=/var/lib/alpr-surveillance/images
MAX_STORAGE_DAYS=30

# Network
HOST=0.0.0.0
PORT=8000
WORKERS=4
EOF

echo "🔧 Creating systemd service..."
sudo tee /etc/systemd/system/alpr-surveillance.service > /dev/null << EOF
[Unit]
Description=ALPR Surveillance System
After=network.target postgresql.service
Requires=postgresql.service

[Service]
Type=exec
User=$SERVICE_USER
Group=$SERVICE_USER
WorkingDirectory=$INSTALL_DIR/api
Environment=PATH=$INSTALL_DIR/venv/bin
ExecStart=$INSTALL_DIR/venv/bin/gunicorn main:app --bind 0.0.0.0:8000 --workers 4 --worker-class uvicorn.workers.UvicornWorker --timeout 300
ExecReload=/bin/kill -s HUP \$MAINPID
Restart=always
RestartSec=3
KillMode=mixed
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
EOF

echo "🌐 Configuring Nginx..."
sudo tee /etc/nginx/sites-available/alpr-surveillance > /dev/null << EOF
server {
    listen 80;
    server_name localhost;

    client_max_body_size 100M;
    client_body_timeout 300s;
    proxy_read_timeout 300s;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_buffering off;
    }

    location /ws/ {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_read_timeout 86400;
    }

    location /static/ {
        alias $INSTALL_DIR/api/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
EOF

# Enable Nginx site
sudo ln -sf /etc/nginx/sites-available/alpr-surveillance /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl enable nginx

echo "📋 Setting up log rotation..."
sudo tee /etc/logrotate.d/alpr-surveillance > /dev/null << EOF
/var/log/alpr-surveillance/*.log {
    daily
    missingok
    rotate 52
    compress
    delaycompress
    notifempty
    create 644 $SERVICE_USER $SERVICE_USER
    postrotate
        systemctl reload alpr-surveillance
    endscript
}
EOF

echo "🛡️ Configuring firewall..."
sudo ufw --force enable
sudo ufw allow 22/tcp   # SSH
sudo ufw allow 80/tcp   # HTTP
sudo ufw allow 443/tcp  # HTTPS

echo "📝 Creating backup scripts..."
sudo -u $SERVICE_USER tee $INSTALL_DIR/scripts/backup-db.sh > /dev/null << 'EOF'
#!/bin/bash
BACKUP_DIR="/var/backups/alpr-surveillance"
DATE=$(date +%Y%m%d_%H%M%S)
mkdir -p $BACKUP_DIR

pg_dump -h localhost -U alpr_user alpr_surveillance | gzip > $BACKUP_DIR/db_backup_$DATE.sql.gz

# Keep only last 7 days
find $BACKUP_DIR -name "db_backup_*.sql.gz" -mtime +7 -delete
EOF

sudo -u $SERVICE_USER tee $INSTALL_DIR/scripts/cleanup-storage.sh > /dev/null << 'EOF'
#!/bin/bash
STORAGE_PATH="/var/lib/alpr-surveillance"
DAYS_TO_KEEP=30

# Clean old videos
find $STORAGE_PATH/videos -name "*.mp4" -mtime +$DAYS_TO_KEEP -delete 2>/dev/null || true

# Clean old images
find $STORAGE_PATH/images -name "*.jpg" -mtime +$DAYS_TO_KEEP -delete 2>/dev/null || true

# Clean database records older than retention period
PGPASSWORD='$DB_PASSWORD' psql -h localhost -U alpr_user alpr_surveillance -c "DELETE FROM plate_detections WHERE timestamp < NOW() - INTERVAL '$DAYS_TO_KEEP days';" 2>/dev/null || true
EOF

sudo chmod +x $INSTALL_DIR/scripts/*.sh

echo "⏰ Setting up cron jobs..."
(sudo crontab -u $SERVICE_USER -l 2>/dev/null; echo "0 2 * * * $INSTALL_DIR/scripts/backup-db.sh") | sudo crontab -u $SERVICE_USER -
(sudo crontab -u $SERVICE_USER -l 2>/dev/null; echo "0 3 * * * $INSTALL_DIR/scripts/cleanup-storage.sh") | sudo crontab -u $SERVICE_USER -

echo "🔧 Optimizing system for camera operations..."
# Increase file limits
echo "$SERVICE_USER soft nofile 65536" | sudo tee -a /etc/security/limits.conf
echo "$SERVICE_USER hard nofile 65536" | sudo tee -a /etc/security/limits.conf

# USB device permissions
sudo tee /etc/udev/rules.d/99-alpr-cameras.rules > /dev/null << EOF
# USB cameras for ALPR system
SUBSYSTEM=="video4linux", GROUP="video", MODE="0664"
SUBSYSTEM=="usb", ATTRS{idVendor}=="*", ATTRS{idProduct}=="*", GROUP="video", MODE="0664"
EOF

sudo udevadm control --reload-rules

echo "📦 Downloading YOLO models..."
sudo -u $SERVICE_USER bash << EOF
cd $INSTALL_DIR
mkdir -p models
cd models

# Download YOLOv8 models
python3 -c "
from ultralytics import YOLO
import os

# Download base model
model = YOLO('yolov8n.pt')
print('Downloaded YOLOv8n model')

# Create model directory
os.makedirs('license_plate', exist_ok=True)
"
EOF

echo "✅ Installation completed successfully!"
echo ""
echo "📋 Next Steps:"
echo "=============="
echo ""
echo "1. Copy your project files to: $INSTALL_DIR"
echo "   sudo cp -r /path/to/your/alpr-demo/* $INSTALL_DIR/"
echo "   sudo chown -R $SERVICE_USER:$SERVICE_USER $INSTALL_DIR"
echo ""
echo "2. Update the .env file with your API keys:"
echo "   sudo nano $INSTALL_DIR/.env"
echo ""
echo "3. Test the database connection:"
echo "   PGPASSWORD='$DB_PASSWORD' psql -h localhost -U $DB_USER $DB_NAME"
echo ""
echo "4. Start the services:"
echo "   sudo systemctl daemon-reload"
echo "   sudo systemctl start alpr-surveillance"
echo "   sudo systemctl start nginx"
echo "   sudo systemctl enable alpr-surveillance"
echo ""
echo "5. Check service status:"
echo "   sudo systemctl status alpr-surveillance"
echo "   sudo journalctl -f -u alpr-surveillance"
echo ""
echo "6. Test camera access:"
echo "   ls /dev/video*"
echo "   v4l2-ctl --list-devices"
echo ""
echo "7. Access the web interface:"
echo "   http://localhost (or your server IP)"
echo ""
echo "🔑 Database Credentials:"
echo "   Database: $DB_NAME"
echo "   Username: $DB_USER" 
echo "   Password: $DB_PASSWORD"
echo ""
echo "⚠️  IMPORTANT: Save the database password above!"
echo ""
echo "🎯 For production deployment, also consider:"
echo "   - SSL certificate (Let's Encrypt)"
echo "   - Domain name configuration"
echo "   - Backup strategy testing"
echo "   - Monitoring setup"
echo "   - Security hardening"
echo ""
echo "📖 See DEPLOYMENT.md for detailed configuration options."