# ALPR System Deployment Guide

## Native Linux/Windows Deployment for Production Surveillance

This guide covers deploying the ALPR system for production use in office surveillance and in-car monitoring scenarios.

## System Requirements

### Minimum Hardware
- **CPU**: Intel i5-8400 / AMD Ryzen 5 2600 or equivalent
- **RAM**: 8GB (16GB recommended for multiple cameras)
- **Storage**: 256GB SSD (1TB+ for video storage)
- **USB**: USB 3.0 ports for cameras
- **Network**: Ethernet port (for IP cameras)

### Recommended Hardware

#### Office Surveillance
- **Intel NUC** or **Mini PC** with Intel i7
- **16-32GB RAM** for multiple camera streams
- **2TB+ NVMe SSD** for video storage
- **Multiple USB 3.0/3.1 ports**
- **Gigabit Ethernet** for IP cameras

#### In-Car Monitoring
- **Raspberry Pi 4 (8GB)** or **NVIDIA Jetson Nano**
- **64-128GB high-endurance microSD**
- **USB 3.0 dashboard camera**
- **4G/LTE modem** for connectivity
- **12V power adapter** for vehicles

## Operating System Setup

### Ubuntu Server 22.04 LTS (Recommended)
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y python3.10 python3.10-venv python3.10-dev
sudo apt install -y build-essential cmake pkg-config
sudo apt install -y libopencv-dev python3-opencv
sudo apt install -y v4l-utils uvcdynctrl
sudo apt install -y postgresql-client libpq-dev
sudo apt install -y nginx supervisor
sudo apt install -y git curl wget

# Install camera drivers
sudo apt install -y linux-modules-extra-$(uname -r)
sudo apt install -y usb-modeswitch usb-modeswitch-data
```

### Windows Server 2019/2022 (Alternative)
```powershell
# Install Python 3.10+
winget install Python.Python.3.10

# Install Visual Studio Build Tools
winget install Microsoft.VisualStudio.2022.BuildTools

# Install Git
winget install Git.Git

# Install PostgreSQL
winget install PostgreSQL.PostgreSQL
```

## Installation

### 1. Clone and Setup Project
```bash
# Clone repository
git clone <your-repo-url> /opt/alpr-surveillance
cd /opt/alpr-surveillance

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install additional production dependencies
pip install gunicorn supervisor psycopg2-binary
```

### 2. Database Setup
```bash
# Install PostgreSQL
sudo apt install -y postgresql postgresql-contrib

# Create database and user
sudo -u postgres psql << EOF
CREATE USER alpr_user WITH PASSWORD 'secure_password_here';
CREATE DATABASE alpr_surveillance OWNER alpr_user;
GRANT ALL PRIVILEGES ON DATABASE alpr_surveillance TO alpr_user;
\q
EOF

# Update database configuration
cp api/database_config.py.example api/database_config.py
# Edit database_config.py with your credentials
```

### 3. Camera Setup

#### USB Cameras
```bash
# Check available cameras
ls /dev/video*
v4l2-ctl --list-devices

# Test camera access
v4l2-ctl --device=/dev/video0 --all
```

#### IP Cameras (RTSP)
```bash
# Test RTSP stream
ffplay rtsp://192.168.1.100:554/stream1

# Add camera configurations to config file
```

## Production Configuration

### Environment Variables
Create `/opt/alpr-surveillance/.env`:
```bash
# Database
DATABASE_URL=postgresql://alpr_user:secure_password_here@localhost/alpr_surveillance

# API Keys
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
```

### Create Storage Directories
```bash
sudo mkdir -p /var/lib/alpr-surveillance/{videos,images}
sudo mkdir -p /var/log/alpr-surveillance
sudo chown -R $USER:$USER /var/lib/alpr-surveillance
sudo chown -R $USER:$USER /var/log/alpr-surveillance
```

## Service Setup (Linux)

### Systemd Service
Create `/etc/systemd/system/alpr-surveillance.service`:
```ini
[Unit]
Description=ALPR Surveillance System
After=network.target postgresql.service
Requires=postgresql.service

[Service]
Type=exec
User=alpr
Group=alpr
WorkingDirectory=/opt/alpr-surveillance/api
Environment=PATH=/opt/alpr-surveillance/venv/bin
ExecStart=/opt/alpr-surveillance/venv/bin/gunicorn main:app --bind 0.0.0.0:8000 --workers 4 --worker-class uvicorn.workers.UvicornWorker
ExecReload=/bin/kill -s HUP $MAINPID
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

### Enable and Start Service
```bash
# Create service user
sudo useradd -r -s /bin/false alpr
sudo chown -R alpr:alpr /opt/alpr-surveillance

# Enable service
sudo systemctl daemon-reload
sudo systemctl enable alpr-surveillance
sudo systemctl start alpr-surveillance

# Check status
sudo systemctl status alpr-surveillance
```

## Nginx Reverse Proxy

Create `/etc/nginx/sites-available/alpr-surveillance`:
```nginx
server {
    listen 80;
    server_name surveillance.yourdomain.com;

    client_max_body_size 100M;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /ws/ {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 86400;
    }

    location /static/ {
        alias /opt/alpr-surveillance/api/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

Enable the site:
```bash
sudo ln -s /etc/nginx/sites-available/alpr-surveillance /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

## Monitoring and Logging

### Log Rotation
Create `/etc/logrotate.d/alpr-surveillance`:
```
/var/log/alpr-surveillance/*.log {
    daily
    missingok
    rotate 52
    compress
    delaycompress
    notifempty
    create 644 alpr alpr
    postrotate
        systemctl reload alpr-surveillance
    endscript
}
```

### System Monitoring
```bash
# Install monitoring tools
sudo apt install -y htop iotop nethogs

# Monitor service
sudo journalctl -f -u alpr-surveillance

# Monitor resources
htop
```

## Security Configuration

### Firewall Setup
```bash
# UFW firewall
sudo ufw allow 22/tcp   # SSH
sudo ufw allow 80/tcp   # HTTP
sudo ufw allow 443/tcp  # HTTPS
sudo ufw --force enable
```

### SSL Certificate (Production)
```bash
# Install Certbot
sudo apt install -y certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d surveillance.yourdomain.com
```

## Backup Strategy

### Database Backup
Create `/opt/alpr-surveillance/scripts/backup-db.sh`:
```bash
#!/bin/bash
BACKUP_DIR="/var/backups/alpr-surveillance"
DATE=$(date +%Y%m%d_%H%M%S)
mkdir -p $BACKUP_DIR

pg_dump -h localhost -U alpr_user alpr_surveillance | gzip > $BACKUP_DIR/db_backup_$DATE.sql.gz

# Keep only last 7 days
find $BACKUP_DIR -name "db_backup_*.sql.gz" -mtime +7 -delete
```

### Video/Image Cleanup
Create `/opt/alpr-surveillance/scripts/cleanup-storage.sh`:
```bash
#!/bin/bash
STORAGE_PATH="/var/lib/alpr-surveillance"
DAYS_TO_KEEP=30

# Clean old videos
find $STORAGE_PATH/videos -name "*.mp4" -mtime +$DAYS_TO_KEEP -delete

# Clean old images
find $STORAGE_PATH/images -name "*.jpg" -mtime +$DAYS_TO_KEEP -delete

# Clean database records older than retention period
psql -h localhost -U alpr_user alpr_surveillance -c "DELETE FROM plate_detections WHERE timestamp < NOW() - INTERVAL '$DAYS_TO_KEEP days';"
```

### Cron Jobs
```bash
# Add to crontab
sudo crontab -e

# Add these lines:
0 2 * * * /opt/alpr-surveillance/scripts/backup-db.sh
0 3 * * * /opt/alpr-surveillance/scripts/cleanup-storage.sh
```

## Performance Tuning

### Camera Settings
```python
# In camera_detector.py
PRODUCTION_CAMERA_SETTINGS = {
    'resolution': (1920, 1080),
    'fps': 15,
    'buffer_size': 1,
    'codec': 'MJPG',
    'auto_exposure': False,
    'brightness': 50,
    'contrast': 50
}
```

### Resource Limits
```bash
# Increase file limits for high-throughput
echo "alpr soft nofile 65536" >> /etc/security/limits.conf
echo "alpr hard nofile 65536" >> /etc/security/limits.conf
```

## Troubleshooting

### Common Issues

#### Camera Not Detected
```bash
# Check USB devices
lsusb | grep -i camera

# Check video devices
ls -la /dev/video*

# Check permissions
sudo usermod -a -G video alpr
```

#### Performance Issues
```bash
# Monitor system resources
htop
iotop
nvidia-smi  # If using GPU

# Check service logs
sudo journalctl -f -u alpr-surveillance
```

#### Database Connection Issues
```bash
# Test database connection
sudo -u alpr psql -h localhost -U alpr_user alpr_surveillance

# Check PostgreSQL status
sudo systemctl status postgresql
```

## Deployment Checklist

- [ ] Operating system installed and updated
- [ ] Dependencies installed
- [ ] Project cloned and configured
- [ ] Database created and configured
- [ ] Cameras tested and working
- [ ] Environment variables set
- [ ] Systemd service created and enabled
- [ ] Nginx configured (if needed)
- [ ] Firewall configured
- [ ] SSL certificate installed (production)
- [ ] Backup scripts configured
- [ ] Monitoring set up
- [ ] Performance tuned
- [ ] Documentation updated

## Next Steps

After completing this deployment:

1. **Configure cameras** for your specific environment
2. **Set up monitoring** and alerting
3. **Test failover scenarios**
4. **Configure remote access** (VPN recommended)
5. **Train users** on the system
6. **Plan maintenance schedule**

For in-car deployment, consider additional steps:
- Power management for ignition on/off
- Cellular connectivity setup
- GPS integration
- Temperature monitoring
- Vibration-resistant mounting