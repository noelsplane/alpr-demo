from roboflow import Roboflow
import os
import shutil

# Initialize Roboflow
rf = Roboflow(api_key="PKGMsAJhODMwliPx2wZU")

# Download license plate datasets
datasets = [
    ("license-plate-recognition-rxg4e", 4),  # 24,000+ images
    ("license-plates-f8vsn", 1),            # 5,000+ images
    ("us-license-plates-qqvyz", 1),        # US specific plates
]

for project_name, version in datasets:
    try:
        project = rf.workspace("roboflow-universe-projects").project(project_name)
        dataset = project.version(version).download("yolov8")
        print(f"Downloaded {project_name} to {dataset.location}")
    except Exception as e:
        print(f"Error downloading {project_name}: {e}")
