from roboflow import Roboflow

# Initialize Roboflow (you can use "dummy" for public models)
rf = Roboflow(api_key="PKGMsAJhODMwliPx2wZU")

# Access the license plate detection project
project = rf.workspace("roboflow-universe-projects").project("license-plate-recognition-rxg4e")
dataset = project.version(4).download("yolov8")

print(f"Dataset downloaded to: {dataset.location}")
print("Now we need to train or use the pre-trained weights if available")
