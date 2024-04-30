import cv2
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image, ImageDraw
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pre-trained model
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
model.to(DEVICE)

# Define the labels for the COCO dataset
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Function to perform object detection on an image
def detect_objects(image):
    # Convert image to tensor
    transform = transforms.Compose([transforms.ToTensor()])
    image = transform(image).to(DEVICE)
    # Perform inference
    with torch.no_grad():
        prediction = model([image])
    return prediction

# Function to read video frames, perform object detection, and display the result
def detect_objects_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB
        pil_image = Image.fromarray(frame)
        prediction = detect_objects(pil_image)
    cap.release()
    cv2.destroyAllWindows()