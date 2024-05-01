
# This is client code to receive video frames over UDP
import cv2, socket
import numpy as np
import time
import base64
import cv2
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image, ImageDraw
import numpy as np
import json

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

# Function to draw bounding boxes and labels on the image
def draw_boxes(image, prediction):
	draw = ImageDraw.Draw(image)
	for score, label, box in zip(prediction[0]['scores'], prediction[0]['labels'], prediction[0]['boxes']):
		if score > 0.5:  # Confidence threshold
			box = [round(i.item()) for i in box]
			draw.rectangle(box, outline='red', width=3)
			draw.text((box[0], box[1]), COCO_INSTANCE_CATEGORY_NAMES[label.item()], fill='red')
	return image

# set server info for client socket
BUFF_SIZE = 65536
client_socket = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
client_socket.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,BUFF_SIZE)
client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)

host_name = socket.gethostname()
host_ip = '192.168.0.142'#  socket.gethostbyname(host_name)
print(host_ip)
port = 9999
message = b'REQUEST from client'

client_socket.sendto(message,(host_ip,port))
fps,st,frames_to_count,cnt = (0,0,20,0)

while True:
	packet,_ = client_socket.recvfrom(BUFF_SIZE)
	data = base64.b64decode(packet,' /')
	npdata = np.fromstring(data,dtype=np.uint8)
	frame = cv2.imdecode(npdata,1)

	# Inferencing, record the time for referencing
	start_time = time.time()
	pil_image = Image.fromarray(frame)
	prediction = detect_objects(pil_image)
	end_time = time.time()
	print(end_time - start_time)
	
	# send prediction back to server
	prediction_json = json.dumps(prediction)
	client_socket.sendto(prediction_json.encode(), (host_ip,port))
	  
	frame = cv2.putText(frame,'FPS: '+str(fps),(10,40),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
	cv2.imshow('Object Detection', frame)
		
	key = cv2.waitKey(1) & 0xFF


