
# This is server code to send video frames over UDP
import cv2, imutils, socket
import numpy as np
import time
import base64
import pickle
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image, ImageDraw

# initialize parameters
BROADCASTED = False    # Used for the initial handshake procedure
BUFF_SIZE = 65536
EDGE_HOST_IP = '192.168.0.142'
PORT = 8080
WIDTH=400       # Width of the frame to be sent
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

# read in camera data at framerate of 30
def gstreamer_pipeline(
	capture_width=1920,
	capture_height=1080,
	display_width=960,
	display_height=540,
	framerate=30,
	flip_method=0,
):
	return (
		"nvarguscamerasrc ! "
		"video/x-raw(memory:NVMM), "
		"width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
		"nvvidconv flip-method=%d ! "
		"video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
		"videoconvert ! "
		"video/x-raw, format=(string)BGR ! appsink drop=True"
		% (
			capture_width,
			capture_height,
			framerate,
			flip_method,
			display_width,
			display_height,
		)
	)

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


# >>>>>>>>>>>>>>>>>> BRODACASTING & TRANSMITTING >>>>>>>>>>>>>>>>>>>>>>
# setting up onboard_host info
edge = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
# define the socket buffer size; buffer size should be big enough
edge.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,BUFF_SIZE)
# enable reuse of port 
edge.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, True)

# setting up transmitting port IP --> ONBOARD
socket_address = (EDGE_HOST_IP,PORT)
edge.bind(socket_address)
print('Begin listening at:',socket_address)

# # BROADCASTING onboard server IP toward offboard
# edge_info = (EDGE_HOST_IP + " " + str(PORT)).encode()    # BROADCAST MESSAGE onboard host IP to be broadcasted, parse in offboard host
# edge.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
# edge.sendto(edge_info, ('<broadcast>', PORT))
# BROADCASTED = True
# print("server IP information sent to client")
# msg,EDGE_SERVER_IP = edge.recvfrom(BUFF_SIZE)     # ------> BLOCKING STATE, confirming the broadcast of host IP
# edge.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 0)    # disable broadcasting mode after sent

# TRANSMITTING INITIALIZED
# setting up connection between onboard and offboard
msg,EDGE_SERVER_IP = edge.recvfrom(BUFF_SIZE)    # ------> BLOCKING STATE wait for connection
print('GOT connection from ',EDGE_SERVER_IP)

# CAPTURING VIDEO
vid = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER) 
fps,st,frames_to_count,cnt = (0,0,20,0)

# variable indicating if it should offload or not, ALWAYS ASSUME NO OFFLOADING
off_loading = True

edge.settimeout(0.1)
while(vid.isOpened()):
	# TODO: 
	# write a code to decide if the offloading should happen or not
	# testing of the connection by send a few frame of data toward client

	# TODO: TIME PROFILING of RTT of a single frame
	
	# TODO: CPU & RAM PROFILING of edge
	
	# TODO: CPU & RAM PROFILING of server

	_, frame = vid.read()

	# computing onboard if not offloaded
	if (not off_loading):
		start_time = time.time()
		pil_image = Image.fromarray(frame)
		prediction = detect_objects(pil_image)
		end_time = time.time()
		print(end_time - start_time)
	# OFFLOADING PROCESS
	else:
		# sending frame to server
		start_time = time.time()
		frame = imutils.resize(frame,width=WIDTH)
		encoded, buffer = cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,80])
		ec_img = base64.b64encode(buffer)
		edge.sendto(ec_img, EDGE_SERVER_IP)
		# waiting for the object detection result sended back from server
		try:
			prediction_result, _ = edge.recvfrom(BUFF_SIZE)
			prediction = pickle.loads(prediction_result)
			end_time = time.time()
			print("OFFLOADING TIME: ", end_time - start_time)
		except socket.timeout:
			print("prediction recieved time out")
			continue
		
		
		
		
		

