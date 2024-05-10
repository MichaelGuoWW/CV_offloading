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
import pickle
import psutil
import subprocess
import crcmod

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pre-trained model
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
model.to(DEVICE)

# Function to perform object detection on an image
def detect_objects(image):
	# Convert image to tensor
	transform = transforms.Compose([transforms.ToTensor()])
	image = transform(image).to(DEVICE)
	# Perform inference
	with torch.no_grad():
		prediction = model([image])
	return prediction


class server:
	def __init__(self) -> None:
		self.BUFF_SIZE = 65536
		self.PROFILING_PORT_IN = 9998      # EDGE_PROFILING_OUT ---> SERVER_PROFILING_IN
		self.PROFILING_PORT_OUT = 9999       # EDGE_PROFILING_IN ---> SERVER_PROFILING_OUT
		self.DATA_PORT = 8080
		self.INFERENCE_PORT = 8081
		self.WIDTH = 400       # Width of the frame to be sent
		self.count = 0
		self.edge_host_ip = ''

		# Initialization of transmission port
		# setup DATA_PORT
		self.data_rx = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
		# define the socket buffer size; buffer size should be big enough
		self.data_rx.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF, self.BUFF_SIZE)
		# enable reuse of port 
		self.data_rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, True)
	
	# recieving profiling information from server's broadcast
	def profiler(self):
		# setting up profiler in 
		profiling_in = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
		# define the socket buffer size; buffer size should be big enough
		profiling_in.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.BUFF_SIZE)
		# enable reuse of port 
		profiling_in.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, True)
		# enable broadcasting mode
		profiling_in.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

		# set up profiling in address
		socket_address = ("", self.PROFILING_PORT_IN)
		profiling_in.bind(socket_address)

		# setting up profiler out
		profiling_out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
		# define the socket buffer size; buffer size should be big enough
		profiling_out.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.BUFF_SIZE)
		# enable reuse of port 
		profiling_out.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, True)

		while True:
			data, address = profiling_in.recvfrom(3096*2)
			# bind the socket to edge for future data transmit
			self.data_rx.bind((address[0], self.DATA_PORT))
			self.edge_host_ip = address[0]
			# decode the data
			edge = pickle.loads(data)
			send_time = edge[0]
			cpu_info = psutil.cpu_percent(interval=0.5)
			gpu_info = self.get_gpu_usage()
			self.count = self.count + 1

			msg = (cpu_info, gpu_info, send_time, self.count)
			encode_msg = pickle.dumps(msg)
			profiling_out.sendto(encode_msg, (address[0], self.PROFILING_PORT_OUT))
			print(address)
			print(send_time)
		
	def get_gpu_usage():
		try:
			output = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'])
			gpu_usage = [int(x) for x in output.strip().split(b'\n')]
			return gpu_usage
		except subprocess.CalledProcessError:
			# Handle the case when nvidia-smi fails to execute
			return None
		
def myCRC(data):
	crc32_func = crcmod.mkCrcFun(0x104c11db7, initCrc=0xFFFFFFFF, xorOut=0xFFFFFFFF)
	return crc32_func(data)

if __name__ == "__main__":
	server_host = server()
	# start profiler
	server_host.profiler()

	# Inference transmitting
	count = 0
	prev_count = 0
	while True:
		packet,_ = server_host.data_rx.recvfrom(server_host.BUFF_SIZE)
		decode_data = pickle.loads(packet)
		# check if packet is lossed
		if (decode_data[0] != (prev_count + 1) % 10):
			missed_packet = pickle.dumps((0, {}))
			server_host.data_rx.sendto(serialized_tensor, (server_host.edge_host_ip, server_host.INFERENCE_PORT))
			continue
		prev_count = decode_data[0]

		data = base64.b64decode(decode_data[1],' /')
		npdata = np.fromstring(data,dtype=np.uint8)
		frame = cv2.imdecode(npdata,1)
		# Inferencing, record the time for referencing
		start_time = time.time()
		pil_image = Image.fromarray(frame)
		prediction = detect_objects(pil_image)
		end_time = time.time()
		print(end_time - start_time)

		# calculate check_sum
		check_sum = myCRC(pickle.dumps(1, prediction))

		# send prediction back to server
		serialized_tensor = pickle.dumps((1, prediction, check_sum))
		server_host.data_rx.sendto(serialized_tensor, (server_host.edge_host_ip, server_host.INFERENCE_PORT))

		# show the predicted 
		cv2.imshow('Object Detection', frame)
	

			



		



