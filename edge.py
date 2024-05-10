import cv2, imutils, socket
import numpy as np
import time
import base64
import pickle
import multiprocessing
import jetson_utils
import jetson_inference
import crcmod
import subprocess
import sys
import psutil

class edge:
	def __init__(self) -> None:
		self.BUFF_SIZE = 65536
		self.EDGE_HOST_IP = '192.168.0.142'
		self.server_address = 'x'
		self.PROFILING_PORT_OUT = 9998      # EDGE_PROFILING_OUT ---> SERVER_PROFILING_IN
		self.PROFILING_PORT_IN = 9999       # EDGE_PROFILING_IN ---> SERVER_PROFILING_OUT
		self.DATA_PORT = 8080
		self.INFERENCE_PORT = 8081
		self.WIDTH=400       # Width of the frame to be sent
		self.server_info = {}       # dictionary used to store hardware profiling result, AND EVERY edge + server STATE
		self.time_out = 5					# seconds a server is considered inactivate

		# Initialization of transmission port
		# setup DATA_PORT
		self.data_tx = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
		# define the socket buffer size; buffer size should be big enough
		self.data_tx.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF, self.BUFF_SIZE)
		# enable reuse of port 
		self.data_tx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, True)

		# PREDICTION_PORT
		self.pred_rx = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
		# define the socket buffer size; buffer size should be big enough
		self.pred_rx.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF, self.BUFF_SIZE)
		# enable reuse of port 
		self.pred_rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, True)

	# sent out a broadcast message in defined frequency
	# MESSAGE_TYPE: (start_send_time, payload)
	def profiling_out(self):
		# setting up profiler
		profiling_out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
		# define the socket buffer size; buffer size should be big enough
		profiling_out.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.BUFF_SIZE)
		# enable reuse of port 
		profiling_out.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, True)
		# enable broadcasting mode
		profiling_out.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

		while True:
			# set sent message
			# send the message
			start_send_time = time.time()
			edge_ip = self.EDGE_HOST_IP
			msg = (start_send_time, edge_ip)
			payload = "12345678"
			msg = (start_send_time, payload)
			encode_msg = pickle.dumps(msg)

			# send the message
			profiling_out.sendto(encode_msg, ('<broadcast>', self.PROFILING_PORT_OUT))
			time.sleep(0.1)

	# recieving profiling information from server and store it 
	def profiling_in(self):
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
		while True:
			data, address = profiling_in.recvfrom(3096*2)
			# TODO: testing purpose only
			self.server_address = address
			recieve_time = time.time()
			# decode the data
			server_n = pickle.loads(data)
			cpu_info = server_n[0]
			gpu_info = server_n[1]
			send_time = server_n[2]
			count = server_n[3]
			# check if there are lossed packets
			if address in self.server_info:
				prev_count = self.server_info[address][4]
				if prev_count + 1 != count:
					# loss of packets occured, NOT resending, viewed as invalid
					continue
			# calculate the RRT
			delay = recieve_time - send_time
			# identify a threshold for preprocessing, if the delay is too big, no need to consider it
			threshold = 0.05
			# store it in the server_info, check if ip adress already profiled due to UDP server duplicate packet
			if delay > threshold:
					continue
			
			# calculate the throughput of connection
			throughput = sys.getsizeof(data) / delay
			# update the server_info dict
			if address in self.server_info:
				if self.server_info[address][0] == send_time:
					continue
				else:
					self.server_info[address] = (send_time, cpu_info, gpu_info, throughput, count, 0)
			else:
				self.server_info[address] = (send_time, cpu_info, gpu_info, throughput, count, 0)

			# update edge device
			self.server_info[self.EDGE_HOST_IP] = (time.time(), psutil.cpu_percent(interval=0.5), self.get_gpu_usage(), 0, 0, 0)
			print("offboard delay is: ", delay)

			# delete server if inactivated for 500
			for ip_adress, tpl in self.server_info.items():
				if time.time() - self.server_info[ip_adress][0] > self.time_out:
					self.server_info.pop(ip_adress)

	# profiler: combination of profiler_outport and profiler_inport
	def profiler(self):
		profiling_out = multiprocessing.Process(target=self.profiling_out)
		profiling_in = multiprocessing.Process(target=self.profiling_in)
		profiling_out.start()
		profiling_in.start()
		# profiler running forever, no need to join the process

	def myCRC(self, data):
		crc32_func = crcmod.mkCrcFun(0x104c11db7, initCrc=0xFFFFFFFF, xorOut=0xFFFFFFFF)
		return crc32_func(data)
	
	def get_gpu_usage(self):
		try:
			output = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'])
			gpu_usage = [int(x) for x in output.strip().split(b'\n')]
			return gpu_usage
		except subprocess.CalledProcessError:
			# Handle the case when nvidia-smi fails to execute
			return None

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
	
if __name__ == "__main__":
	edge_host = edge()

	# start profiler
	edge_host.profiler()

	# read video source
	vid = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER) 

	# load AI model
	net = jetson_inference.detectNet("ssd-mobilenet-v2", threshold=0.5)

	offloading = False
	frame_count = 0
	while(vid.isOpened()):
		if len(edge_host.server_info) == 0:
			continue
		_, frame = vid.read()

		# TODO: determine offloading or not every 5
		# 1. onboard cpu, gpu info ----> onbaord processing time
		# 2. offboard cpu, gpu info ----> offboard processing time + payload size / data throughput
		# weighted score: (processing time - transmission time) * (if onboard 1, if offboard, 1.5)
		frame_count = frame_count + 1
		offload_score = {}
		if (frame_count == 5):
			for ip_adress, tpl in edge_host.server_info.items():
				if ip_adress == edge_host.EDGE_HOST_IP:
					offload_score[ip_adress] = tpl[1] + 1.5*tpl[2]
					continue
				offload_score[ip_adress] = tpl[1] + 1.5*tpl[2]- sys.getsizeof(frame) / tpl[3]
			max_ip_addr = max(offload_score, key=offload_score.get)
			if max_ip_addr == edge_host.EDGE_HOST_IP:
				offloading = False
			else:
				offloading = True
	
		if not offloading:
			start_time = time.time()
			# Convert the frame to RGBA format expected by Jetson Inference
			rgba_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
			# Convert the frame to jetson.utils.cudaImage
			cuda_frame = jetson_utils.cudaFromNumpy(rgba_frame)
			# Detect objects in the frame
			detections = net.Detect(cuda_frame)
			end_time = time.time()
			print("onboard computing time: ", end_time - start_time)
		else:
			# DATA PORT
			frame = imutils.resize(frame,width=edge_host.WIDTH)
			encoded, buffer = cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,80])
			ec_img = base64.b64encode(buffer)
			# DATA port include header count to keep track of the loss packet
			img = pickle.dumps((frame_count, ec_img))
			edge_host.data_tx.sendto(img, (edge_host.server_address[0], edge_host.DATA_PORT))

			# INFERENCE PORT
			edge_host.pred_rx.settimeout(0.1)
			edge_host.pred_rx.bind((edge_host.server_address, edge_host.INFERENCE_PORT))
			prediction_result, server_addr = edge.recvfrom(edge_host.BUFF_SIZE)
			prediction = pickle.loads(prediction_result)
			valid = prediction[0]
			result = prediction[1]
			checksum = prediction[2]

			# check if it is valid data
			# if continuing losting image transmission, remove the server from server_info temporraly
			if valid == 0:
				edge_host.server_info[server_addr][-1] += 1
				if (edge_host.server_info[server_addr][-1] == 28*3):
					edge_host.server_info.pop(server_addr)
				continue

			# check if check sum is correct
			check = edge_host.myCRC(pickle.loads((valid, result)))
			if check != checksum:
				continue	# discarded the prediction result if the checksum did not pass


		
			



	

			



		



