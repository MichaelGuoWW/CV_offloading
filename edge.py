import cv2, imutils, socket
import numpy as np
import time
import base64
import pickle
import multiprocessing
from PIL import Image, ImageDraw
import jetson_utils
import jetson_inference
class edge:
    def __init__(self) -> None:
        self.BUFF_SIZE = 65536
        self.EDGE_HOST_IP = '192.168.0.142'
        self.PROFILING_PORT_OUT = 9998      # EDGE_PROFILING_OUT ---> SERVER_PROFILING_IN
        self.PROFILING_PORT_IN = 9999       # EDGE_PROFILING_IN ---> SERVER_PROFILING_OUT
        self.DATA_PORT = 8080
        self.INFERENCE_PORT = 8081
        self.WIDTH=400       # Width of the frame to be sent
        self.server_info = {}       # dictionary used to store

    # sent out a broadcast message in defined frequency
    # MESSAGE_TYPE: (start_send_time, )
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
            print("broadcasting to all ->>>>>>>>")
            time.sleep(0.1)

    # recieving profiling information from server and store it at 
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

            # update the server_info dict
            if address in self.server_info:
                if self.server_info[address][0] == send_time:
                    continue
                else:
                    self.server_info[address] = (send_time, cpu_info, gpu_info, delay, count)
            else:
                self.server_info[address] = (send_time, cpu_info, gpu_info, delay, count)
            print(address, "RRT is: ", delay)

    # profiler: combination of profiler_outport and profiler_inport
    def profiler(self):
        freq = 1
        profiling_out = multiprocessing.Process(target=self.profiling_out)
        profiling_in = multiprocessing.Process(target=self.profiling_in)
        profiling_out.start()
        profiling_in.start()
        # profiler running forever, no need to join the process
        
if __name__ == "__main__":
    edge_host = edge()
    # start profiler
    edge_host.profiler()

    # read video source
    vid = cv2.VideoCapture("test_video.mp4") 

    # load AI model
    net = jetson_inference.detectNet("ssd-mobilenet-v2", threshold=0.5)

    offloading = False
    while(vid.isOpened()):
        _, frame = vid.read()

        if not offloading:
            start_time = time.time()
            # Convert the frame to RGBA format expected by Jetson Inference
            rgba_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            # Convert the frame to jetson.utils.cudaImage
            cuda_frame = jetson_utils.cudaFromNumpy(rgba_frame)
            # Detect objects in the frame
            detections = net.Detect(cuda_frame)
            end_time = time.time()
            print(end_time - start_time)


    

            



        



