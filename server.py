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
import pickle
import multiprocessing


class server:
    def __init__(self) -> None:
        self.BUFF_SIZE = 65536
        self.PROFILING_PORT_IN = 9998      # EDGE_PROFILING_OUT ---> SERVER_PROFILING_IN
        self.PROFILING_PORT_OUT = 9999       # EDGE_PROFILING_IN ---> SERVER_PROFILING_OUT
        self.DATA_PORT = 8080
        self.INFERENCE_PORT = 8081
        self.WIDTH=400       # Width of the frame to be sent
    
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
        # enable broadcasting mode
        profiling_out.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

        while True:
            data, address = profiling_in.recvfrom(3096*2)
            start_time = time.time()
            # decode the data
            edge = pickle.loads(data)
            send_time = edge[0]
            cpu_info = 0
            gpu_info = 0
            end_time = time.time()
            # calculate time used on profiling and decoding inputs
            profiling_time = end_time - start_time
            msg = (cpu_info, gpu_info, profiling_time, send_time)
            encode_msg = pickle.dumps(msg)
            profiling_out.sendto(encode_msg, (address[0], self.PROFILING_PORT_OUT))
            print(send_time)
        

if __name__ == "__main__":
    server_host = server()

    # start profiler
    server_host.profiler()
    

            



        



