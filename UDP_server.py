
# This is server code to send video frames over UDP
import cv2, imutils, socket
import numpy as np
import time
import base64
import json

# initialize parameters
BROADCASTED = False    # Used for the initial handshake procedure

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


BUFF_SIZE = 65536
server_socket = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
# define the socket buffer size; buffer size should be big enough
server_socket.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,BUFF_SIZE)
# enable reuse of port 
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, True)

# getting host server ip
host_ip = '192.168.0.142'
port = 9999
socket_address = (host_ip,port)
server_socket.bind(socket_address)
print('Begin listening at:',socket_address)

# BROADCAST MESSAGE server IP to be broadcasted and setup
server_ip = (host_ip + " " + str(port)).encode()

vid = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER) 
fps,st,frames_to_count,cnt = (0,0,20,0)

while True:
    msg,client_addr = server_socket.recvfrom(BUFF_SIZE)
    print('GOT connection from ',client_addr)
    WIDTH=400
    while(vid.isOpened()):
        # BROADCAST the host-ip info toward client
        if (not BROADCASTED):
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            server_socket.sendto(message, ('<broadcast>', port))
            BROADCASTED = True
            print("server IP information sent to client")
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, False)    # disable socket after send
            continue

        # TODO: 
        # write a code to decide if the offloading should happen or not
        # testing of the connection by send a few frame of data toward client

        _,frame = vid.read()
        frame = imutils.resize(frame,width=WIDTH)
        encoded,buffer = cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,80])
        message = base64.b64encode(buffer)
        server_socket.sendto(message,client_addr)
        frame = cv2.putText(frame,'FPS: '+str(fps),(10,40),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        cv2.imshow('TRANSMITTING VIDEO',frame)

        # TODO:
        # code used to recieve prediction results 
        
        # code exit on q
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            server_socket.close()
            break