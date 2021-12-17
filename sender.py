import numpy as np
from socket import *
import time
from PIL import Image
import cv2


class UDPSender:
    def __init__(self, ip, port):
        self.params = [cv2.IMWRITE_JPEG_QUALITY, 50]
        self.socket = socket(AF_INET, SOCK_DGRAM)
        self.ip = ip
        self.port = port
        self.bufsize = 65536

    def send_something(self, data):
        length = self.socket.sendto(data, (self.ip, self.port))
        if length < 0:
            print("send error")
            exit(0)
        reply = self.socket.recvfrom(self.bufsize)[0]
        return reply

    def encode_send_images(self,images=[]):
        times = len(images)
        if times > 0:
            for i in range(times):
                img_encode = cv2.imencode('.jpg', images[i], self.params)[1]
                data_encode = np.array(img_encode)
                data = data_encode.tostring()
                length = self.socket.sendto(data, (self.ip, self.port))

    def get_encode_result(self,length):
        string_to_decode = []
        for j in range(length):
            left_view_code = np.fromstring(self.socket.recvfrom(65535)[0], np.uint8)
            string_to_decode.append(left_view_code)
        return string_to_decode

    def decode_and_save(self,encodeImgs=[]):
        result = []
        length = len(encodeImgs)
        for j in range(length):
            img_decode = cv2.imdecode(encodeImgs[j], cv2.IMREAD_COLOR)
            result.append(img_decode)
        return result


