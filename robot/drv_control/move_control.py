#!/usr/bin/python3

import time
import serial
import cv2
import threading
from queue import Queue
import inspect
#from tensorflow.keras.models import load_model
import os
import numpy as np

from ps_control import ps_controller
from ser_control import ser_controller

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class move_controller(threading.Thread):

    cam = None
    model = None
    ser = None
    turn = 0
    speed = 0

    run_or_stop = False

    cali = 1000 * 1.5

    def __init__(self, model_name, speed, jobj, sobj):
        threading.Thread.__init__(self)

        self.cam = cv2.VideoCapture(0)

        #print(f'load_model = {model_name}')
        #self.model = load_model(model_name)
        self.speed = speed
        self.sobj = sobj
        self.jobj = jobj


    def limit_check(self):
        if self.turn > 800: self.turn = 800
        if self.turn < -800: self.turn = -800
        if self.speed > 1000 : self.speed = 1000
        if self.speed < -1000 : self.speed = -1000


    def run(self):
        while True :
            if self.jobj.key_sts[0] == 8: 
                self.run_or_stop = True
            if self.jobj.key_sts[0] == 9:
                self.run_or_stop = False

            if self.run_or_stop == True:
                #if True:    
                if (self.cam.isOpened()) :
                    ret, img = self.cam.read()
                    print(img.shape)

                    img_del = img[:, 100:]  # org 480,640 좌측을 100 만큼 짤라냄.

                    cv2.imshow('img', img_del)

                    angle = -1 * cam_img_to_angle(img_del)
                    
                    self.limit_check()
                    print(f'angle =  {angle}')
                    self.sobj.cmd(angle, 200)

                    if cv2.waitKey(1) & 0xff == ord('q') :
                        break


            if self.jobj.key_sts[0] == 7: 
                self.sobj.cmd(0,0)


            time.sleep(0.01)


if __name__ == '__main__' :

    js_fd = "/dev/input/js0"
    ps_con = ps_controller(interface=js_fd,connecting_using_ds4drv=False)
    ps_con.start()
    ps_ser = ser_controller('/dev/ttyUSB0')

    ps_ai = ai_controller('model.h5', 200, ps_con, ps_ser)
    ps_ai.start()






