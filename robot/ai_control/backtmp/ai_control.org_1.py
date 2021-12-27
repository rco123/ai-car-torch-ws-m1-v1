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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class ai_controller(threading.Thread):

    cam = None
    img = None

    def __init__(self):
        threading.Thread.__init__(self)

    def cam_img_to_angle(self, img=None):

        # org array:480,640 좌측을 100 만큼 짤라냄. => array:480,540 / img:540*480
        #img_del = img[:, 100:]  # org 480,640 좌측을 100 만큼 짤라냄.
        #print(f'img shape = {img.shape}')

        # if img == None :
        #     img = self.img

        img = self.img.copy()

        img_del = img[:,100:] # org array:480,640 좌측을100만큼 짤라냄. => array:480,540/img:540*480
        
        #print(f'img shape = {img.shape}')

        imgHsv = cv2.cvtColor(img_del, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 83, 0])
        upper = np.array([85, 255, 255])
        mask = cv2.inRange(imgHsv, lower, upper)  # 이차원 이미지로 변경

        #print(f'mask shape = {mask.shape}')

        img2 = mask[int(mask.shape[0]/2):, :] #높이를 반으로 줄임. (240,540)
        #print(f'img2 shape = {img2.shape}')

        half_width = int(img2.shape[1] / 2)
        
        try:
            his_values = np.sum(img2, axis = 0)
            max_value = np.max(his_values)
            #print(f'max vlaue = {max_value}')
            index = np.where( his_values > int(max_value * 0.1) )
            base_point = int(np.average(index))
            
        except ValueError as e:
            print(e)
            print('is not line')
            base_point = half_width
        
        rtv = -int( (base_point - half_width) / half_width * 800 )
        #print(f'base_point = {base_point} rtv = {rtv}')
        return rtv


    def cam_open(self):
        self.cam = cv2.VideoCapture(0)

        if (self.cam.isOpened() == True) :
            #original camera image = 640*480
            self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            return True
        else:
            return False

    def cam_img_get(self):
        ret, img = self.cam.read()
        if ret != True:
            print('read error')
        #print(img.shape)
        #self.img = img[:, 100:]  # org array:480,640 좌측을 100 만큼 짤라냄. => array:480,540 / img:540*480

        self.img = img
        return self.img

    def img_write(self,filename):
        cv2.imwrite(filename, self.img)

    def img_read(self,filename):
        img = cv2.imread(filename, cv2.IMREAD_COLOR)
        self.img = img
        return img
                
    def img_display(self,ratio=1):
        #img = cv2.resize(self.img,None,fx = ratio, fy = ratio)
        cv2.imshow('img', self.img)
        cv2.waitKey(1)

    def cam_dis_start(self):
        self.run_stop = False
        self.run()

    def cam_dis_stop(self):
        self.run_stop = True
        
    def run(self):
        while True :
            if this.run_stop == True:
                break
            ret, self.img = self.cam.read()
            cv2.imshow('img', self.img)
            cv2.waitKey(1)

if __name__ == '__main__' :

    js_fd = "/dev/input/js0"
    # ps_con = ps_controller(interface=js_fd,connecting_using_ds4drv=False)
    # ps_con.start()
    # ps_ser = ser_controller('/dev/ttyUSB0')
    # ps_ai = ai_controller('model.h5', 200, ps_con, ps_ser)
    # ps_ai.start()
