#!/usr/bin/python3
import time
import cv2
import threading
from queue import Queue
import inspect
import os
import numpy as np

# from . import lane_det_m1 as lane_dec
#from tensorflow.keras.models import load_model
# from . import detect_one_line as dtc_ol
# from . import traffic_light_m2 as dtc_tl
# from . import traffic_sign_m2 as dtc_sgn
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from robot.ai_control.lane_det_m3 import lane_det_m3
from robot.ai_control.lane_det_m2 import lane_det_m2
from robot.ai_control.lane_det_cnn import lane_det_cnn
from robot.ai_control.traffic_sign_m3 import traffic_sign_det
from robot.ai_control.traffic_light_m2 import traffic_light_det

from jetcam.usb_camera import USBCamera

class ai_controller(threading.Thread):

    cam = None
    img = None
    lane_follower = None
    cam_dis_ratio = 0.5
    main_img_win = 'main_img'
    vid_out = None

    # usb_camera = USBCamera(width=1280, height=720, capture_width=1280, capture_height=720, capture_device=1)

    def __init__(self):
        threading.Thread.__init__(self)

        #self.lane_follower = lane_dec.HandCodedLaneFollower()
        self.lane_det_m3 = lane_det_m3()
        self.lane_det_m2 = lane_det_m2()
        self.lane_det_cnn = lane_det_cnn()

        self.traffic_light = traffic_light_det()
        self.traffic_sign = traffic_sign_det()
        
  
    def lane_det_cnn_load_model(self,file):
        self.lane_det_cnn.load_model(file) 
        
    def traffic_sign_load_model(self,file):
        self.traffic_sign.load_model(file)

    def traffic_sign_detector_check(self,th=150):
        self.traffic_sign.img_check(self.img,th)
        
    def traffic_sign_detector(self,th=150):
        
        rtn = self.traffic_sign.detect(self.img,th)
        if rtn :
            for i in self.traffic_sign.traffic_sign_loc:
                x,y,w,h = i
                # cv2.rectangle(self.img, (x, y), (x + w, y + h), (0, 255,0), 1)
                # #ext = 20
                ext = int(w * 0.15)
                cv2.rectangle(self.img, (x-ext, y-ext), (x-ext + w+ext*2, y-ext + h+ext*2), (255, 255, 0), 2)

            no = self.traffic_sign.check()
            return no
        else:
            return -1

    def traffic_light_detector(self,th=150):
        rtn = self.traffic_light.detect(self.img,th)
        if rtn :
            for i in self.traffic_light.traffic_light_loc:
                x,y,w,h = i
                cv2.rectangle(self.img, (x, y), (x + w, y + h), (0, 255,0), 1)

            num = self.traffic_light.check()
            print(f'light no = {num}')
            return num
        else:
            # print('no_detect')
            return -1

    def traffic_light_detector_check(self,th=150):
        self.traffic_light.img_check( self.img, th)


    def cam_img_to_angle_m1(self, img=None):
        
        img = self.img.copy()

        dpoint = self.lane_det_m1.img_to_angle(img)

        half_width = int(img.shape[1] /2 )
        diff = dpoint - half_width
        rtv = - int( (diff) / half_width * 800 )
        
        if rtv > 800 : 
            rtv = 800
        if rtv < -800 : 
            rtv = -800

        #print(f'diff = {diff} rtv = {int(rtv)}')
        return int(rtv)


    def cam_img_to_angle_m2(self,threshold):
        
        img = self.img
        diff = self.lane_det_m2.img_to_angle(img,threshold)
        rtv =  - ( diff * 1 )

        if rtv >= 800 :
            rtv = 800
        if rtv <= -800 :
            rtv = -800
        #print(f'diff = {diff} rtv = {int(rtv)}')
        return int(rtv)


    def cam_img_to_angle_m3(self, threshold):
        
        img = self.img.copy()

        bpoint = self.lane_det_m3.img_to_angle(img, threshold)
        half_width = int(img.shape[1] / 2)
        diff =  bpoint - half_width
        
        rtv =  - ( ((diff) / half_width) * 800 ) * 1.2 
        if rtv > 800 : 
            rtv = 800
        if rtv < -800 : 
            rtv = -800
        #print(f'diff = {diff} rtv = {int(rtv)}')
        return int(rtv)

    def cam_img_to_angle_m3_check(self, threshold):
        
        img = self.img.copy()
        self.lane_det_m3.img_check(img, threshold)
        cv2.waitKey(1)
        
    def cam_img_to_angle_cnn(self):
        
        img = self.img.copy()
        rtv = self.lane_det_cnn.img_to_angle(img)
        
        #print(f'diff = {diff} rtv = {int(rtv)}')
        return int(rtv)

    def cam_open(self):

        # self.cam = cv2.VideoCapture(0)
        self.cam = cv2.VideoCapture('/dev/video1')
        
        # ORBBEC Astro Pro, 1280x720, 16:9 Ratio

        if (self.cam.isOpened() == True) :
            #original camera image = 1280x720
            self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            return True
        else:
            return False

    def cam_img_read(self):
        ret, img = self.cam.read()
        if ret != True:
            print('read error')

        self.img = img
        return self.img

    def img_write(self,filename):
        cv2.imwrite(filename, self.img)

    def img_read(self,filename):
        img = cv2.imread(filename, cv2.IMREAD_COLOR)
        self.img = img
        return img

    def img_dis_ratio(self,ratio):
        self.cam_dis_ratio = ratio

    def img_display(self):
        resize = cv2.resize( self.img, dsize=(0,0) ,fx = self.cam_dis_ratio, fy = self.cam_dis_ratio,interpolation=cv2.INTER_LINEAR)
        cv2.imshow('main_img', resize)
        cv2.waitKey(1)

    def img_dis_pos(self,x,y):
        cv2.namedWindow('main_img')
        x = int(x)
        y = int(y)
        cv2.moveWindow('main_img', x, y)

    def cam_img_to_avi_open(self, file):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        image_size = (1280,720)
        # str = './vids/out.avi'
        self.vid_out = cv2.VideoWriter(file , fourcc, 20.0, image_size)
        
    def cam_img_to_avi_write(self):
        self.vid_out.write(self.img)
    
    def cam_img_to_avi_release(self):
        self.vid_out.release()
    
    def cam_dis_start(self):
        self.run_stop = False
        self.run()

    def cam_dis_stop(self):
        self.run_stop = True
        
    def run(self):
        while True :
            if self.run_stop == True:
                break
            ret, self.img = self.cam.read()
            cv2.imshow('img', self.img)
            cv2.waitKey(1)


if __name__ == '__main__' :

    js_fd = "/dev/input/js0"
    ai = ai_controller()

    # ps_con = ps_controller(interface=js_fd,connecting_using_ds4drv=False)
    # ps_con.start()
    # ps_ser = ser_controller('/dev/ttyUSB0')
    # ps_ai = ai_controller('model.h5', 200, ps_con, ps_ser)
    # ps_ai.start()
