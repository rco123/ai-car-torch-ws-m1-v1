#!/usr/bin/python3

import time
import cv2
import threading
from queue import Queue
import inspect
#from tensorflow.keras.models import load_model
import os
import numpy as np
from . import lane_det_m1 as lane_dec
#import detect_one_line as dtc_ol
from . import detect_one_line as dtc_ol
from . import traffic_light_m1 as dtc_tl
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class ai_controller(threading.Thread):

    cam = None
    img = None
    lane_follower = None
    cam_dis_ratio = 0.5

    def __init__(self):
        threading.Thread.__init__(self)
        self.lane_follower = lane_dec.HandCodedLaneFollower()
        #self.lane_follower = dtc_ol.detection_one_line()
        self.traffic_light = dtc_tl.traffic_light_det()

    def traffic_light_detector(self):
        rtn = self.traffic_light.traffic_light_detect(self.img)
        if rtn :
            for i in self.traffic_light.traffic_light_loc:
                x,y,w,h = i
                cv2.rectangle(self.img, (x, y), (x + w, y + h), (0, 255,0), 1)

            num = self.traffic_light.traffic_light_check()
            print(f'light no = {num}')
            return num
        else:
            # print('no_detect')
            return -1
        

    def xml_detector(self, xml='./classifier/cascade.xml'):

        xml_classi = cv2.CascadeClassifier(xml)

        img = self.img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        pos = xml_classi.detectMultiScale(gray, scaleFactor=2, minNeighbors=3)

        for(x,y,w,h) in pos:

            cnt = 0
            cv2.rectangle(img,(x,y), (x+w, y+h), (255,0,0),2)
            cv2.putText(img,'object', (x,y-10),2,0.7,(0,255,0),2, cv2.LINE_AA)

            #circles = cv2.HoughCircles(gray,
            #    cv2.HOUGH_GRADIENT, 1, minDist=20, param1=50,
            #    param2=30, minRadius=10, maxRadius=40)

            '''
            if circles is not None:
                detected_circles = np.uint16(np.around(circles))
                for pt in detected_circles[0, :]:
                    a, b, r = pt[0], pt[1], pt[2]
                    # Draw the circumference of the circle.
                    cv2.circle(img, (a, b), r, (0, 255, 0), 2)
                    #Draw a small circle (of radius 1) to show the center.
                    cv2.circle(img, (a, b), 1, (0, 0, 255), 3)

                    if (a > x) and (a < x+w) :
                        cnt +=1

            '''
            #if cnt < 2 : break
            cv2.rectangle(img,(x,y), (x+w, y+h), (255,0,0),2)
            cv2.putText(img,'object', (x,y-10),2,0.7,(0,255,0),2, cv2.LINE_AA)

            if x < 20 : detect = True
            else: detect = False
            return detect

    def cam_img_to_angle(self, img=None):

        '''
        img = self.img.copy()

        bpoint = self.lane_follower.img_to_base_point(img)
        half_width = int(img.shape[1] / 2)
        diff =  bpoint - half_width
        
        rtv =  - ( ((diff) / half_width) * 800 ) * 1.2 
        if rtv > 800 : 
            rtv = 800
        if rtv < -800 : 
            rtv = -800

        print(f'diff = {diff} rtv = {int(rtv)}')
        return int(rtv)
        '''


        
        img = self.img.copy()
        
        frame = img[:,200:,:] # org array:480,640 좌측을100만큼짤라냄. => array:480,540/img:540*480

        self.lane_follower.follow_lane(frame)
        diff = angle = self.lane_follower.curr_steering_angle

        #angle_diff = 90 - angle
        #rtvalue = int(angle_diff * 800/45)
        #print(f'angle rtn value = {rtvalue}')
        #return rtvalue
        half_width = int( frame.shape[1] / 2 )


        rtv =  - ( ((diff) / half_width) * 800 ) * 1 
        if rtv > 800 : 
            rtv = 800
        if rtv < -800 : 
            rtv = -800

        print(f'diff = {diff} rtv = {int(rtv)}')
        
        return int(rtv)
        
        
        
    '''
    def cam_img_to_angle(self, img=None): # 라즈베리파이용

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
    '''

    def cam_open(self):
        self.cam = cv2.VideoCapture(0)
        # ORBBEC Astro Pro, 1280x720, 16:9 Ratio

        if (self.cam.isOpened() == True) :
            #original camera image = 1280x720
            self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            return True
        else:
            return False

    def cam_img_get(self):
        ret, img = self.cam.read()
        if ret != True:
            print('read error')

        # Add Bright
        # val = 100
        # array = np.full(img.shape, (val,val,val), dtype=np.uint8)    
        # img = cv2.add(img, array)

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

    def img_dis_ratio(self,ratio):
        self.cam_dis_ratio = ratio
                
    def img_display(self):
        dst = cv2.resize( self.img, dsize=(0,0) ,fx = self.cam_dis_ratio, fy = self.cam_dis_ratio)
        #h, w, d = dst.shape
        #print(h,w,d)
        cv2.imshow('img', dst)
        
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
    ai = ai_controller()

    # ps_con = ps_controller(interface=js_fd,connecting_using_ds4drv=False)
    # ps_con.start()
    # ps_ser = ser_controller('/dev/ttyUSB0')
    # ps_ai = ai_controller('model.h5', 200, ps_con, ps_ser)
    # ps_ai.start()
