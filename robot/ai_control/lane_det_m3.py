import cv2
import numpy as np
import logging
import math
import datetime
import sys
import matplotlib.pyplot as plt
import os

def halt(time=0):
    cv2.waitKey(time)

def imshow(title, img, show=False):
    if show:
        img = cv2.resize(img,dsize=(0,0),fx=0.5,fy=0.5, interpolation=cv2.INTER_LINEAR)
        cv2.imshow(title, img)


# polygon_area = np.array([[
#             (0, height * 1 / 2),
#             (width, height * 1 / 2),
#             (width, height),
#             (0, height)]] ,np.int32)

def region_of_interest(frame, polygon_area):
    height, width = frame.shape
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, polygon_area, 255)
    masked_image = cv2.bitwise_and(frame, mask)
    imshow("mask", masked_image,False)
    return masked_image

def stabilize_steering_angle(curr_steering_angle, new_steering_angle, max_angle_deviation=20):

    angle_deviation = new_steering_angle - curr_steering_angle
    if abs(angle_deviation) > max_angle_deviation:
        stabilized_steering_angle = int(curr_steering_angle +
                                        max_angle_deviation * (angle_deviation / abs(angle_deviation)))
        print(f'angl = {stabilized_steering_angle}')
    else:
        stabilized_steering_angle = new_steering_angle

    print(f'curr {curr_steering_angle}, new {new_steering_angle}, return:{stabilized_steering_angle}')
    return stabilized_steering_angle


class lane_det_m3():

    direction_point = 0

    def __init__(self):
        print('init det on line')
        self.base_point = int(1280 / 2)


    def img_check(self,img, threshold):

        #img_del = cimg[:,100:]  # org array:480,640 좌측을100만큼 짤라냄. => array:480,540/img:540*480

        img_copy = img.copy()

        hsv = cv2.cvtColor(img_copy, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        h = cv2.inRange(h, 22, 25)  # yellow color
        hsv = cv2.bitwise_and(hsv, hsv, mask=h)

        out_line = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        imshow('out_line', out_line, True)

        #addition filter
        out_line = cv2.GaussianBlur(out_line,(9,9),0)
        imshow('blue', out_line, True)
        out_line = cv2.medianBlur(out_line,21, dst=None )
        imshow ('meida', out_line, True)


        gray = cv2.cvtColor(out_line, cv2.COLOR_RGB2GRAY)
        imshow('gray', gray, True)

        # gray = cv2.medianBlur(gray,21, dst=None )

        rtn, img_r = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        cv2.moveWindow('img_r', 320, 180)
        imshow('img_r', img_r, True)
        

        # dst = cv2.fastNlMeansDenoising(img_r,None,h=3, templateWindowSize=7, searchWindowSize=21)
        # imshow('dst', dst, True)


        # height, width = img_r.shape

        # polygon_area = np.array([[
        #             (width * 1/3 , height * 2/3),
        #             (width - (width * 1/3) , height * 2/3),
        #             (width, height),
        #             (0, height) ]], np.int32)

        # masked_img_r = region_of_interest(img_r, polygon_area)
        # imshow('masked_img_r',masked_img_r,False)



    def img_to_angle(self,img, threshold):

        img_copy = img.copy()

        hsv = cv2.cvtColor(img_copy, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        h = cv2.inRange(h, 22, 25)  # yellow color
        hsv = cv2.bitwise_and(hsv, hsv, mask=h)

        out_line = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        imshow('out_line', out_line, False)

        #addition filter
        out_line = cv2.GaussianBlur(out_line,(9,9),0)
        imshow('blue', out_line, False)
        out_line = cv2.medianBlur(out_line,21, dst=None )
        imshow ('meida', out_line, False)

        gray = cv2.cvtColor(out_line, cv2.COLOR_RGB2GRAY)
        imshow('gray', gray, False)

        rtn, img_r = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        cv2.moveWindow('img_r', int(320/2), int(180/2))
        imshow('img_r', img_r, True)
        

        height, width = img_r.shape

        polygon_area = np.array([[
                    (width * 1/3 , height * 2/3),
                    (width - (width * 1/3) , height * 2/3),
                    (width, height),
                    (0, height) ]], np.int32)

        masked_img_r = region_of_interest(img_r, polygon_area)
        imshow('masked_img_r',masked_img_r,False)

        his_values = np.sum(masked_img_r, axis=0)
        # print(f'his_values.size = {his_values.size}')

        max_value = np.max(his_values)
        min_value = np.min(his_values)
        print(f'max min vlaue = {max_value},{min_value} ')

        index = np.where( his_values > 10000) # 특정값이상 설정.
        # print(f'index size = {index[0].size}')

        if index[0].size:
            new_point = int(np.average(index))
        else :
            new_point = self.base_point

        r_base = stabilize_steering_angle(self.direction_point, new_point ,max_angle_deviation=100)
        self.direction_point = r_base

        return self.direction_point


if __name__ == '__main__' :

    det = lane_det_m3()

    files = os.listdir('./sb-imgs/imgs-1')

    for file in files:

        img = cv2.imread('./sb-imgs/imgs-1/'+file, cv2.IMREAD_COLOR)
        
        base = det.img_to_angle(img)

        cv2.circle(img,(det.base_point, img.shape[0]), 100, (0,255,255),2)
        imshow('imgc',img,True)

        if cv2.waitKey(0) & 0xFF == ord('s'):
            pass

