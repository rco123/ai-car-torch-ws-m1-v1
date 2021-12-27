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


class lane_det_m1():

    direction_point = 0

    def __init__(self):
        print('init det on line')
        self.base_point = int(1280 / 2)

    def img_to_angle(self,img):

        imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
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
            d_point = int(np.average(index))
            
        except ValueError as e:
            print(e)
            print('is not line')
            d_point = half_width
        
        rtv = -int( (base_point - half_width) / half_width * 800 )
        #print(f'base_point = {base_point} rtv = {rtv}')
        return d_point

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

