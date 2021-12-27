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


def length_of_line_segment(line):
    x1, y1, x2, y2 = line
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def display_lines(frame, lines, line_color=(0, 255, 0), line_width=1):
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return line_image


def detect_line_segments(cropped_edges):
    # tuning min_threshold, minLineLength, maxLineGap is a trial and error process by hand
    rho = 1  # precision in pixel, i.e. 1 pixel
    angle = np.pi / 180  # degree in radian, i.e. 1 degree
    min_threshold = 10  # minimal of votes
    line_segments = cv2.HoughLinesP(cropped_edges, rho, angle, min_threshold, np.array([]), minLineLength=20,
                                    maxLineGap = 1)

    if line_segments is not None:
        for line_segment in line_segments:
            logging.debug('detected line_segment:')
            logging.debug("%s of length %s" % (line_segment, length_of_line_segment(line_segment[0])))
            # print('detected line_segment:')
            # print("%s of length %s" % (line_segment, length_of_line_segment(line_segment[0])))

    return line_segments


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


class detection_one_line():

    base_point = 0

    def __init__(self):
        print('init det on line')
        self.base_point = int(1280 / 2)

    def img_to_base_point(self,img):

        # org array:480,640 좌측을 100 만큼 짤라냄. => array:480,540 / img:540*480
        # img_del = img[:, 100:]  # org 480,640 좌측을 100 만큼 짤라냄.
        # print(f'img shape = {img.shape}')

        cimg = img.copy()
        img_del = cimg[:,100:]  # org array:480,640 좌측을100만큼 짤라냄. => array:480,540/img:540*480

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        h = cv2.inRange(h, 20, 35)
        hsv = cv2.bitwise_and(hsv, hsv, mask=h)

        out_line = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        imshow('out_line', out_line, False)

        gray = cv2.cvtColor(out_line, cv2.COLOR_RGB2GRAY)
        imshow('gray', gray, False)

        rtn, img_r = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        imshow('img_r', img_r, False)

        height, width = img_r.shape

        polygon_area = np.array([[
                    (width * 1/3 , height * 2/3),
                    (width - (width * 1/3) , height * 2/3),
                    (width, height),
                    (0, height) ]], np.int32)

        masked_img_r = region_of_interest(img_r, polygon_area)
        imshow('masked_img_r',masked_img_r,True)

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

        r_base = stabilize_steering_angle(self.base_point, new_point ,max_angle_deviation=100)
        self.base_point = r_base

        return self.base_point


if __name__ == '__main__' :


    det = detection_one_line()

    files = os.listdir('./sb-imgs/imgs-1')

    for file in files:

        img = cv2.imread('./sb-imgs/imgs-1/'+file, cv2.IMREAD_COLOR)
        imgc = img.copy()

        base = det.img_to_angle(imgc)

        cv2.circle(imgc,(det.base_point, imgc.shape[0]), 100, (0,255,255),2)
        imshow('imgc',imgc,True)

        if cv2.waitKey(0) & 0xFF == ord('s'):
            pass

