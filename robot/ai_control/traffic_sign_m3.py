import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

import tensorflow as tf
import time

def imshow(win,img,show=False):
    if show == True:
        img = cv2.resize(img,(0,0),fx=0.5,fy=0.5,interpolation=cv2.INTER_LINEAR)
        cv2.imshow(win,img)
        cv2.waitKey(1)

class traffic_sign_det():

    def __init__(self):
        self.traffic_sign_loc = []

    def load_model(self,model_file):
        print('load model start')
        self.model = tf.keras.models.load_model(model_file)
        # self.model.summary()
        ta = np.zeros((1,32,32,1))
        predictions = self.model(ta) # pre test
        predictions = self.model(ta) # pre test
        print('load model end')
      
    def img_check(self,img,th=150):

        self.img = img
        self.img_copy = img.copy()

        traffic_sign_loc = []

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # print(f'img shape = {gray.shape}')
        ret, binary = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY)
        imshow('threshold_gray', binary, True)
        # detect edges
        canny_img = cv2.Canny(binary, 100, 200)
        imshow('canny_img',canny_img,True)

        # Detecting contours in image.
        contours, _= cv2.findContours(canny_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:

            x,y,w,h = cv2.boundingRect(c)
            area_size = cv2.contourArea(c)
            # # inside extractor

            if x < 30:
                continue

            if area_size < 1000:
                continue

            # # size compare
            if w < 50 :
                continue
            if w >= 120:
                continue
            if h < 50 :
                continue
            if h > 120 :
                continue

            # 비율 비교
            if w/h < 0.8 or w/h > 1.2 :
                continue

            # position compare
            height,width = canny_img.shape
            if x > int(width/4):
                continue
            if y > int(height/2):
                continue
            
            # area compare
            ratio = area_size / (w * h)
            if ratio < 0.7:
                continue

            traffic_sign_loc.append((x,y,w,h))
            print('check contours')

            print(f'axis = {x},{y},{w},{h}, {w * h}, {w/h}, {ratio}')
            print(f'area_size = {area_size}' )
            cv2.rectangle(self.img_copy, (x, y), (x + w, y + h), (0, 255,0), 2)
            imshow('sign mark', self.img_copy, True)


    def detect(self,img,th=150):

        self.img = img
        self.img_copy = img.copy()

        traffic_sign_loc = []

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # print(f'img shape = {gray.shape}')
        ret, binary = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY)
        imshow('threshold_gray', binary, True)
        # detect edges
        canny_img = cv2.Canny(binary, 100, 200)
        imshow('canny_img',canny_img,False)

        # Detecting contours in image.
        contours, _= cv2.findContours(canny_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:

            x,y,w,h = cv2.boundingRect(c)
            # # inside extractor
            if x < 30:
                continue

            area_size = cv2.contourArea(c)
            if area_size < 1000:
                continue

            # # size compare
            if w < 50 :
                continue
            if w >= 120:
                continue
            if h < 50 :
                continue
            if h > 120 :
                continue

            # 비율 비교
            if w/h < 0.8 or w/h > 1.2 :
                continue

            # position compare
            height,width = canny_img.shape
            if x > int(width/4):
                continue
            if y > int(height/2):
                continue
            
            # area compare
            ratio = area_size / (w * h)
            if ratio < 0.7:
                continue

            traffic_sign_loc.append((x,y,w,h))

            print(f'axis = {x},{y},{w},{h}, {w * h}, {w/h}, {ratio}')
            print(f'area_size = {area_size}' )
            # cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255,0), 1)

        if len(traffic_sign_loc):
            self.traffic_sign_loc = traffic_sign_loc
            return True
        else:
            return False


    def check(self):

        loc_list = self.traffic_sign_loc

        for loc in loc_list:
            x,y,w,h = loc
            # want_area = img[y - (h * 2):y + h, x - 10:x + w]
            # want_area = img[y:y + h, x:x + w]
            ext = int(w * 0.25)
            want_area = self.img_copy[y-ext:y-ext+h+ext*2,x - ext:x - ext + w + ext * 2 ]

            want_gray = cv2.cvtColor(want_area, cv2.COLOR_BGR2GRAY)
            # print(f'img shape = {gray.shape}')
            # ret, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
            imshow('want_gray', want_gray, False)
            # detect edges
            print(f'wan_gray b shape = {want_gray.shape}')
            img_chk = cv2.resize(want_gray,(32,32), interpolation=cv2.INTER_AREA)
            imshow('img_chk',img_chk,False)

            img_chk = img_chk.reshape(1, 32, 32, 1)
            #print(f'want_gray shape = {img_chk.shape}')
            # predictions = self.model.predict(img_chk)

            predictions = self.model(img_chk)
            print(predictions)
            no = np.argmax(predictions)

            #classIndex = self.model.predict_classes(img_chk)
            #probabilityValue = np.amax(predictions)
            #print(f'predictions = {predictions}')
            #print(f'classIndex = {classIndex}')
            #print(f'probabilityValue = {probabilityValue}')
            return no


if __name__ == '__main__' :
    det = traffic_sign_det()

    #dir = '../data/1/imgs'
    # dir = 'D:\\pycham-prj\\ai-work\\1.bk-ai-car-m1\\2.detect\\data\\sign\\imgs\\'
    dir = 'D:\\pycham-prj\\ai-work\\1.bk-ai-car-m1\\2.detect\\data\\sign\\imgs\\'

    f_model = 'D:\\pycham-prj\\ai-work\\1.bk-ai-car-m1\\traffic-cnn\\model_trained.h5'

    det.load_model(f_model)

    files = os.listdir(dir)
    files = files[150:]

    for file in files:

        file = os.path.join(dir,file)
        font = cv2.FONT_HERSHEY_COMPLEX

        img = cv2.imread(file)
        # det = traffic_sign_det(img)
        rtn = det.traffic_sign_detect(img)

        print(rtn)
        if rtn :
            for i in det.traffic_sign_loc:
                x,y,w,h = i
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255,0), 1)
                #ext = 20
                ext = int(w * 0.25)
                cv2.rectangle(img, (x-ext, y-ext), (x-ext + w+ext*2, y-ext + h+ext*2), (255, 255, 0), 1)

            no = det.traffic_sign_check()
            print(f'light no = {no}')
        else:
            print('no_detect')

        imshow('img',img,True)
        # del det
        cv2.waitKey()

