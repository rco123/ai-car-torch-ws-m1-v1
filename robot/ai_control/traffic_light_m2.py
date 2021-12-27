import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def imshow(win,img,show=False):
    if show == True:
        img = cv2.resize(img,(0,0),fx=0.5,fy=0.5,interpolation=cv2.INTER_LINEAR)
        cv2.imshow(win,img)
        cv2.waitKey(1)


class traffic_light_det():

    def __init__(self):

        self.traffic_light_loc = []

    def img_check(self,img,th):
        self.img = img
        self.img_copy = img.copy()

        traffic_light_loc = []

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # print(f'img shape = {gray.shape}')
        gray = cv2.blur(gray,(3,3),None)
        ret, img_threshold = cv2.threshold(gray,th, 255, cv2.THRESH_BINARY)
        imshow('threshold',img_threshold,True)
                          # detect edges
                          
        canny_img = cv2.Canny(img_threshold, 100, 200)
        imshow('canny_img',canny_img,True)

        # Detecting contours in image.
        contours, _= cv2.findContours(canny_img, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
        print('contours check')
        for c in contours:
            
            area_size = cv2.contourArea(c)
            x,y,w,h = cv2.boundingRect(c)
            ratio = area_size / (w * h)
            
            if area_size < 500:
                continue
            if area_size > 2000:
                continue
            if x < 30 : # almost left side location
                continue
            
            #size compare
            if w < 50:
                continue
            if w >= 75:
                continue

            if h < 15:
                continue
            if h > 25:
                continue

            # # 비율 비교
            if w / h < 2 or w / h > 3:
                continue

            # # # position compare
            height,width = canny_img.shape
            if x > int(width/4):
                continue
            if y > int(height / 2):
                continue
            
            # # area compare
            # if ratio < 0.7:
            #     continue
            # traffic_light_loc.append((x,y,w,h))

            print(f'axis = {x},{y},{w},{h}, {w * h}, {w/h}, {ratio}')
            print(f'area_size = {area_size}' )
            cv2.rectangle(self.img_copy, (x, y), (x + w, y + h), (255, 255,0), 3)

        imshow('img_copy',self.img_copy,True)
    


    def detect(self,img,th=150):

        self.img = img
        self.img_copy = img.copy()

        traffic_light_loc = []

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # print(f'img shape = {gray.shape}')
        gray = cv2.blur(gray,(3,3),None)
        ret, img_threshold = cv2.threshold(gray,th, 255, cv2.THRESH_BINARY)
        imshow('threshold',img_threshold,False)
                          # detect edges
        canny_img = cv2.Canny(img_threshold, 100, 200)
        imshow('canny_img',canny_img,False)

        # Detecting contours in image.
        contours, _= cv2.findContours(canny_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:

            area_size = cv2.contourArea(c)
            x,y,w,h = cv2.boundingRect(c)
            ratio = area_size / (w * h)
            
            if area_size < 500:
                continue
            if area_size > 2000:
                continue
            if x < 30 : # almost left side location
                continue
            
            #size compare
            if w < 50:
                continue
            if w >= 75:
                continue

            if h < 15:
                continue
            if h > 25:
                continue

            # # 비율 비교
            if w / h < 2 or w / h > 3:
                continue

            # # # position compare
            height,width = canny_img.shape
            if x > int(width/4):
                continue
            if y > int(height / 2):
                continue
            
            # area compare
            ratio = area_size / (w * h)
            if ratio < 0.7:
                continue

            traffic_light_loc.append((x,y,w,h))

            # print(f'axis = {x},{y},{w},{h}, {w * h}, {w/h}, {ratio}')
            # print(f'area_size = {area_size}' )
            cv2.rectangle(self.img, (x, y), (x + w, y + h), (255, 255,0), 2)

        # imshow('img_copy',self.img_copy,False)

        if len(traffic_light_loc):
            self.traffic_light_loc = traffic_light_loc
            return True
        else:
            return False

    def check(self):

        loc_list = self.traffic_light_loc

        for loc in loc_list:
            x,y,w,h = loc

            want_area = self.img[y - (h * 2):y + h, x - 10:x + w]
            # cv2.rectangle(self.img, (x-10, y - int(h*1.5)), (x + w, y + h), (255, 0,255), 1)

            gray = cv2.cvtColor(want_area, cv2.COLOR_BGR2GRAY)
            # print(f'img shape = {gray.shape}')
            ret, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            imshow('light threshold binary', binary, False )  # detect area image
            # detect edges

            x_axis_sum = np.sum(binary, axis=0)
            # print(x_axis_sum)
            # print(len(x_axis_sum))

            # plt.plot(np.arange(len(x_axis_sum)),x_axis_sum)
            # plt.pause(1)

            x_pos_points = np.where(x_axis_sum > 2000)
            # print('light shape',x_pos_points,x_pos_points[0].size)

            if x_pos_points[0].size == 0:
                return 0

            x_pos_val = np.mean(x_pos_points)
            loc_ratio = (x_pos_val / len(x_axis_sum))

            # print(x_pos_val, loc_ratio)

            if loc_ratio > 0 and loc_ratio <= 0.3:
                return 1
            elif loc_ratio > 0.3 and loc_ratio <= 0.6:
                return 2
            elif loc_ratio > 0.6 and loc_ratio <= 1:
                return 3



if __name__ == '__main__' :

    run_type = 'video'

    det = traffic_light_det()

    vid_file = 'D:\\pycham-prj\\ai-work\\1.bk-ai-car-m1\\2.detect\\data-1\\output-1.avi'

    if run_type == 'video':
        cap = cv2.VideoCapture(vid_file)
        while True:
            ret, img = cap.read()
            if not ret :
                print('read error')
                exit()
            rtn = det.traffic_light_detect(img)
            if rtn:
                # for i in det.traffic_light_loc:
                #     x, y, w, h = i
                #     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)

                no = det.traffic_light_check()
                print(f'light no = {no}')
            else:
                # print('no_detect')
                pass

            imshow('img', img, True)
            cv2.waitKey(0)

        exit()


    dir = '../data/1/imgs'
    dir = 'D:\\pycham-prj\\ai-work\\1.bk-ai-car-m1\\2.detect\\data\\tls\\2\\imgs\\'
    dir = 'D:\\pycham-prj\\ai-work\\1.bk-ai-car-m1\\2.detect\\data-1\\imgs\\'

    files = os.listdir(dir)
    files = files[361:]

    for file in files:
        print(file)
        file = os.path.join(dir,file)
        font = cv2.FONT_HERSHEY_COMPLEX

        img = cv2.imread(file)
        rtn = det.traffic_light_detect(img)
        print(rtn)
        if rtn :
            # for i in det.traffic_light_loc:
            #     x,y,w,h = i
            #     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255,0), 1)

            no = det.traffic_light_check()
            print(f'light no = {no}')
        else:
            print('no_detect')

        imshow('img',img,True)
        cv2.waitKey()

