import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def imshow(win,img,show=False):
    if show == True:
        cv2.imshow(win,img)

class traffic_light_det():
    
    def __init__(self):
        self.traffic_light_loc = []
        self.img = None
        
    def traffic_light_detect(self,img):

        self.img = img.copy()
        t_loc = []

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # print(f'img shape = {gray.shape}')
        ret, binary = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
        # detect edges
        canny_img = cv2.Canny(binary, 100, 200)
        # imshow('canny_img',canny_img,True)

        # Detecting contours in image.
        contours, _= cv2.findContours(canny_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:

            x,y,w,h = cv2.boundingRect(c)
            area_size = cv2.contourArea(c)

            # size compare
            if w < 50 :
                continue
            if w >= 70:
                continue
            #if h < 15 :
            #    continue
            if h > 25 :
                continue

            # 비율 비교
            if w/h < 2.9 or w/h > 4 :
                continue

            # position compare
            height,width = canny_img.shape
            if x > int(width/3):
                continue
            if y > int(height/2):
                continue

            # area compare
            ratio = area_size / (w * h)
            if ratio < 0.7:
                continue

            t_loc.append((x,y,w,h))

            print(f'axis = {x},{y},{w},{h}, {w * h}, {w/h}, {ratio}')
            print(f'area_size = {area_size}' )
            
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255,0), 1)
        
        imshow('traffic',img,True)
        self.traffic_light_loc = t_loc
        
        if len(t_loc):
            return True
        else:
            return False


    def traffic_light_check(self):

        img = self.img

        loc_list = self.traffic_light_loc

        for loc in loc_list:
            x,y,w,h = loc
            want_area = img[y - (h * 2):y + h, x - 10:x + w]

            gray = cv2.cvtColor(want_area, cv2.COLOR_BGR2GRAY)
            # print(f'img shape = {gray.shape}')
            ret, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
            # imshow('binary', binary, True)
            
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

            print(x_pos_val, loc_ratio)



            if loc_ratio >= 0 and loc_ratio <= 0.33:
                return 1
            elif loc_ratio > 0.33 and loc_ratio <= 0.6:
                return 2
            elif loc_ratio > 0.6 and loc_ratio <= 1:
                return 3
            else:
                return -1


if __name__ == '__main__' :
    
    dir = '../data/1/imgs'
    dir = 'D:\\pycham-prj\\ai-work\\1.bk-ai-car-m1\\2.detect\\data\\tls\\2\\imgs\\'

    files = os.listdir(dir)

    for file in files:

        file = os.path.join(dir,file)
        font = cv2.FONT_HERSHEY_COMPLEX

        img = cv2.imread(file)
        det = traffic_light_det(img)
        rtn = det.traffic_light_detect()
        print(rtn)
        if rtn :
            for i in det.traffic_light_loc:
                x,y,w,h = i
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255,0), 1)

            no = det.traffic_light_check()
            print(f'light no = {no}')
        else:
            print('no_detect')

        imshow('img',img,True)
        cv2.waitKey()

