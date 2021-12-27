#!/usr/bin/python3
import time
#import board
#import busio
import sys
import cv2

import threading
import inspect
import os
from traitlets import HasTraits, Int, Unicode, default, observe

from robot.ps_control.ps_control import ps_controller 
from robot.ser_control.ser_control import ser_controller
from robot.ser_control.ser_control import cmd

import signal

# Import the HT16K33 LED segment module.
#from adafruit_ht16k33 import segments

'''
def sigint_handler(signal, frame):
    print('sg Interrupted')
    time.sleep(0.1)
    pid=os.getpid()
    os.system(f'kill -9 {pid}')
    print('kill process')
'''


class robot_controller(threading.Thread, HasTraits):

    angle = 0
    speed = 0
    
    def __init__(self, **kwargs):

       signal.signal(signal.SIGINT, self.sigint_handler)


    def sigint_handler(self, signal, frame):

        print('sg Interrupted')
        print('move(0,0)')
        self.move(0,0)

        time.sleep(0.1)
        pid=os.getpid()
        os.system(f'kill -9 {pid}')
        print('kill process')


    def segment(self, str, colon ):
        str = list(str)
        print(str)
        # Create the I2C interface.
        i2c = busio.I2C(board.SCL, board.SDA)
        print('craate i2c')

        display = segments.Seg7x4(i2c)

        for i in range(4):
            display[i] = str[i]
        
        display.colon = colon

    def hello(self):
        print('hello, I am bot')

    def pin_input(self, pin_no):
        return True

    def img_dis_on(self, img, type):
        if type == 'color' :
            img = cv2.imread(img, cv2.IMREAD_COLOR )
        elif type == 'mono' :
            img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

        cv2.imshow('img_show', img)
        cv2.waitKey(1)

    def img_dis_off(self):
        cv2.destroyAllWindows()

    def cam_dis_on(self):
        pass

    def cam_dis_off(self):
        pass

    def cam_recod_on(self):
        pass

    def cam_recod_off(self):
        pass

    def img_capture(self, name):
        pass


    #//-800 ~ 800, -800 ~ 800
    def move(self,angle,speed):
        if self.angle == angle and self.speed == speed :
            return
        else:
            cmd(angle, speed)
            self.angle = angle
            self.speed = speed
        
    def delay(self, sec):
        time.sleep(sec)



#===========================================

import signal
import sys
import os

'''
def sigint_handler(signal, frame):
    print('sg Interrupted')
    pid=os.getpid()
    os.system(f'kill -9 {pid}')
    print('kill process')
'''

#signal.signal(signal.SIGINT, sigint_handler)


#Test part
#if __name__ == "__main__" :
#     ps_ctl = ps4_control()
#     while True:
#         time.sleep(1)
#         sts = ps_ctl.key_sts
#         print(f'key sts = {sts}')

if __name__ == "__main__" :

    robo = robot_controller()

    while True :

        time.sleep(1)
        print('loop')
