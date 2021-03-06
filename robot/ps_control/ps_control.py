import time
import serial
import cv2
import threading
from queue import Queue
import inspect
import os
from traitlets import HasTraits, Int, Unicode, default, observe
from pyPS4Controller.controller import Controller
from robot.ser_control.ser_control import cmd, m_angle, m_speed

#mode 1 Key Map
class ps_controller(Controller, threading.Thread, HasTraits):

    key_sts = Int()
    key_event = False
    xdebug = False

    angle=0
    speed=0

    def __init__(self, **kwargs):

        Controller.__init__(self, interface='/dev/input/js0', connecting_using_ds4drv=False)
        threading.Thread.__init__(self)

        self.key_sts = 0
        self.start()

        t = threading.Thread( target=self.angle_speed_mng )
        t.start()

    def angle_speed_mng(self):
        while True:
            ainc = 10; binc= 2
            if self.key_sts == 1 : # left click state
                self.angle += ainc
                if self.angle > 800 : self.angle = 800
            if self.key_sts == 2 :
                self.angle -= ainc
                if self.angle < -800: self.angle = -800

            if self.key_sts == 4 : 
                self.speed += binc
                if self.speed > 1000 : self.speed = 1000
            if self.key_sts == 5 :
                self.speed -= binc
                if self.speed < -1000 : self.speed = -1000
            #print(f'a s {self.angle}, {self.speed}')
            time.sleep(0.01)

    def mynameis(self):
        if self.xdebug:
            print('-> ' + inspect.stack()[1][3])
            print(f'key = {self.key_sts}')
        else:
            pass

    @observe('key_sts')
    def _observe_key_sts(self, change):
        #print(change['old'])
        #print(change['new'])
        self.key_event = True

    def check_event(self):
        return self.key_event

    def key_read(self):
        self.key_event = False
        return self.key_sts

    def read_angle(self):
        return self.angle
    def read_speed(self):
        return self.speed

    def run(self):
        print('start process')
        self.listen()

        
    #def on_L3_left(self, value): #  no 1 key go left command
    def on_left_arrow_press(self): #  no 1 key go left command
        self.key_sts = 1
        self.key_event = True
        self.mynameis()

    def on_right_arrow_press(self): # no 2 go key right command
        self.key_sts =  2
        self.key_event = True
        self.mynameis()

    def on_left_right_arrow_release(self):    # no 1,2, release relf right button clear
        self.key_sts = 3
        self.key_event = True
        self.mynameis()

    def on_up_arrow_press(self): #  no 3 key head go speed up  ahead command
        self.key_sts = 4
        self.key_event = True
        self.mynameis()

    def on_down_arrow_press(self): # no 4 key back word go slow  down command
        self.key_sts = 5
        self.key_event = True
        self.mynameis()

    def on_up_down_arrow_release(self): #no 3,4 release
        self.key_sts = 6
        self.key_event = True
        self.mynameis()

    ################### Button is changing time to tim ###################

    #number 7
    # def on_circle_press(self): #  b tutton using stop command
    #     self.key_sts = 7
    #     self.key_event = True
    #     self.mynameis()

    #     self.angle = 0
    #     self.speed = 0
        
    # def on_circle_release(self): #  b tutton using stop command
    #     self.key_sts = 71
    #     self.key_event = True
    #     self.mynameis()

    #     cmd(0,0)
    #     pid = os.getpid()
    #     os.system(f'kill -9 {pid}')
    #     cmd(0,0)
    #     print('kill process')

    def on_triangle_press(self): # Y button start ai_controller
        self.key_sts = 7
        self.key_event = True
        self.mynameis()

    def on_triangle_release(self): # Y button start ai_controller  release
        self.key_sts = 71
        self.key_event = True
        self.mynameis()

    #number 8
    def on_square_press(self): # X button click ai_controller stop
        self.key_sts = 8
        self.key_event = True
        self.mynameis()

    def on_square_release(self): # X button release ai_controller stop
        self.key_sts = 81
        self.key_event = True
        self.mynameis()


    #number 9
    # def on_x_press(self): # A button click ai_controller stop
    #     self.key_sts = 9
    #     self.key_event = True
    #     self.mynameis()
        
    #     # angle and speed inital 
    #     #self.angle = 0
    #     angle = m_angle
    #     # cmd(angle,0) # get the current angle

    # def on_x_release(self): # A button release ai_controller stop
    #     self.key_sts = 91
    #     self.key_event = True
    #     self.mynameis()

    def on_circle_press(self): # X button click ai_controller stop
        self.key_sts = 9
        self.key_event = True
        self.mynameis()

    def on_circle_release(self): # X button release ai_controller stop
        self.key_sts = 91
        self.key_event = True
        self.mynameis()

    #number 10
    def on_x_press(self): # Y button start ai_controller
        self.key_sts = 10
        self.key_event = True
        self.mynameis()

    def on_x_release(self): # Y button start ai_controller  release
        self.key_sts = 11
        self.key_event = True
        self.mynameis()



if __name__ == "__main__":

    js_fd = "/dev/input/js0"
    controller = ps_controller(interface=js_fd, connecting_using_ds4drv=False)
    controller.start()

    while True:
        try:
            if  controller.key_sts == 71 :
                pid = os.getpid()
                os.system(f'kill -9 {pid}')
                print('kill process')
                break
            time.sleep(1)

        except KeyboardInterrupt:
            print('KeyboardInterrupt')
            pid=os.getpid()
            os.system(f'kill -9 {pid}')
            print('kill process')
            break

