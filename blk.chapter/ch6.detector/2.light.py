#!/usr/bin/python3
from robot import ps_controller
from robot import ai_controller
from robot import robot_controller

robo = robot_controller()
ai = ai_controller()

ai.cam_open()

while True:

    ai.cam_img_read()
    rtn = ai.traffic_light_detector(75)
    if rtn >= 0:
        print(f'==>rtn= {rtn}')
        if rtn == 0 :
            print('no light')
        elif rtn == 1:
            print('green light')
        elif rtn == 2 :
            print('yellow light')
        if rtn == 3 :
            print('red light')
            
    ai.img_display()
    robo.delay(0.01)

