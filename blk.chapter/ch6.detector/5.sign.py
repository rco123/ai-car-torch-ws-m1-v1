#!/usr/bin/python3
from robot import ps_controller
from robot import ai_controller
from robot import robot_controller

ps = ps_controller()
robo = robot_controller()
ai = ai_controller()
ai.cam_open()

ai.traffic_sign_load_model('model_trained.h5')

while True:

    ai.cam_img_read()

    rtn = ai.traffic_sign_detector(75)

    if rtn >= 0:
        print(f'rtn value = {rtn}')
        if rtn == 0 :
            print('speed 30Km')
        if rtn == 1:
            print('speed 50Km')


    ai.img_display()

    robo.delay(0.1)

