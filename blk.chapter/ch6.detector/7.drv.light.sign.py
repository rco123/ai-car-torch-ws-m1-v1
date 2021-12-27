#!/usr/bin/python3

from robot import ps_controller
from robot import ai_controller
from robot import robot_controller

ps = ps_controller()
robo = robot_controller()
ai = ai_controller()
ai.cam_open()

ai.traffic_sign_load_model('model_trained.h5')

speed = 0
angle = 0
gostop = 'stop'

while True:

    kevent = ps.check_event()

    if kevent == True:
        key = ps.key_read()

        if key == 7 :
            robo.delay(1)
        if key == 8 :
            gostop = 'go' 
            speed = 200
        if key == 9 or key == 91 :
            gostop = 'stop'
            speed = 0
            robo.move(angle, speed)


    ai.cam_img_read()
    rtn = ai.traffic_sign_detector(75)
    if rtn >= 0:
        if gostop == 'go':
            print(f'sign rtn value = {rtn}')
            if rtn == 0 :
                speed = 150
            if rtn == 1:
                speed = 200


    rtn = ai.traffic_light_detector(75)
    if rtn >= 0:
        print(f'light rtn value = {rtn}')
        if rtn == 3 :
            gostop = 'stop'
            robo.move(angle,0)
        else:
            gostop = 'go'


    ai.img_display()

    if gostop == 'go' :           
        angle = ai.cam_img_to_angle_m3(75)
        print(f'get_angle = {angle}')
        print(f'{angle}, {speed}')
        robo.move(angle, speed )
 
    robo.delay(0.01)

