#!/usr/bin/python3
from robot import Ps4_controller
from robot import Ai_controller
from robot import Robot
import time

ps = Ps4_controller()
robo = Robot()
ai = Ai_controller()

ai.cam_open()

speed = 200
gostop = 'stop'

while True:

    kevent = ps.check_event()
    #print(f'kevent check = {kevent}')

    if kevent == True:
        key = ps.key_read()

        if key == 7 :
            robo.delay(1)

        if key == 8 :
            gostop = 'go' 
            speed = 200
        if key == 9 :
            gostop = 'stop'
            speed = 0
            robo.move(angle, speed )

    ai.cam_img_get()
    ai.img_display()

    if gostop == 'go' :           
        angle = ai.cam_img_to_angle()
        robo.move(angle, speed )
        print(f'{angle}, {speed}')
 
    robo.delay(0.01)
