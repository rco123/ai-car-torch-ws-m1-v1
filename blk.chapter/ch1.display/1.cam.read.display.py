#!/usr/bin/python3
from robot import ai_controller
from robot import robot_controller

robo = robot_controller()
ai = ai_controller()

ai.cam_open()

for i in range(100):
    print(f'for cnt i = {i}')
    ai.cam_img_read()
    ai.img_display()
    robo.delay(0.1)
