#!/usr/bin/python3
from robot import ps_controller
from robot import ai_controller
from robot import robot_controller
import time

ps = ps_controller()
robo = robot_controller()
ai = ai_controller()

ai.cam_open()

while True:

    #ai.img_read('1.jpg')
    ai.cam_img_read()
    ai.cam_img_to_angle_m3_check(75)
    robo.delay(0.01)

