#!/usr/bin/python3
from robot import ps_controller
from robot import ai_controller
from robot import robot
import time

ps = ps_controller()
robo = robot()
ai = ai_controller()

ai.cam_open()

while True:

    speed = 200

    ai.cam_img_get()
    angle = ai.cam_img_to_angle()

    robo.move(angle, speed )

    ai.img_display()

    robo.delay(0.01)

    
