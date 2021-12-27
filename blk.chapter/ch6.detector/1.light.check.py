#!/usr/bin/python3

from robot import ps_controller
from robot import ai_controller
from robot import robot_controller

ps = ps_controller()
robo = robot_controller()
ai = ai_controller()

ai.cam_open()

while True:

    ai.cam_img_read()

    ai.traffic_light_detector_check(75)

    robo.delay(0.01)


