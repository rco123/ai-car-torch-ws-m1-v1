#!/usr/bin/python3 
from robot import ps_controller
from robot import ai_controller
from robot import robot_controller

ps = ps_controller()
robo = robot_controller()
ai = ai_controller()

ai.cam_open()

while True:

    angle = ps.read_angle()
    speed = ps.read_speed()

    ai.cam_img_read()
    ai.img_display()

    print(f'angle, speed = {angle}, {speed}')

    robo.move(angle, speed )
  

    robo.delay(0.01)

