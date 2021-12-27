#!/usr/bin/python3
from robot import ai_controller
from robot import robot_controller

robo = robot_controller()
ai = ai_controller()

print('Display Image')

ai.img_read('imgs/aicar.jpg')

while True:
    ai.img_display()
