#!/usr/bin/python3
from robot import ai_controller
from robot import robot_controller

robo = robot_controller()
ai = ai_controller()

print('read Lena Image')
ai.img_read('./imgs/lena.jpg')

print('Write Image')
ai.img_write('./imgs/lena_1.jpg')

print('Display Image')
while True:
    ai.img_display()

