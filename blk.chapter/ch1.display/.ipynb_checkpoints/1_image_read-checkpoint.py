#!/usr/bin/python3
from robot import Ai_controller
from robot import Robot

robo = Robot()
ai = Ai_controller()

for i in range(10):
    print(f'for cnt i = {i}')
    ai.img_read('lena.jpg')
    ai.img_display()
    robo.delay(1)
