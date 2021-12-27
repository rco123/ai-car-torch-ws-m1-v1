#!/usr/bin/python3
from robot import ai_controller
from robot import ps_controller
from robot import robot_controller

robo = robot_controller()
ai = ai_controller()
ps = ps_controller()

ai.cam_open()

speed = 200
cnt = 0
while True:

    ai.cam_img_read()
    ai.img_display()

    angle = ps.read_angle()
    robo.move(angle,speed)


    file = f'./imgs/{cnt:03d}.jpg'
    print(file)
    ai.img_write(file)
    cnt += 1

    robo.delay(0.01)

