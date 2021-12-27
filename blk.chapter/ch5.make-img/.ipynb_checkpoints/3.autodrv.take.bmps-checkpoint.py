#!/usr/bin/python3

kevent = None
key = None
speed = None
gostop = None
cnt = None
angle = None

from robot import ps_controller
from robot import ai_controller
from robot import robot

ps = ps_controller()
robo = robot()
ai = ai_controller()
ai.cam_open()
speed = 200
gostop = 'stop'
cnt = 0
skip_no = 2 # 1/10 take image
skip_cnt = 0

while True:
    kevent = ps.check_event()
    if kevent == True:
        key = ps.key_read()
        if key == 7:
            key = ps.key_read()
            robo.delay(1)
        if key == 8:
            gostop = 'go'
            speed = 200
        if key == 9:
            gostop = 'stop'
            speed = 0
            robo.move(angle, speed)

    ai.cam_img_get()
    ai.img_display()

    if gostop == 'go':
        if skip_cnt == 0 :
            ai.img_write((f'./imgs/{cnt}.bmp'))
            cnt = cnt + 1

        skip_cnt += 1
        if skip_cnt == skip_no : skip_cnt = 0

        angle = ai.cam_img_to_angle()
        robo.move(angle, speed)
        print(f'{angle}, {speed}')

    robo.delay(0.01)

