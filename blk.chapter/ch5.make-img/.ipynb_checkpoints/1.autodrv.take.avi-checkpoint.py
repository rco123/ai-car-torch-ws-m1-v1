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
import cv2

ps = ps_controller()
robo = robot()
ai = ai_controller()
ai.cam_open()
speed = 200
gostop = 'stop'
cnt = 0
skip_no = 10 # 1/10 take image
skip_cnt = 0


fourcc = cv2.VideoWriter_fourcc(*'XVID')
image_size = (640,480)
str = './vids/out.avi'
vid_out = cv2.VideoWriter(str , fourcc, 20.0, image_size)


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
            vid_out.release()

    ai.cam_img_get()
    ai.img_display()

    if gostop == 'go':
        vid_out.write(ai.img) 
        angle = ai.cam_img_to_angle()
        robo.move(angle, speed)
        print(f'{angle}, {speed}')

    robo.delay(0.01)

