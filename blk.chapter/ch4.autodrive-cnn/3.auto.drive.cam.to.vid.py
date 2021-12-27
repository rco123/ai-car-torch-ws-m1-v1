#!/usr/bin/python3

kevent = None
key = None
speed = None
gostop = None
cnt = None
angle = None


from robot import ps_controller
from robot import ai_controller
from robot import robot_controller

import cv2
import os

ps = ps_controller()
robo = robot_controller()
ai = ai_controller()
ai.cam_open()

ai.lane_det_cnn_load_model('lane_model.h5')

speed = 200
gostop = 'stop'
cnt = 0

fourcc = cv2.VideoWriter_fourcc(*'XVID')
str = os.path.join(os.getcwd(),'vids','output.avi')
vid_out = cv2.VideoWriter(str, fourcc, 20.0, (1280,720))


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

            vid_out.release()
            robo.move(angle, speed)


    ai.cam_img_read()
    ai.img_display()

    if gostop == 'go':
        angle = ai.cam_img_to_angle_cnn()
        #ai.img_write((f'./imgs/{cnt:03d}_{angle:03d}.jpg'))
        #ai.img_write((f'./imgs/{cnt:03d}.jpg'))
        #cnt = cnt + 1
        vid_out.write(ai.img)

        robo.move(angle, speed)
        print(f'{angle}, {speed}')
    robo.delay(0.01)




