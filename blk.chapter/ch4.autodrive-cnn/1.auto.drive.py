#!/usr/bin/python3
from robot import ps_controller
from robot import ai_controller
from robot import robot_controller
import time

ps = ps_controller()
robo = robot_controller()
ai = ai_controller()

ai.cam_open()
ai.lane_det_cnn_load_model('lane_model.h5')

while True:

    speed = 200

    ai.cam_img_read()

    angle = ai.cam_img_to_angle_cnn()
    robo.move(angle, speed )

    print(angle, speed)

    ai.img_display()
    robo.delay(0.01)

    
