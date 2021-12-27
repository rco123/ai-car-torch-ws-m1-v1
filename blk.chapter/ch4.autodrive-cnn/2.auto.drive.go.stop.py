from robot import ps_controller
from robot import ai_controller
from robot import robot
import time

ps = ps_controller()
robo = robot()
ai = ai_controller()

ai.cam_open()
ai.lane_det_cnn_load_model('lane_model.h5')

speed = 200
angle = 0
gostop = 'stop'

while True:

    kevent = ps.check_event()
    #print(f'kevent check = {kevent}')

    if kevent == True:
        key = ps.key_read()

        if key == 7 :
            robo.delay(1)

        if key == 8 :
            gostop = 'go' 
            speed = 200
        if key == 9 :
            gostop = 'stop'
            speed = 0
            robo.move(angle, speed )

    ai.cam_img_read()
    ai.img_display()

    if gostop == 'go' :           
        angle = ai.cam_img_to_angle_cnn()
        print(f'get_angle = {angle}')
        print(f'{angle}, {speed}')
        robo.move(angle, speed )
 

    robo.delay(0.01)

