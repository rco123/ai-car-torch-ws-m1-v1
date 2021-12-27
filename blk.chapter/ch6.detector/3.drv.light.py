from robot import ps_controller
from robot import ai_controller
from robot import robot_controller
import time

ps = ps_controller()
robo = robot_controller()
ai = ai_controller()

ai.cam_open()

speed = 200
pre_speed = 200
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
    rtn = ai.traffic_light_detector(75)
    if rtn >= 0:
        print(f'==>rtn= {rtn}')
        if rtn == 3 :
            gostop = 'stop'
            robo.move(angle, 0 )
        else:
            gostop = 'go'


    ai.img_display()

    if gostop == 'go' :           
        angle = ai.cam_img_to_angle_m3(75)
        print(f'get_angle = {angle}')
        print(f'{angle}, {speed}')
        robo.move(angle, speed )
 

    robo.delay(0.01)

