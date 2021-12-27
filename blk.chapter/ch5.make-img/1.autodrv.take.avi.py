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
#!/usr/bin/python3

ps = ps_controller()
robo = robot_controller()
ai = ai_controller()
ai.cam_open()
speed = 200
gostop = 'stop'
cnt = 0
skip_no = 10 # 1/10 take image
skip_cnt = 0


ai.cam_img_to_avi_open('./vids/out.avi')


while True:
    kevent = ps.check_event()
    if kevent == True:
        key = ps.key_read()
        if key == 7:
            key = ps.key_read()
            robo.delay(1)
        if key == 8:
            gostop = 'go'
            speed = 150
        if key == 9:
            gostop = 'stop'
            speed = 0
            robo.move(angle, speed)
            ai.cam_img_to_avi_release()

    ai.cam_img_read()
    ai.img_display()

    if gostop == 'go':

        ai.cam_img_to_avi_write()

        angle = ai.cam_img_to_angle_m3(100)

        robo.move(angle, speed)
        print(f'{angle}, {speed}')

    robo.delay(0.01)

