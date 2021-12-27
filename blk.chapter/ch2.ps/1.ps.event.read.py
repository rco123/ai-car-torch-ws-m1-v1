#!/usr/bin/python3

from robot import ps_controller
from robot import robot_controller

robo = robot_controller()
ps = ps_controller()

while True:

    kevent = ps.check_event()
    if kevent == True:
        print(f'kevent check = {kevent}')

    if kevent == True:
        key = ps.key_read()
        print(f'read key value = {key}')

    
    robo.delay(0.1)
