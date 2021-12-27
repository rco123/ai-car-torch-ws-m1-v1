#!/usr/bin/python3

from robot import ps_controller
from robot import robot_controller

robo = robot_controller()
ps = ps_controller()

while True:
    
    kevent = ps.check_event()
    if kevent :
        print(f'kevent check = {kevent}')

    if kevent == True:
        key = ps.key_read()
        if key == 1 : 
            print(f'get key = {key}')
            robo.move(100,0)
        if key == 2 :
            print(f'get key = {key}')
            robo.move(-100,0) 
        if key == 7 : 
            print(f'get key = {key}')
            print('break')
            break
        
    robo.delay(0.01)

