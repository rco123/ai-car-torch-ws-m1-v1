#!/usr/bin/python3

from robot import ps_controller
from robot import robot_controller

robo = robot_controller()
ps = ps_controller()

while True:
    
    kevent = ps.check_event()
    print(f'kevent check = {kevent}')

    if kevent == True:
        key = ps.key_read()
        print(f'get key = {key}')
        
        if key == 1 : 
            robo.move(100,0)
        if key == 2 :
            robo.move(-100,0) 
        if key == 9 : 
            print('set angle = 0, speed = 0')
            robo.move(0,0) 
        if key == 7 : 
            print('exit program')
        
    robo.delay(0.1)
