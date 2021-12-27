#!/usr/bin/python3 
from robot import ps_controller
from robot import robot_controller

ps = ps_controller()
robo = robot_controller()

while True:

    angle = ps.read_angle()
    speed = ps.read_speed()

    print(f'angle, speed = {angle}, {speed}')
    
    robo.move(angle, speed )
    robo.delay(0.01)

    

