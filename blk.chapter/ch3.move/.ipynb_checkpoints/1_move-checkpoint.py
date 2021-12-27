#!/usr/bin/python3 
from robot import Ps4_controller
from robot import Robot


ps = Ps4_controller()
robo = Robot()

while True:

    angle = ps.read_angle()
    speed = ps.read_speed()

    print(f'angle, speed = {angle}, {speed}')
    
    robo.move(angle, speed )

    robo.delay(0.01)

    




