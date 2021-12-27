from robot import ai_controller
from robot import robot

robo = robot()
ai = ai_controller()

for i in range(10):
    print(f'for cnt i = {i}')
    ai.img_read('lena.jpg')
    ai.img_display()
    robo.delay(1)
