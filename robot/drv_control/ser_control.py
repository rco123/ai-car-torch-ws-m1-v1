
import time
import serial
import cv2
import threading
from queue import Queue

FRAME_HEADER = 0X7B
FRAME_TAIL = 0X7D

m_angle = 0
m_speed = 0

def Check_Sum(Count_Number, Send_Data):
    check_sum = 0x00 
    for k in range(Count_Number):
        check_sum = check_sum ^ Send_Data[k]
    return check_sum

def open_serial():
    ser = serial.Serial(port = '/dev/scmd',
    baudrate=115200, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE,
    bytesize=serial.EIGHTBITS, timeout = 0)
    return ser

def close_serial(ser):
    ser.close()

def cmd(angle, speed):

    Send_Data = [0 for i in range(11)]
    transition = 0

    Send_Data[0]=FRAME_HEADER
    Send_Data[1] = 1
    Send_Data[2] = 0

    transition=0;
    transition =   int(speed )
    Send_Data[4] = int(transition & 0x00FF  )
    Send_Data[3] = int((transition >> 8 ) &0xFF)

    Send_Data[5] = 0
    Send_Data[6] = 0

    transition=0;
    transition = int(angle//2)

    Send_Data[8] = int(transition & 0xFF)
    Send_Data[7] = int((transition >> 8 )& 0xFF)

    Send_Data[9]= Check_Sum(9, Send_Data )
    Send_Data[10]=FRAME_TAIL

    m_angle = angle; m_speed = speed
    ser = open_serial()
    ser.write( bytearray(Send_Data))
    close_serial(ser)
    

class ser_controller(threading.Thread):

    que = None
    ser = None
    port = '/dev/ttyUSB0'

    def __init__(self,port):
        threading.Thread.__init__(self)
        self.port  = port
      
        #print(f'connect serial port ={self.port}')
        # self.ser = serial.Serial(port,
        # baudrate=115200, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE,
        # bytesize=serial.EIGHTBITS, timeout = 0)

    def __del__(self):
        pass
        #print('close serial')
        #self.ser.close()

    def open_serial(self):
        self.ser = serial.Serial(self.port,
        baudrate=115200, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS, timeout = 0)

    def close_serial(self):
        self.ser.close()


    def Check_Sum(self, Count_Number, Send_Data):
        check_sum = 0x00 
        for k in range(Count_Number):
            check_sum = check_sum ^ Send_Data[k]
        return check_sum


    def cmd(self, angle, speed):

        self.open_serial()

        Send_Data = [0 for i in range(11)]
        transition = 0

        Send_Data[0]=FRAME_HEADER
        Send_Data[1] = 1
        Send_Data[2] = 0

        transition=0;
        transition =   int(speed )
        Send_Data[4] = int(transition & 0x00FF  )
        Send_Data[3] = int((transition >> 8 ) &0xFF)

        Send_Data[5] = 0
        Send_Data[6] = 0

        transition=0;
        print(f'angle = {angle}')
        transition = int(angle//2)

        Send_Data[8] = int(transition & 0xFF)
        Send_Data[7] = int((transition >> 8 )& 0xFF)

        Send_Data[9]= self.Check_Sum(9, Send_Data )
        Send_Data[10]=FRAME_TAIL

        self.ser.write( bytearray(Send_Data))
        
        self.ser.close()


if __name__ == '__main__' :

    ps_ser = ser_controller('/dev/scmd')
    #ps_ser = ser_controller('/dev/ttyUSB1')
    while True :

        key = input('enter key value 0 ~ 9 = ')
        print('key value = %d' % int(key))

        speed = 100 * int(key)
        angle = 200 
        print('speed = %f, angle = %f' % (speed, angle))
        ps_ser.cmd(angle, speed)

