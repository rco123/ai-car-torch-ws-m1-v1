
import cv2
import numpy as np
import logging
import math
import datetime
import sys

_SHOW_IMAGE = False

class HandCodedLaneFollower(object):

    def __init__(self, car=None):
        logging.info('Creating a HandCodedLaneFollower...')
        self.car = car
        self.curr_steering_angle = 0
        self.img_cnt = 0

    def follow_lane(self, frame):
        # Main entry point of the lane follower
        show_image("orig", frame)

        self.detect_lane(frame)
        diff = self.curr_steering_angle

        print(frame.shape)
        height = frame.shape[0]
        width = frame.shape[1]

        cv2.imwrite(f'./imgs/im_{self.img_cnt:03d}.jpg',frame)
        self.img_cnt += 1

        
        print(f'he,w ={diff}, {height}, {width}')
        cv2.line(frame ,(  int(width/2), height), ( int(width/2 + diff), int(height/2)), (255,0,0), 3)
        show_image('line',frame, True)

        return frame


    def detect_lane(self,frame):

        frame = cv2.blur(frame, (5, 5))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        show_image("hsv", hsv)
        lower_blue = np.array([0, 0, 0])
        upper_blue = np.array([255, 255, 100])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        show_image("blue mask", mask)
        height = mask.shape[0]
        width = mask.shape[1]

        polygon_1 = np.array([[
            (0, height * 2 /4),
            (width, height * 2 / 4),
            (width, height * 3 / 4),
            (0, height * 3 / 4),
        ]], np.int32)

        polygon_2 = np.array([[
            (0, height * 3 / 4),
            (width, height * 3 / 4),
            (width, height * 4 / 4),
            (0, height * 4 / 4),
        ]], np.int32)

        region_1 = region_of_interest(mask, polygon_1)
        region_2 = region_of_interest(mask, polygon_2)

        show_image('region_1', region_1) # UP
        show_image('region_2', region_2, True) # DOWN

        his_values = np.sum(region_2, axis=0) # check bottom
        #print(his_values)

        value_average = np.average(his_values)
        print(f'value_average = {value_average}')
        if value_average == 0 :
            return

        index = np.where(his_values > int(value_average))
        #print(index)


        reg = 0
        cnt = 0
        loc = 0
        for i, v in enumerate(index[0]):
            #print(f'i, v, = {i}, {v}')
            if i == 0 :
                reg = v
            else:
                if abs(reg - v) > 200 :
                    cnt +=1
                    print('cnt up')
                    loc = i
                reg = v
        print(f'cnt = {cnt}')

        if cnt == 1 :
            l_index = index[0][:loc]
            r_index = index[0][loc:]

            l_point = int(np.average(l_index))
            r_point = int(np.average(r_index))
            print(f'l, r = {l_point}, {r_point}')
            base_point = (l_point + r_point )/ 2
            print(f'l base_point = {base_point}')

            diff = int(base_point - frame.shape[1]/2)
            self.curr_steering_angle =  diff

        if cnt == 0:

            his_up_values = np.sum(region_2, axis=0)
            #print(his_up_values)

            value_average = np.average(his_up_values)

            print(f'value_average = {value_average}') 
            #if average == 0 , do not anything, return pre base point
            if int(value_average) == 0 :
                return 


            index = np.where(his_values > int(value_average))
            #print(index)
            point1 = int(np.average(index))

            polygon_3 = np.array([[
                (0, height * 1 / 2),
                (width, height * 1 / 2),
                (width, height ),
                (0, height ),
            ]], np.int32)

            region_3 = region_of_interest(mask, polygon_3)
            edges_img = cv2.Canny(region_3, 200, 400)
            lines_segment = detect_line_segments(edges_img)

            if lines_segment is not None:
                dir = average_slope_intercept(lines_segment)
            else:
                print('lines_segment = None')
                return

            point_dist = 500
            if dir > 0 :
                print('left line')
                base_point = (point1 + (point1 + point_dist)) / 2
            else:
                print('right line')
                base_point = (point1 - (point_dist - point_dist)) / 2

            print(f'base_point = {base_point}')
            diff = int(base_point - frame.shape[1]/2)
            diff =  stabilize_steering_angle(self.curr_steering_angle, diff, cnt, max_angle_deviation= 20)
            self.curr_steering_angle =  diff

        return


############################
# Frame processing steps
############################


def detect_edges(frame):
    # filter for blue lane lines

    frame = cv2.blur(frame, (5, 5))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    show_image("hsv", hsv)
    lower_blue = np.array([0, 0, 0])
    upper_blue = np.array([255, 255, 100])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    show_image("blue mask", mask)
    # detect edges
    # edges_img = cv2.Canny(mask, 200, 400)
    # show_image('edged_img', edges_img, True)
    # return edges_img
    return mask

def detect_line_area(frame):
    # filter for blue lane lines

    frame = cv2.blur(frame, (5, 5))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    show_image("hsv", hsv)
    lower_blue = np.array([0, 0, 0])
    upper_blue = np.array([255, 255, 100])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    show_image("blue mask", mask)
    # detect edges
    # edges_img = cv2.Canny(mask, 200, 400)
    # show_image('edged_img', edges_img, True)
    # return edges_img
    return mask

def region_of_interest(frame, polygen):
    height, width = frame.shape
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, polygen, 255)
    masked_image = cv2.bitwise_and(frame, mask)
    show_image("mask", masked_image)
    return masked_image


def detect_line_segments(cropped_edges):
    # tuning min_threshold, minLineLength, maxLineGap is a trial and error process by hand
    rho = 1  # precision in pixel, i.e. 1 pixel
    angle = np.pi / 180  # degree in radian, i.e. 1 degree
    min_threshold = 10  # minimal of votes
    line_segments = cv2.HoughLinesP(cropped_edges, rho, angle, min_threshold, np.array([]), minLineLength=20,
                                    maxLineGap=5)

    if line_segments is not None:
        for line_segment in line_segments:
            logging.debug('detected line_segment:')
            logging.debug("%s of length %s" % (line_segment, length_of_line_segment(line_segment[0])))
            # print('detected line_segment:')
            # print("%s of length %s" % (line_segment, length_of_line_segment(line_segment[0])))

    return line_segments


def average_slope_intercept(line_segments):

    left_fit = []
    right_fit = []

    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:
                logging.info('skipping vertical line segment (slope=inf): %s' % line_segment)
                continue
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))

    if len(left_fit) > len(right_fit) :
        return 1
    else :
        return 0


def compute_steering_angle(frame, lane_lines):
    """ Find the steering angle based on lane line coordinate
        We assume that camera is calibrated to point to dead center
    """
    x_offset = 0; y_offset = 0
    if len(lane_lines) == 0:
        logging.info('No lane lines detected, do nothing')
        return -90

    height, width, _ = frame.shape
    if len(lane_lines) == 1:
        logging.debug('Only detected one lane line, just follow it. %s' % lane_lines[0])
        print('Only detected one lane line, just follow it. %s' % lane_lines[0])
        x1, y1, x2, y2 = lane_lines[0][0]
        x_offset = x2 - x1
        print(f'x_offset = {x_offset}')
        y_offset = y1 - y2
    else:
        _, _, left_x2, _ = lane_lines[0][0]
        _, _, right_x2, _ = lane_lines[1][0]
        camera_mid_offset_percent = 0.02  # 0.0 means car pointing to center, -0.03: car is centered to left, +0.03 means car pointing to right
        mid = int(width / 2 * (1 + camera_mid_offset_percent))
        x_offset = (left_x2 + right_x2) / 2 - mid
        print(f'two line offset = {x_offset}')
        # find the steering angle, which is angle between navigation direction to end of center line
        y_offset = int(height / 2)


    angle_to_mid_radian = math.atan(x_offset / y_offset)  # angle (in radian) to center vertical line
    angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)  # angle (in degrees) to center vertical line
    print(f'angle_to_deg = {angle_to_mid_deg}')
    steering_angle = angle_to_mid_deg + 90  # this is the steering angle needed by picar front wheel

    logging.debug('new steering angle: %s' % steering_angle)
    print('new steering angle: %s' % steering_angle)
    return steering_angle


'''
def stabilize_steering_angle(curr_steering_angle, new_steering_angle, num_of_lane_lines,
                             max_angle_deviation_two_lines=5, max_angle_deviation_one_lane= 5):
    """
    Using last steering angle to stabilize the steering angle
    This can be improved to use last N angles, etc
    if new angle is too different from current angle, only turn by max_angle_deviation degrees
    """
    if num_of_lane_lines == 2:
        # if both lane lines detected, then we can deviate more
        max_angle_deviation = max_angle_deviation_two_lines
        print(f'two line max_angle_deviation = {max_angle_deviation}')
    else:
        # if only one lane detected, don't deviate too much
        max_angle_deviation = max_angle_deviation_one_lane
        print(f'one line max_angle_deviation = {max_angle_deviation}')
    angle_deviation = new_steering_angle - curr_steering_angle
    if abs(angle_deviation) > max_angle_deviation:

        stabilized_steering_angle = int(curr_steering_angle
                                        + max_angle_deviation * angle_deviation / abs(angle_deviation))
    else:
        stabilized_steering_angle = new_steering_angle

    logging.info('Proposed angle: %s, stabilized angle: %s' % (new_steering_angle, stabilized_steering_angle))
    return stabilized_steering_angle
'''

def stabilize_steering_angle(curr_steering_angle, new_steering_angle, num_of_lane_lines, max_angle_deviation= 5):
    """
    Using last steering angle to stabilize the steering angle
    This can be improved to use last N angles, etc
    if new angle is too different from current angle, only turn by max_angle_deviation degrees
    """
    if num_of_lane_lines == 2:
        return new_steering_angle
    else:

        angle_deviation = new_steering_angle - curr_steering_angle

    if abs(angle_deviation) > max_angle_deviation:

        stabilized_steering_angle = int(curr_steering_angle
                                        + max_angle_deviation * angle_deviation / abs(angle_deviation))
    else:
        stabilized_steering_angle = new_steering_angle

    logging.info('Proposed angle: %s, stabilized angle: %s' % (new_steering_angle, stabilized_steering_angle))
    return stabilized_steering_angle



############################
# Utility Functions
############################
def display_lines(frame, lines, line_color=(0, 255, 0), line_width=1):
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return line_image


def display_heading_line(frame, steering_angle, line_color=(0, 0, 255), line_width=5, ):
    heading_image = np.zeros_like(frame)
    height, width, _ = frame.shape

    # figure out the heading line from steering angle
    # heading line (x1,y1) is always center bottom of the screen
    # (x2, y2) requires a bit of trigonometry

    # Note: the steering angle of:
    # 0-89 degree: turn left
    # 90 degree: going straight
    # 91-180 degree: turn right
    steering_angle_radian = steering_angle / 180.0 * math.pi
    x1 = int(width / 2)
    y1 = height
    x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
    y2 = int(height / 2)


    cv2.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)
    heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)

    return heading_image


def length_of_line_segment(line):
    x1, y1, x2, y2 = line
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def show_image(title, frame, show=_SHOW_IMAGE):
    if show:
        cv2.imshow(title, frame)


def make_points(frame, line):
    height, width, _ = frame.shape
    slope, intercept = line
    y1 = height  # bottom of the frame
    y2 = int(y1 * 1 / 2)  # make points from middle of the frame down

    # bound the coordinates within the frame
    x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
    return [[x1, y1, x2, y2]]


############################
# Test Functions
############################
def test_photo(file):
    land_follower = HandCodedLaneFollower()
    frame = cv2.imread(file)
    combo_image = land_follower.follow_lane(frame)
    show_image('final', combo_image, True)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_imgs():

    lane_follower = HandCodedLaneFollower()

    # cap = cv2.VideoCapture(video_file)
    # skip first second of video.
    # for i in range(3):
    #     _, frame = cap.read()
    # video_type = cv2.VideoWriter_fourcc(*'XVID')
    # fname = video_file.split('.')[0]
    # video_overlay = cv2.VideoWriter(f'{fname}_ovray.avi', video_type, 20.0, (320, 240))

    files = os.listdir('./imgs')
    for file in files:
        i = 0
        frame = cv2.imread('./imgs/' + file )
        print(f'frame {i}, {file}')

        frame = frame[:,100:,:]
        combo_image = lane_follower.follow_lane(frame)

        # cv2.imwrite("%s_%03d_%03d.png" % (video_file, i, lane_follower.curr_steering_angle), frame)
        # cv2.imwrite("%s_overlay_%03d.png" % (video_file, i), combo_image)
        # cv2.imwrite(f'./datas/{fname}_{i:0>3}_{lane_follower.curr_steering_angle}.jpg', frame)
        # cv2.imwrite(f'./datas/{fname}_{i:0>3}.png', combo_image)
        # # print(f'./datas/{video_file}_{i:0<3}_{lane_follower.curr_steering_angle}.jpg')
        # print(f'./datas/{video_file}_{i:0>3}.png')
        # video_overlay.write(combo_image)
        #
        cv2.imshow("Road with Lane line", combo_image)

        i += 1
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    # cap.release()
    # video_overlay.release()
    cv2.destroyAllWindows()


import os

if __name__ == '__main__':

    # Datos = 'p'
    # if not os.path.exists('datas'):
    #     print('Carpeta creada: ', 'datas')
    #     os.makedirs('datas')

    logging.basicConfig(level=logging.INFO)
    test_imgs()
    # test_photo('/home/pi/DeepPiCar/driver/data/video/car_video_190427_110320_073.png')
    # test_photo(sys.argv[1])
    # test_video(sys.argv[1])


