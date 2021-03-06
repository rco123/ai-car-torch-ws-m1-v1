
import cv2
import numpy as np
import logging
import math
import datetime
import sys

_SHOW_IMAGE = False
def halt(time=0):
    cv2.waitKey(time)

class HandCodedLaneFollower(object):

    def __init__(self, car=None):
        logging.info('Creating a HandCodedLaneFollower...')
        self.curr_steering_angle = 0
        self.img_cnt = 0

    def follow_lane(self, frame):

        frame = cv2.resize(frame,dsize=(0,0),fx=0.5,fy=0.5, interpolation=cv2.INTER_LINEAR)

        self.detect_lane(frame)
        diff = self.curr_steering_angle

        height = frame.shape[0]
        width = frame.shape[1]

        #cv2.imwrite(f'./imgs/im_{self.img_cnt:03d}.jpg', frame)
        #self.img_cnt += 1
        # print(f'diff, h , w = {diff}, {height}, {width}')
        # cv2.line(frame, (int(width / 2), height), (int(width / 2 + diff), int(height / 2)), (255, 0, 0), 3)
        # show_image('line', frame, True)
        print(f'diff = {diff}')
        heading_point = int((width / 2) + diff)
        heading_img = display_heading_line(frame, heading_point, line_color=(0, 0, 255), line_width=5)
        show_image('heading', heading_img, True)

        return frame


    def detect_lane(self, frame):
        show_image('frame',frame, False)

        canny_edges = detect_canny_edges(frame)
        show_image('edges', canny_edges, False)

        height = frame.shape[0]
        width = frame.shape[1]

        polygon_1 = np.array([[
            (0, height * 1 / 2),
            (width, height * 1 / 2),
            (width, height),
            (0, height),
        ]], np.int32)

        cropped_edges = region_of_interest(canny_edges, polygon_1)
        show_image('edges cropped', cropped_edges, False)

        line_segments = detect_line_segments(cropped_edges)
        line_segment_image = display_lines(frame, line_segments)
        show_image("line segments", line_segment_image, False)

        lane_lines,line_cnt,left_line_is,right_line_is = average_slope_intercept(frame, line_segments)
        if line_cnt == 0 :
            print('nothing do')
            return

        lane_lines_image = display_lines(frame, lane_lines,line_width=3)
        show_image("ave line", lane_lines_image, False)

        heading_point = 0
        pp_dist = 300 # setting ???????????? ??????
        #pp_dist = 800 # setting ???????????? ??????
        pp_dist = 400 # setting ???????????? ??????

        if line_cnt == 2 :
            a = lane_lines[0][0][2]
            b = lane_lines[1][0][2]
            c = a + int((b - a) / 2)
            print(c)
            heading_point = c

        if line_cnt == 1 :
            if left_line_is is True :
                a = lane_lines[0][0][2]
                b = a + pp_dist
                c = a + int((b - a) / 2)
                print(c)
                heading_point = c

            if right_line_is is True:
                a = lane_lines[0][0][2]
                b = a - pp_dist
                c = a + int((b - a) / 2)
                print(c)
                heading_point = c

        diff = heading_point - (width / 2) + (-50)  # ????????? ??????
        diff = diff * 2
        self.curr_steering_angle = stabilize_steering_angle(self.curr_steering_angle, diff, max_angle_deviation=70)

        # heading_img = display_heading_line(frame, heading_point, line_color=(0, 0, 255), line_width=5)
        # show_image('heading',heading_img,True)

        return


############################
# Frame processing steps
############################

def display_heading_line(frame, heading_point, line_color=(0, 0, 255), line_width=5, ):
    heading_image = np.zeros_like(frame)
    height, width, _ = frame.shape

    x1 = int(width / 2)
    y1 = height
    x2 = heading_point
    y2 = int(height / 2)

    print(f'this value {x1},{y1},{x2},{y2}')

    cv2.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)
    heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)
    return heading_image


def detect_canny_edges(frame):

    frame = cv2.blur(frame, (5, 5))

    # filter for blue lane lines
    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # show_image("hsv", hsv)
    # lower_blue = np.array([0, 0, 0])
    # upper_blue = np.array([255, 255, 100])

    # mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # show_image("blue mask", mask, True)

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    show_image("blue mask", gray, False)
    rtn, mask = cv2.threshold(gray, 50, 255 , cv2.THRESH_BINARY)
    show_image("threshold", mask, False)

    # detect edges
    canny_edges = cv2.Canny(mask, 200, 400)

    return canny_edges


def detect_line_area(frame):
    # filter for blue lane lines

    frame = cv2.blur(frame, (5, 5))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    show_image("hsv", hsv, False)
    lower_blue = np.array([0, 0, 0])
    upper_blue = np.array([255, 255, 50])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    show_image("blue mask", mask, False)
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
    line_segments = cv2.HoughLinesP(cropped_edges, rho, angle, min_threshold, np.array([]), minLineLength=30,
                                    maxLineGap = 2)

    if line_segments is not None:
        for line_segment in line_segments:
            logging.debug('detected line_segment:')
            logging.debug("%s of length %s" % (line_segment, length_of_line_segment(line_segment[0])))
            # print('detected line_segment:')
            # print("%s of length %s" % (line_segment, length_of_line_segment(line_segment[0])))

    return line_segments

def average_slope_intercept(frame, line_segments):
    """
    This function combines line segments into one or two lane lines
    If all line slopes are < 0: then we only have detected left lane
    If all line slopes are > 0: then we only have detected right lane
    """
    line_cnt = 0
    left_line = False
    right_line = False
    lane_lines = []
    if line_segments is None:
        logging.info('No line_segment segments detected')
        return lane_lines, line_cnt, 0, 0

    height, width, _ = frame.shape
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
            if slope > -5 and slope < - 0.3 :
                left_fit.append((slope, intercept))
            elif slope < 5 and slope > 0.3 :
                right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0) #????????? ????????? ?????? ???????????? ?????????.
    if len(left_fit) > 0:
        lane_lines.append(make_points(frame, left_fit_average))
        line_cnt +=1
        left_line = True

    right_fit_average = np.average(right_fit, axis=0)
    if len(right_fit) > 0:
        lane_lines.append(make_points(frame, right_fit_average))
        line_cnt +=1
        right_line = True

    print('lane lines: %s' % lane_lines)  # [[[316, 720, 484, 432]], [[1009, 720, 718, 432]]]
    return lane_lines, line_cnt, left_line, right_line


def compute_steering_angle(frame, lane_lines):
    """ Find the steering angle based on lane line coordinate
        We assume that camera is calibrated to point to dead center
    """
    x_offset = 0;
    y_offset = 0
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

    #logging.debug('new steering angle: %s' % steering_angle)
    #print('new steering angle: %s' % steering_angle)
    return steering_angle



def stabilize_steering_angle(curr_steering_angle, new_steering_angle,max_angle_deviation = 10):
    """
    Using last steering angle to stabilize the steering angle
    This can be improved to use last N angles, etc
    if new angle is too different from current angle, only turn by max_angle_deviation degrees
    """

    angle_deviation = new_steering_angle - curr_steering_angle

    if abs(angle_deviation) > max_angle_deviation:
        stabilized_steering_angle = int(curr_steering_angle +
                                        max_angle_deviation *  ( angle_deviation  / abs(angle_deviation)))
        print(f'angl = {stabilized_steering_angle}')
    else:
        stabilized_steering_angle = new_steering_angle

    #print(f'new {new_steering_angle}, return: {stabilized_steering_angle}')
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


def length_of_line_segment(line):
    x1, y1, x2, y2 = line
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def show_image(title, img, show=False):
    if show:
        #img_resize = cv2.resize(frame,dsize=(0,0),fx=0.5,fy=0.5, interpolation=cv2.INTER_LINEAR)
        cv2.imshow(title, img)


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
        frame = cv2.imread('./imgs/' + file)
        print(f'frame {i}, {file}, {frame.shape}')

        #frame = frame[:, 100:, :]
        show_image('org', frame)

        combo_image = lane_follower.follow_lane(frame)

        # cv2.imwrite("%s_%03d_%03d.png" % (video_file, i, lane_follower.curr_steering_angle), frame)
        # cv2.imwrite("%s_overlay_%03d.png" % (video_file, i), combo_image)
        # cv2.imwrite(f'./datas/{fname}_{i:0>3}_{lane_follower.curr_steering_angle}.jpg', frame)
        # cv2.imwrite(f'./datas/{fname}_{i:0>3}.png', combo_image)
        # # print(f'./datas/{video_file}_{i:0<3}_{lane_follower.curr_steering_angle}.jpg')
        # print(f'./datas/{video_file}_{i:0>3}.png')
        # video_overlay.write(combo_image)
        #cv2.imshow("Road with Lane line", combo_image)

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



