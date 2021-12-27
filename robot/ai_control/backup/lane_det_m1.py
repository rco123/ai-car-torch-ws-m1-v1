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

    def follow_lane(self, frame):
        # Main entry point of the lane follower
        show_image("orig", frame)
        l_point, r_point, cnt  = detect_lane(frame)
        final_frame = self.steer(frame, l_point, r_point, cnt)
        return frame

    def steer(self, frame, l_point, r_point, cnt):
        logging.debug('steering...')
        point_dist = 450
        if cnt == 0:
            logging.error('No lane lines detected, nothing to do.')
            return frame

        if cnt == 2:
             diff = ((l_point + r_point) / 2) -  frame.shape[1] / 2
             print('cnt2')
        if cnt == 1:
            if l_point > 0 :
                diff =  (l_point + (l_point + point_dist))/ 2  - frame.shape[1]/2
                print('cnt1-l')
            if r_point > 0:
                print('cnt1-r')
                diff = (r_point + (r_point - point_dist)) / 2 - frame.shape[1]/2


        self.curr_steering_angle = int(diff)
        print(f'current diff = {int(diff)}, {l_point}, {r_point}, {cnt}')

        # curr_heading_image = display_heading_line(frame, self.curr_steering_angle)
        # show_image("heading", curr_heading_image)

        return frame


############################
# Frame processing steps
############################

def detect_lane(frame):

    cnt = 0
    logging.debug('detecting lane lines...')

    area_img = detect_line_area(frame)
    show_image('area', area_img, True)

    cropped_area = region_of_interest(area_img)
    show_image('area cropped', cropped_area, True)

    half = int(cropped_area.shape[1] / 2)
    l_area = cropped_area[:,:half]
    print(f'l_area shape = {l_area.shape}')
    his_value = np.sum(l_area, axis=0)

    if np.sum(his_value) > 0 :
        # print(f'his_value = {his_value}')
        max_value = np.max(his_value)
        # print(f'max vlaue = {max_value}')
        index = np.where(his_value > int(max_value * 0.1))
        l_base_point = int(np.average(index))
        print(f'l base_point = {l_base_point}')
        cv2.circle(cropped_area, (l_base_point, cropped_area.shape[1]), 100, (255,255,255), 2 )
        show_image('area cropped', cropped_area, True)
        cnt += 1
    else:
        l_base_point = -1

    r_area = cropped_area[:,half:]
    his_value = np.sum(r_area, axis=0)
    if np.sum(his_value) > 0:

        # print(f'his_value = {his_value}')
        max_value = np.max(his_value)
        # print(f'max vlaue = {max_value}')
        index = np.where(his_value > int(max_value * 0.1))
        r_base_point = int(np.average(index))

        r_base_point = half + r_base_point
        cv2.circle(cropped_area, (r_base_point, cropped_area.shape[1]), 100, (255,255,255), 2 )
        show_image('area cropped', cropped_area, True)
        print(f'base_point = {r_base_point}')
        cnt += 1
    else:
        r_base_point = -1

    return l_base_point, r_base_point, cnt


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
    edges_img = cv2.Canny(mask, 200, 400)
    cv2.imshow('edged_img', edges_img)

    return edges_img


def region_of_interest(canny):
    height, width = canny.shape
    mask = np.zeros_like(canny)

    # only focus bottom half of the screen

    polygon = np.array([[
        (0, height * 1 / 2),
        (width, height * 1 / 2),
        (width, height * 2/3),
        (0, height * 2/ 3),
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    masked_image = cv2.bitwise_and(canny, mask)
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


def average_slope_intercept(frame, line_segments):
    """
    This function combines line segments into one or two lane lines
    If all line slopes are < 0: then we only have detected left lane
    If all line slopes are > 0: then we only have detected right lane
    """
    lane_lines = []
    if line_segments is None:
        logging.info('No line_segment segments detected')
        return lane_lines

    height, width, _ = frame.shape
    left_fit = []
    right_fit = []

    boundary = 1 / 4 #sgkim
    left_region_boundary = width * (1 - boundary)  # left lane line segment should be on left 2/3 of the screen
    right_region_boundary = width * boundary  # right lane line segment should be on left 2/3 of the screen


    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:
                logging.info('skipping vertical line segment (slope=inf): %s' % line_segment)
                continue
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))
            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    if len(left_fit) > 0:
        lane_lines.append(make_points(frame, left_fit_average))

    right_fit_average = np.average(right_fit, axis=0)
    if len(right_fit) > 0:
        lane_lines.append(make_points(frame, right_fit_average))

    logging.debug('lane lines: %s' % lane_lines)  # [[[316, 720, 484, 432]], [[1009, 720, 718, 432]]]
    print('lane lines: %s' % lane_lines)
    return lane_lines


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
        x_offset = x_offset * 1 # sgkim cali
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


def stabilize_steering_angle(curr_steering_angle, new_steering_angle, num_of_lane_lines,
                             max_angle_deviation_two_lines=5, max_angle_deviation_one_lane= 1):
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
    y2 = int(height * 2 / 3) #sgkim
    

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

