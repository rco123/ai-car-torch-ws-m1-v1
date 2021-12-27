
import cv2
import numpy as np
import logging
import math
import datetime
import sys
import os


def halt(time=0):
    cv2.waitKey(time)
def imshow(title, img, show=False):
    if show:
        img = cv2.resize(img,dsize=(0,0),fx=0.5,fy=0.5, interpolation=cv2.INTER_LINEAR)
        cv2.imshow(title, img)


class lane_det_m2(object):

    def __init__(self):
        logging.info('Creating a HandCodedLaneFollower...')
        self.curr_steering_angle = 0
        self.img_cnt = 0
        self.direction_point = 0


    # def follow_lane(self, img,threshold):
    #
    #     # frame = cv2.resize(frame,dsize=(0,0),fx=0.5,fy=0.5, interpolation=cv2.INTER_LINEAR)
    #
    #     self.img_to_angle(img,threshold)
    #     diff = self.curr_steering_angle
    #
    #     height = img.shape[0]
    #     width = img.shape[1]
    #
    #     #cv2.imwrite(f'./imgs/im_{self.img_cnt:03d}.jpg', frame)
    #     #self.img_cnt += 1
    #     # print(f'diff, h , w = {diff}, {height}, {width}')
    #     # cv2.line(frame, (int(width / 2), height), (int(width / 2 + diff), int(height / 2)), (255, 0, 0), 3)
    #     # show_image('line', frame, True)
    #
    #     print(f'diff = {diff}')
    #     heading_point = int((width / 2) + diff)
    #     heading_img = display_heading_line(frame, heading_point, line_color=(0, 0, 255), line_width=5)
    #     imshow('heading', heading_img, False)
    #
    #     display_heading_line_on_img(frame, heading_point, line_color=(0, 0, 255), line_width=5)
    #     imshow('heading', heading_img, False)
    #
    #     return frame


    def img_to_angle(self, img,threshold):

        canny_edges = detect_canny_edges(img, threshold)
        imshow('edges', canny_edges, False)

        height, width, dim = img.shape

        # polygon_1 = np.array([[
        #     (0, height * 1 / 2),
        #     (width, height * 1 / 2),
        #     (width, height),
        #     (0, height),
        # ]], np.int32)

        polygon_1 = np.array([[
            (int(width / 3), height * 1 / 2),
            ( int((width/ 3 ) *2) , height * 1 / 2),
            (width, height),
            (0, height),
        ]], np.int32)


        cropped_edges = region_of_interest(canny_edges, polygon_1)
        imshow('edges cropped', cropped_edges, True)

        line_segments = detect_line_segments(cropped_edges)
        line_segment_image = display_lines(img, line_segments)
        imshow("line segments", line_segment_image, True)

        lane_lines,line_cnt,left_line_is,right_line_is = average_slope_intercept(img, line_segments)
        print(f'lane_lines = {lane_lines}')

        heading_point = 0
        pp_dist = 400 # setting 사이간격 조정
        # pp_dist = 600 # setting 사이간격 조정
        # pp_dist = 800 # setting 사이간격 조정

        if line_cnt == 0 :
            print('nothing do')
            heading_point = self.direction_point

        if line_cnt == 2 :
            a = lane_lines[0][0][2]
            b = lane_lines[1][0][2]
            c = a + int((b - a) / 2)
            #print(c)
            heading_point = c

        if line_cnt == 1 :
            if left_line_is is True :
                a = lane_lines[0][0][2]
                b = a + pp_dist
                c = a + int((b - a) / 2)
                #print(c)
                heading_point = c

            if right_line_is is True:
                a = lane_lines[0][0][2]
                b = a - pp_dist
                c = a + int((b - a) / 2)
                # print(c)
                heading_point = c

        r_base = stabilize_steering_angle(self.direction_point, heading_point ,max_angle_deviation=20)
        self.direction_point = r_base

        diff = heading_point - (width / 2)
        print(f'heading point = {heading_point}, diff = {diff}')

        # lane_lines_image = display_lines(img, lane_lines,line_width=3)
        # imshow("ave line", lane_lines_image, True)
        display_lines_on_img(img, lane_lines,(0,255,255),line_width=3)
        # heading_img = display_heading_line(frame, heading_point, line_color=(0, 0, 255), line_width=5)
        # imshow('heading',heading_img,True)
        display_heading_line_on_img(img, heading_point, line_color=(0, 0, 255), line_width=5)

        return diff


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

def display_heading_line_on_img(frame, heading_point, line_color=(0, 0, 255), line_width=5, ):
    height, width, _ = frame.shape

    x1 = int(width / 2)
    y1 = height
    x2 = heading_point
    y2 = int(height / 2)

    # print(f'this value {x1},{y1},{x2},{y2}')
    cv2.line(frame, (x1, y1), (x2, y2), line_color, line_width)
   

def detect_canny_edges(frame,threshold):

    frame = cv2.blur(frame, (5, 5))
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    imshow("blue mask", gray, False)
    rtn, mask = cv2.threshold(gray, threshold, 255 , cv2.THRESH_BINARY)
    imshow("threshold", mask, True)
    # detect edges
    canny_edges = cv2.Canny(mask, 200, 400)
    return canny_edges

def region_of_interest(frame, polygen):
    height, width = frame.shape
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, polygen, 255)
    masked_image = cv2.bitwise_and(frame, mask)
    imshow("mask", masked_image,False)
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

    left_fit_average = np.average(left_fit, axis=0) #숫자중 차이가 많이 나는것은 버린다.
    if len(left_fit) > 0:
        lane_lines.append(make_points(frame, left_fit_average))
        line_cnt +=1
        left_line = True

    right_fit_average = np.average(right_fit, axis=0)
    if len(right_fit) > 0:
        lane_lines.append(make_points(frame, right_fit_average))
        line_cnt +=1
        right_line = True

    # print('lane lines: %s' % lane_lines)  # [[[316, 720, 484, 432]], [[1009, 720, 718, 432]]]
    return lane_lines, line_cnt, left_line, right_line


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
        # print(f'angl = {stabilized_steering_angle}')
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

def display_lines_on_img(frame, lines, line_color=(0, 255, 0), line_width=1):
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(frame, (x1, y1), (x2, y2), line_color, line_width)
    


def length_of_line_segment(line):
    x1, y1, x2, y2 = line
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def make_points(frame, line):
    height, width, _ = frame.shape
    slope, intercept = line
    y1 = height  # bottom of the frame
    y2 = int(y1 * 1 / 2)  # make points from middle of the frame down

    # bound the coordinates within the frame
    x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
    return [[x1, y1, x2, y2]]



if __name__ == '__main__':

    det = lane_det_m2(threshold=50)

    # cap = cv2.VideoCapture(video_file)
    # skip first second of video.
    # for i in range(3):
    #     _, frame = cap.read()
    # video_type = cv2.VideoWriter_fourcc(*'XVID')
    # fname = video_file.split('.')[0]
    # video_overlay = cv2.VideoWriter(f'{fname}_ovray.avi', video_type, 20.0, (320, 240))

    img_dir = './imgs/lane_m2/imgs/'
    files = os.listdir(img_dir)
    cnt =0
    for file in files:

        frame = cv2.imread(img_dir + file)
        print(f'frame {cnt}, {file}, {frame.shape}')

        # combo_image = det.follow_lane(frame,threshold=50)

        diff = det.img_to_angle(frame)
        if diff <= -800 :
            diff = -800
        elif diff >= 800 :
            diff = 800

        imshow('org', frame, True)

        cnt += 1
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    # cap.release()
    # video_overlay.release()
    cv2.destroyAllWindows()
