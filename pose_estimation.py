'''
Sample Usage:-
python pose_estimation.py --K_Matrix calibration_matrix.npy --D_Coeff distortion_coefficients.npy --type DICT_5X5_100 --Aruco maze.txt
'''


import numpy as np
import cv2
import sys
from utils import ARUCO_DICT
import argparse
import time

# Sample Maze Layout:
#    1 2
#  8     3
#  7     4
#    6 5
# maze_ids = [[1, 0, 1], [2, 0, 2], [3, 1, 3], [4, 2, 3], [5, 3, 2], [6, 3, 1], [7, 2, 0], [8, 1, 0]]


def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients, maze_ids):

    '''
    frame - Frame from the video stream
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients associated with your camera

    return:-
    frame - The frame with the axis drawn on it
    '''

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters_create()

    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, cv2.aruco_dict,parameters=parameters,
        cameraMatrix=matrix_coefficients,
        distCoeff=distortion_coefficients)

    is_location_drawn = False
    absolute_location = ""

    # If markers are detected

    if len(corners) > 0:
        for i in range(0, len(ids)):
            # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.074, matrix_coefficients,
                                                                       distortion_coefficients)
            (rvec - tvec).any()
            # Draw a square around the markers
            cv2.aruco.drawDetectedMarkers(frame, corners) 

            # Draw Axis
            cv2.aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)

            # Draw location
            topLeft, topRight, bottomRight, bottomLeft = corners[i].astype(int).reshape((4, 2))

            tvec = tvec[0][0]
            rvec = rvec[0][0]
            location = str(ids[i]) + ", ".join(["{:.2f}".format(a) for a in tvec])
            # rotation = ", ".join(["{:.2f}".format(a) for a in rvec[0][0]])

            ### PROTOTYPE TO PRINT LOCATION OF CAMERA WHEN DETECTS ARUCO MARKER IN MAZE ###



            x = 1.1168 + 0.6126*tvec[0] + 1.423*tvec[2] - 0.4916*rvec[0] - 0.2856*rvec[1] + 0.4777*rvec[2]
            y = 9.1582 + 10.7215*tvec[0] + 4.239*tvec[2] - 3.662*rvec[0] - 2.61*rvec[1] - 1.8533*rvec[2]

            if len(absolute_location) == 0:
                for j in range(len(maze_ids)):
                    if ids[i] == maze_ids[j][0]:
                        x = 1.883 + 1.209*tvec[2] - 0.716*rvec[0] + 0.763*rvec[2]
                        y = -1.128 + 6.423*tvec[0] + 2.727*tvec[2] - 1.849*rvec[2]
                        if maze_ids[j][1]==0: # Facing North
                            absolute_location = "(" + "{:.2f}".format(x) + ", " + "{:.2f}".format(y + maze_ids[j][2] - 1) + ")"
                        elif maze_ids[j][2]==len(maze)-1: # Facing East
                            absolute_location = "(" + "{:.2f}".format(maze_rows - y - maze_ids[j][2] + 1) + ", " + "{:.2f}".format(maze_columns - x) + ")"
                        elif maze_ids[j][1]==len(maze)-1: # Facing South
                            absolute_location = "(" + "{:.2f}".format(maze_rows - x) + ", " + "{:.2f}".format(maze_columns - y - maze_ids[j][2] + 1) + ")"
                        else: # Facing West
                            absolute_location = "(" + "{:.2f}".format(y + maze_ids[j][1] - 1) + ", " + "{:.2f}".format(x) + ")"
                        break

            # absolute_location = "tvec: (" + "{:.2f}".format(tvec[0]) + ", " + "{:.2f}".format(tvec[1]) + ", " + "{:.2f}".format(tvec[2]) + ")\nrvec: (" + "{:.2f}".format(rvec[0]) + ", " + "{:.2f}".format(rvec[1]) + ", " + "{:.2f}".format(rvec[2]) + ")"

            if not is_location_drawn:
                cv2.putText(frame, absolute_location, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)


            # cv2.putText(frame, location, (topLeft[0], topLeft[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # cv2.putText(frame, rotation, (bottomLeft[0], bottomLeft[1]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--K_Matrix", required=True, help="Path to calibration matrix (numpy file)")
    ap.add_argument("-d", "--D_Coeff", required=True, help="Path to distortion coefficients (numpy file)")
    ap.add_argument("-a", "--Aruco", required=True, type=str, help=".txt file of arrangement of ArUco markers")
    ap.add_argument("-t", "--type", type=str, default="DICT_4X4_50", help="Type of ArUCo tag to detect")
    args = vars(ap.parse_args())

    
    if ARUCO_DICT.get(args["type"], None) is None:
        print("ArUCo tag type '{args['type']}' is not supported")
        sys.exit(0)

    aruco_dict_type = ARUCO_DICT[args["type"]]
    calibration_matrix_path = args["K_Matrix"]
    distortion_coefficients_path = args["D_Coeff"]
    
    k = np.load(calibration_matrix_path)
    d = np.load(distortion_coefficients_path)

    aruco_markers_file = open(args["Aruco"], 'r')
    maze = aruco_markers_file.readlines()
    maze_id_count = []
    aruco_id = ""
    x = 1
    for i in maze[0]:
        if '1' <= i <= '9':
            aruco_id += i
        elif len(aruco_id) > 0:
            maze_id_count.append([int(aruco_id), 0, x])
            x += 1
            # print("found aruco marker " + aruco_id + " in 1st row")
            aruco_id = ""

    if len(aruco_id) > 0:
        maze_id_count.append([int(aruco_id), 0, x])
        x += 1
        # print("found aruco marker " + aruco_id + " in 1st row")
        aruco_id = ""
    maze_columns = len(maze_id_count)-1
    maze_rows = len(maze)-3
    y = x
    x = 1

    for i in maze[len(maze)-1]:
        if '1' <= i <= '9':
            aruco_id += i
        elif len(aruco_id) > 0:
            maze_id_count.append([int(aruco_id), len(maze) - 1, x])
            x += 1
            # print("found aruco marker a" + aruco_id + " in last row")
            aruco_id = ""

    if len(aruco_id) > 0:
        maze_id_count.append([int(aruco_id), len(maze)-1, x])
        # print("found aruco marker " + aruco_id + " in last row")
        aruco_id = ""
        x += 1

    if x != y:
        raise ValueError('Number of aruco markers in 1st and last row are not equal')
    for i in range(1, len(maze)-1):
        x = 0
        cur_id_size = len(maze_id_count)
        for j in maze[i]:
            if '1' <= j <= '9':
                aruco_id += j
            elif len(aruco_id) > 0:
                maze_id_count.append([int(aruco_id), i, x])
                x = y
                aruco_id = ""

        if len(aruco_id) > 0:
            maze_id_count.append([int(aruco_id), i, x])
            aruco_id = ""

        if len(maze_id_count) - cur_id_size != 2:
            raise ValueError('Row ' + str(i) + ' does not have exactly 2 ids')

    # for x in maze_id_count:
    #     print(str(x[0]) + ", " + str(x[1]) + ", " + str(x[2]))

    video = cv2.VideoCapture(1)
    time.sleep(2.0)

    while True:
        ret, frame = video.read()

        if not ret:
            break
        
        output = pose_estimation(frame, aruco_dict_type, k, d, maze_id_count)

        cv2.imshow('Estimated Pose', output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()