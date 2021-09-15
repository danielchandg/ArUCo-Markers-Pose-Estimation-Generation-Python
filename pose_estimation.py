'''
Sample Usage:-
python pose_estimation.py --K_Matrix calibration_matrix.npy --D_Coeff distortion_coefficients.npy --type DICT_5X5_100
'''


import numpy as np
import cv2
import sys
from utils import ARUCO_DICT
import argparse
import time


def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):

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

            tvec = [a for a in tvec[0][0]] # Rounded to two decimals
            rvec = rvec[0][0]
            location = str(ids[i]) + ", ".join(["{:.2f}".format(a) for a in tvec])
            # rotation = ", ".join(["{:.2f}".format(a) for a in rvec[0][0]])

            ### PROTOTYPE TO PRINT LOCATION OF CAMERA WHEN DETECTS ARUCO MARKER IN MAZE ###
            absolute_location = ""
            if ids[i]==1:
                absolute_location = "(" + "{:.2f}".format(tvec[0]) + ", " + "{:.2f}".format(tvec[2]-0.12) + ")"
            elif ids[i]==2:
                absolute_location = "(" + "{:.2f}".format(tvec[2]-0.12) + ", " + "{:.2f}".format(tvec[0]) + ")"
            elif ids[i]==6:
                absolute_location = "(" + "{:.2f}".format(tvec[2]-0.12) + ", " + "{:.2f}".format(tvec[0]-0.232) + ")"
            elif ids[i]==9:
                absolute_location = "(" + "{:.2f}".format(tvec[0]+0.232) + ", " + "{:.2f}".format(tvec[2]-0.12) + ")"

            if not is_location_drawn:
                cv2.putText(frame, absolute_location, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                is_location_drawn = True


            cv2.putText(frame, location, (topLeft[0], topLeft[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # cv2.putText(frame, rotation, (bottomLeft[0], bottomLeft[1]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--K_Matrix", required=True, help="Path to calibration matrix (numpy file)")
    ap.add_argument("-d", "--D_Coeff", required=True, help="Path to distortion coefficients (numpy file)")
    ap.add_argument("-t", "--type", type=str, default="DICT_4X4_50", help="Type of ArUCo tag to detect")
    args = vars(ap.parse_args())

    
    if ARUCO_DICT.get(args["type"], None) is None:
        print(f"ArUCo tag type '{args['type']}' is not supported")
        sys.exit(0)

    aruco_dict_type = ARUCO_DICT[args["type"]]
    calibration_matrix_path = args["K_Matrix"]
    distortion_coefficients_path = args["D_Coeff"]
    
    k = np.load(calibration_matrix_path)
    d = np.load(distortion_coefficients_path)

    video = cv2.VideoCapture(2)
    time.sleep(2.0)

    while True:
        ret, frame = video.read()

        if not ret:
            break
        
        output = pose_estimation(frame, aruco_dict_type, k, d)

        cv2.imshow('Estimated Pose', output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()