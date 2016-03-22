#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <windows.h> //for GetTickCount 
#include <iostream>
#include <stdio.h>
#include <string>

#include "inputprocessing.h"

using namespace cv;
using namespace std;

/** Global variables */

const String windowName = "Feature detection demo (Press Esc to terminate)";
const bool DEBUG_MODE = true;
const int INPUT_TYPE = InputProcessing::INPUT_TYPE_CAMERA_INPUT;

// sorted from min x to max x coord:
// (rightEyeOuterCorner, rightEyeInnerCorner, leftEyeInnerCorner, leftEyeOuterCorner)
vector<Point2i> corners;


int main__(int, char) {

	unsigned long frameCount = 0;
	namedWindow(windowName, 1);
	InputProcessing ip(INPUT_TYPE, DEBUG_MODE);
	Mat frame, gray, prevRed, red;

	// user terminates application with esc key
	while (waitKey(1) != 27) { 
	
		//get canera frame
		frame = ip.getNextFrame(frameCount);
		

		red = ip.getRedChannelMatrix(frame);
		

		frameCount++;

		//if we don't have eye features to track, find them 
		if (corners.size() < 4) {

			cvtColor(frame, gray, CV_BGR2GRAY);

			//locate face
			Rect face = ip.getFacePosition(gray);

			if (face.width == 0) {
				if (waitKey(30) == 27)
					break;
				imshow(windowName, frame);
				continue;
			}

			Rect leftEye = ip.getLeftEyePosition(gray, face);
			Rect rightEye = ip.getRightEyePosition(gray, face);
			if (leftEye.width == 0 || rightEye.width == 0) {
				continue;
			}
			
			corners.push_back(ip.getLeftEyeCorner(red, rightEye, frame));
			corners.push_back(ip.getRightEyeCorner(red, rightEye, frame));
			corners.push_back(ip.getLeftEyeCorner(red, leftEye, frame));
			corners.push_back(ip.getRightEyeCorner(red, leftEye, frame));
			

			if (corners[0].x == -1 || corners[1].x == -1
				|| corners[2].x == -1 || corners[3].x == -1) {
				continue;
			}

		}
		else {
			vector<uchar> status(4);
			vector<float> errors(4);
			vector<Point> nextCorners(4);
			try {
				calcOpticalFlowPyrLK(prevRed, red, corners, nextCorners, status, errors);
				
			}
			catch (cv::Exception e) {
				cout << e.err << "\n";
				corners.clear();
				continue;
			}
			
			//if not all 4 corners were found
			if ((status[0] && status[1] && status[2] && status[3]) != 1) {
				corners.clear();
				continue;
			}
			corners = nextCorners;
		}


		circle(frame, corners[0], 2, Scalar(10, 255, 255), -1, 8, 0);
		circle(frame, corners[1], 2, Scalar(10, 255, 255), -1, 8, 0);
		circle(frame, corners[2], 2, Scalar(10, 255, 255), -1, 8, 0);
		circle(frame, corners[3], 2, Scalar(10, 255, 255), -1, 8, 0);

		Point rightCenter = ip.getEyeCenter(red, corners[0], corners[1]);
		Point leftCenter = ip.getEyeCenter(red, corners[2], corners[3]);

		circle(frame, leftCenter, 2, Scalar(20, 210, 21), -1, 8, 0);
		circle(frame, rightCenter, 2, Scalar(20, 210, 21), -1, 8, 0);

		prevRed = red;
		imshow(windowName, frame);
		
	}
	// the camera will be deinitialized automatically in VideoCapture destructor

	return 0;
}

