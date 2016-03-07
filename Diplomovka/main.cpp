#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <windows.h> //for GetTickCount 
#include <iostream>
#include <stdio.h>

#include "inputprocessing.h"

using namespace cv;
using namespace std;

/** Global variables */

const String windowName = "Feature detection demo (Press Esc to terminate)";
const bool DEBUG_MODE = true;
const int INPUT_TYPE = InputProcessing::INPUT_TYPE_GI4E_DB;


/** Function variables */
Rect getLargestRect(vector<Rect> v);


int main(int, char) {
	int liceCount = 0;
	int lceCount = 0;
	float errThresh = 3;
	int successCount = 0;
	int cornerNotInROI = 0;

	unsigned long frameCount = 16;//12; 
	namedWindow(windowName, 1);
	InputProcessing ip(INPUT_TYPE, DEBUG_MODE);
	Mat frame, gray, red;

	// stopwatch
	SYSTEMTIME time;
	WORD tic, toc;
	
	while (true) {
		frame = ip.getNextFrame(frameCount);
		if (frame.rows == 0) {
			break;
		}
		
		red = ip.getRedChannelMatrix(frame);
		cvtColor(frame, gray, CV_BGR2GRAY);
		
		frameCount++;	

		Rect face = ip.getFacePosition(gray);
				
		if (face.width == 0) {
			if (waitKey(30) == 27)
				break;
			imshow(windowName, frame);
			continue;
		}
		if (DEBUG_MODE) {
			rectangle(frame, face, Scalar(45, 200, 200, 100));
		}
	
		Rect leftEye = ip.getLeftEyePosition(gray, face);
		Rect rightEye = ip.getRightEyePosition(gray, face);
		if (leftEye.width == 0 || rightEye.width == 0) {
			continue;
		}
		
	
		if (DEBUG_MODE) {
			rectangle(frame, leftEye, Scalar(45, 45, 200, 100));
			rectangle(frame, rightEye, Scalar(200, 200, 45, 100));
		}

		Point2i leftEyeCorner = ip.getLeftEyeCorner(red, leftEye, frame);
		Point2i rightEyeCorner = ip.getRightEyeCorner(red, rightEye, frame);
		if (leftEyeCorner.x == -1 || rightEyeCorner.x == -1)
			continue;

		circle(frame, leftEyeCorner, 2, Scalar(10, 255, 255), -1, 8, 0);
		circle(frame, rightEyeCorner, 2, Scalar(10, 255, 255), -1, 8, 0);
	
		
		GetSystemTime(&time);
		tic = (time.wSecond * 1000) + time.wMilliseconds;
		Point lcenter = ip.timm2011accurate(red, leftEye);
		GetSystemTime(&time);
		toc = (time.wSecond * 1000) + time.wMilliseconds;

		int timm2011TimeMilis = toc - tic;

		Point rcenter = ip.timm2011accurate(red, rightEye);

		circle(frame, lcenter, 2, Scalar(255, 10, 255), -1, 8, 0);
		circle(frame, rcenter, 2, Scalar(255, 10, 255), -1, 8, 0);

		GetSystemTime(&time);
		tic = (time.wSecond * 1000) + time.wMilliseconds;
		Point lcenter2 = ip.getEyeCenter(red, leftEye);
		GetSystemTime(&time);
		toc = (time.wSecond * 1000) + time.wMilliseconds;

		int ourTimeMilis = toc - tic;


		Point rcenter2 = ip.getEyeCenter(red, rightEye);

		circle(frame, lcenter2, 2, Scalar(20, 210, 21), -1, 8, 0);
		circle(frame, rcenter2, 2, Scalar(20, 210, 21), -1, 8, 0);
	
		Point groundTruth = ip.getGroundTruth(frameCount, ip.GROUND_TRUTH_LEFT_CENTER);
		double timm2011Dist = norm(Mat(groundTruth), Mat(lcenter));
		double ourDist = norm(Mat(groundTruth), Mat(lcenter2));

		printf("timm: %dms %fpx \t our: %dms %fpx \n", timm2011TimeMilis, timm2011Dist, ourTimeMilis, ourDist);

		//left eye corner error
		float lICError = norm(leftEyeCorner - ip.getGroundTruth(frameCount - 1, ip.GROUND_TRUTH_LEFT_INNER_CORNER));
		float lCError = norm(lcenter - ip.getGroundTruth(frameCount - 1, ip.GROUND_TRUTH_LEFT_CENTER));

		if (lICError > errThresh) {
			liceCount++;
			if (!leftEye.contains(ip.getGroundTruth(frameCount - 1, ip.GROUND_TRUTH_LEFT_INNER_CORNER))) {
				cornerNotInROI++;
			}
		}
		if (lCError > errThresh) {
			lceCount++;
		}
		successCount++;
		imshow(windowName, frame);
		if (waitKey(1) == 27)
			break;
	}
	// the camera will be deinitialized automatically in VideoCapture destructor
	printf("Succeses: %d\tErrors: LEC: %d\tLEIC %d\t CornerNotInRoi %d \n", successCount, lceCount, liceCount, cornerNotInROI);
	
	return 0;
}

