#pragma once
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
const int INPUT_TYPE = InputProcessing::INPUT_TYPE_GI4E_DB;

void test() {

	unsigned long frameCount = 0; 
	namedWindow(windowName, 1);
	InputProcessing ip(INPUT_TYPE, DEBUG_MODE);
	Mat frame, gray, red;

	// stopwatch
	SYSTEMTIME time;
	WORD tic, toc,atic, atoc;
	ofstream csv;
	csv.open("..\\test_results\\gray_channel.csv");
	csv << "test #2" << ";\n";
	csv << "file;left outer error; left center error; left inner error; right inner error;right center error; right outer error;\n";

	while (true) {
		GetSystemTime(&time);
		atic = (time.wSecond * 1000) + time.wMilliseconds;
		frame = ip.getNextFrame(frameCount);
		unsigned int pnum = frameCount / 12 + 1;
		unsigned int fnum = frameCount % 12 + 1;

		csv << setw(3) << setfill('0') << pnum << "_"
			<< setw(2) << setfill('0') << fnum << ".png" << ";";

		if (frame.rows == 0) {
			csv << '\n';
			break;
			
		}

		red = ip.getRedChannelMatrix(frame);
		cvtColor(frame, gray, CV_BGR2GRAY);

		frameCount++;

		GetSystemTime(&time);
		tic = (time.wSecond * 1000) + time.wMilliseconds;

		Rect face = ip.getFacePosition(gray);

		if (face.width == 0) {
			if (waitKey(30) == 27)
				break;
			imshow(windowName, frame);
			csv << '\n';
			continue;
		}
		if (DEBUG_MODE) {
			rectangle(frame, face, Scalar(45, 200, 200, 100));
		}

		Rect leftEye = ip.getLeftEyePosition(gray, face);
		Rect rightEye = ip.getRightEyePosition(gray, face);

		GetSystemTime(&time);
		toc = (time.wSecond * 1000) + time.wMilliseconds;

		if (leftEye.width == 0 || rightEye.width == 0) {
			csv << '\n';
			continue;
		}


		if (DEBUG_MODE) {
			rectangle(frame, leftEye, Scalar(45, 45, 200, 100));
			rectangle(frame, rightEye, Scalar(200, 200, 45, 100));
		}

		Point2i leftEyeInnerCorner = ip.getLeftEyeCorner(gray, leftEye);
		Point2i rightEyeInnerCorner = ip.getRightEyeCorner(gray, rightEye);
		Point2i leftEyeOuterCorner = ip.getRightEyeCorner(gray, leftEye);
		Point2i rightEyeOuterCorner = ip.getLeftEyeCorner(gray, rightEye);
		if (leftEyeInnerCorner.x == -1 || rightEyeInnerCorner.x == -1) {
			csv << '\n';
			continue;
		}

		if (leftEyeOuterCorner.x == -1 || rightEyeOuterCorner.x == -1) {
			csv << '\n';
			continue;
		}
		circle(frame, leftEyeInnerCorner, 2, Scalar(10, 255, 255), -1, 8, 0);
		circle(frame, rightEyeInnerCorner, 2, Scalar(10, 255, 255), -1, 8, 0);
		circle(frame, leftEyeOuterCorner, 2, Scalar(10, 255, 255), -1, 8, 0);
		circle(frame, rightEyeOuterCorner, 2, Scalar(10, 255, 255), -1, 8, 0);


		
		Point leftCenter = ip.getEyeCenter(red, leftEye);
		

		Point rightCenter = ip.getEyeCenter(red, rightEye);

		circle(frame, leftCenter, 2, Scalar(255, 10, 255), -1, 8, 0);
		circle(frame, rightCenter, 2, Scalar(255, 10, 255, 20), -1, 8, 0);




		circle(frame, leftCenter, 2, Scalar(20, 210, 21), -1, 8, 0);
		circle(frame, rightCenter, 2, Scalar(20, 210, 21), -1, 8, 0);

		Point groundTruth = ip.getGroundTruth(frameCount, ip.GROUND_TRUTH_LEFT_CENTER);

		//console interface
		if (fnum == 1)
			std::printf("\n %d/103 ", pnum);
		std::printf(".");

		//fill errors in csv
		double err = 0.0;
		err = norm(leftEyeOuterCorner - ip.getGroundTruth(frameCount - 1, ip.GROUND_TRUTH_LEFT_OUTER_CORNER));
		csv << err << ';';
		err = norm(leftCenter - ip.getGroundTruth(frameCount - 1, ip.GROUND_TRUTH_LEFT_CENTER));
		csv << err << ';';
		err = norm(leftEyeInnerCorner - ip.getGroundTruth(frameCount - 1, ip.GROUND_TRUTH_LEFT_INNER_CORNER));
		csv << err << ';';
		err = norm(rightEyeInnerCorner - ip.getGroundTruth(frameCount - 1, ip.GROUND_TRUTH_RIGHT_INNER_CORNER));
		csv << err << ';';
		err = norm(rightCenter - ip.getGroundTruth(frameCount - 1, ip.GROUND_TRUTH_RIGHT_CENTER));
		csv << err << ';';
		err = norm(rightEyeOuterCorner - ip.getGroundTruth(frameCount - 1, ip.GROUND_TRUTH_RIGHT_OUTER_CORNER));
		csv << err << ';';


		imshow(windowName, frame);
		if (waitKey(1) == 27)
			break;
		csv << "\n";

		// the camera will be deinitialized automatically in VideoCapture destructor
		GetSystemTime(&time);
		atoc = (time.wSecond * 1000) + time.wMilliseconds;
		WORD a = atoc - atic;
		WORD b = toc - tic;

		}
		//close csv file
		csv.close();
}

