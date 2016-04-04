#pragma once
#include <windows.h> //for GetTickCount 

#include "training.h"
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


int main(int, char) {

	unsigned long frameCount = 0;
	namedWindow(windowName, 1);
	InputProcessing ip(INPUT_TYPE, DEBUG_MODE);
	Mat frame, gray, prevRed, red;

	trainingPhase(ip, Size(1920, 1080));
	
	// the camera will be deinitialized automatically in VideoCapture destructor

	return 0;
}

