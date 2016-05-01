#pragma once
#include <windows.h> //for GetTickCount 

#include "training.h"
#include "tracking_phase.h"
#include "inputprocessing.h"


using namespace cv;
using namespace std;

/** Global variables */

const String windowName = "Press \"Enter\" to start training phase ";
const bool DEBUG_MODE = false;
const int INPUT_TYPE = InputProcessing::INPUT_TYPE_CAMERA_INPUT;

// sorted from min x to max x coord:
// (rightEyeOuterCorner, rightEyeInnerCorner, leftEyeInnerCorner, leftEyeOuterCorner)
vector<Point2i> corners;


int main(int, char) {
	
	unsigned long frameCount = 0;
	namedWindow(windowName, 1);
	InputProcessing ip(INPUT_TYPE, DEBUG_MODE);

	Mat m(48, 11, CV_32F);
	ip.loadMatFromCSVFile("data/meas-testing-martin.data", m, 11, 48);

	ip.processTrainingFile("data/meas-training-martin.data", 240, "data/meas-testing-martin.data",  48, "data/meas-output-martin-30-6.csv");
	
	Mat frame, gray, prevRed, red;
	

	//show user until enter is pressed
	while  (true) {
		frame = ip.getNextFrame(0);
		imshow(windowName, frame);
		char key = waitKey(10);
		//escape closes application
		if (key == 27)
			return 0;
		//enter closes window
		if (key == 32 || key == 13)
			break;
	}
	destroyWindow(windowName);
	trainingPhase(ip, getScreenRes());
	// the camera will be deinitialized automatically in VideoCapture destructor

	//videos of screen and user face will be captured
	//captureVids(ip);

	return 0;
}

