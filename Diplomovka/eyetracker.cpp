#pragma once
#include <windows.h> //for GetTickCount 

#include "training_phase.h"
#include "tracking_phase.h"
#include "inputprocessing.h"

#include "testing.h"

using namespace cv;
using namespace std;

/** Global variables */

const String windowName = "Press \"Enter\" to start training phase ";
const bool DEBUG_MODE = false;
const int INPUT_TYPE = InputProcessing::INPUT_TYPE_GI4E_DB;

// sorted from min x to max x coord:
// (rightEyeOuterCorner, rightEyeInnerCorner, leftEyeInnerCorner, leftEyeOuterCorner)
vector<Point2i> corners;


int main(int, char) {
	unsigned long frameCount = 0;
	namedWindow(windowName, 1);
	InputProcessing ip(INPUT_TYPE, DEBUG_MODE);

	testFeatures(ip);

	printf("waiting for Enter...\n");

	Mat frame;
	

	//show user's face until enter is pressed
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
	
	// the camera will be deinitialized automatically in VideoCapture destructor
	//videos of screen and user face will be captured

	const char * trainingFile = "data/train.data";
	const char * trackFile = "data/features.data";
	const char * outputFile = "output/output.data";
	const char * trackingFolder = "output";
	const float scaleFactor = .5;
	
	bool completed = trainingPhase(ip, getScreenRes(), trainingFile);
	if (!completed)
		return 0;
		
	long captureCount = captureVids(ip, trackingFolder, scaleFactor);
	printf("processing webcam images (%d)...\n", captureCount);
	processTrackingData(ip, trackingFolder, captureCount, trackFile);
	printf("estimating gaze points...\n");
	processOutput(ip, getTrainingCount(), trainingFile, captureCount, trackFile, outputFile, trackingFolder, getScreenRes(scaleFactor));
	return 0;
}

