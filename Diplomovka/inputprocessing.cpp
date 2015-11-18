

#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>

#include "inputprocessing.h"

using namespace cv;
using namespace std;






InputProcessing::InputProcessing(int inputType) :
	faceCascade(CascadeClassifier()),
	inputType(inputType)
	//leftEyeCascade(CascadeClassifier()),
	//rightEyeCascade(CascadeClassifier())

{
	printf("loading \n");
	if (!faceCascade.load("haarcascade_frontalface_alt.xml")) {
		printf("--(!)File not found faceCascade\n"); exit(-11);
	}
	if (!leftEyeCascade.load("haarcascade_lefteye_2splits.xml")) {
		printf("--(!)File not found leftEyeCascade\n"); exit(-12);
	}
	if (!rightEyeCascade.load("haarcascade_righteye_2splits.xml")) {
		printf("--(!)File not found rightEyeCascade\n"); exit(-13);
	}
	if (inputType < 1 || inputType > 3) {
		printf("--(!)Input type %i not specified\n", inputType); exit(-15);
	} 
	else if (inputType == INPUT_TYPE_CAMERA_INPUT) {
		// set default camera
		cap = VideoCapture(0);
		if (!cap.isOpened()) {
			printf("--(!)Camera 0 not available\n");
		}
	}

}

Mat InputProcessing::getNextFrame(long frameNum) {
	Mat frame;
	switch (inputType) {
	case INPUT_TYPE_VIDEO_INPUT:
		if (cap.get(CV_CAP_PROP_FRAME_COUNT) <= frameNum) {
			break;
		}
	case INPUT_TYPE_CAMERA_INPUT:
		cap >> frame; //get next frame from camera/video
		break;
	case INPUT_TYPE_BIOID_DB:
		// when we used all photos in db, return empty matrix
		if (waitKey(700) == 27) { exit(0); }
		if (frameNum < 340) {
			frame = imread("../BioID-FaceDatabase-V1.2/BioID_" + to_string(frameNum + 1181) + ".pgm");
		}

		break;
	default:
		printf("--(!)Input type %i not specified\n", inputType); exit(-15);
	} 
	return frame;
}



Mat InputProcessing::getSingleChannelMatrix(const Mat  frame) {
	if (frame.channels() == 1) {
		return frame;
	}
	else if (frame.channels() == 3) {
		Mat channel[3];
		split(frame, channel);
		/* if input is in rgb, we use red channel, since it maximizes contrast
		*  between iris and sclera, regardless of person's eye color.
		*/
		return channel[2];
	}
	printf("--(!)Unsupported number of channels : " + frame.channels());
	exit(101);
}


Rect InputProcessing::getFacePosition(const Mat  frame) {
	vector<Rect> faces(1);
	int slidingWindowSize = min(frame.rows,frame.cols)/4;
	faceCascade.detectMultiScale(frame, faces, 1.2, 3, CV_HAAR_FIND_BIGGEST_OBJECT, Size(slidingWindowSize, slidingWindowSize));

	if (faces.size() >= 1) {
		return  faces[0];
	}
	return Rect(0, 0, 0, 0);
}

Rect InputProcessing::getRightEyePosition(Mat frame, Rect facePosition) {
	vector<Rect> eyes(1);
	Rect rightEyeRect = Rect(facePosition.x + facePosition.width/8,
		facePosition.y + facePosition.height/4, 
		3*facePosition.width/8,
		facePosition.height/4);
	return rightEyeRect;
}

Rect InputProcessing::getLeftEyePosition(Mat frame, Rect facePosition) {
	vector<Rect> eyes(1);
	Rect leftEyeRect = Rect(facePosition.x + facePosition.width / 2,
		facePosition.y + 5 *facePosition.height /16,
		2 * facePosition.width / 8,
		facePosition.height / 8);
	return leftEyeRect;
}


/* if input is live video stream, specifie camera number */
void InputProcessing::setCamera(int deviceNum) {

}

/* if input is a video file, set file */
void InputProcessing::setVideo(string file) {

}


