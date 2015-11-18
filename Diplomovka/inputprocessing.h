
#pragma once


#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;


class InputProcessing {

	CascadeClassifier faceCascade;
	CascadeClassifier leftEyeCascade;
	CascadeClassifier rightEyeCascade;
	int inputType;
	VideoCapture cap;

public:
	const int static INPUT_TYPE_CAMERA_INPUT = 1;
	const int static INPUT_TYPE_VIDEO_INPUT = 2;
	const int static INPUT_TYPE_BIOID_DB = 3;

	InputProcessing(int inputType);

	/* returns next frame of for input type selected in constructor */
	Mat getNextFrame(long frameNum);

	/* returns single channel image, if input is rgb */
	Mat getSingleChannelMatrix(const Mat frame);

	/* returns area of frame where the face is */
	Rect getFacePosition(const Mat frame);

	/* returns area of user's right eye (user's right) */
	Rect getRightEyePosition(Mat frame, Rect facePosition);

	/* returns area of user's left eye */
	Rect getLeftEyePosition(Mat frame, Rect facePosition);

	/* if input is live video stream, specifie camera number */
	void setCamera(int deviceNum);

	/* if input is a video file, set file */
	void setVideo(string file);

};
