
#pragma once


#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;


class InputProcessing {

	CascadeClassifier & faceCascade;
	CascadeClassifier & leftEyeCascade;
	CascadeClassifier & rightEyeCascade;

public:
	InputProcessing();
/* returns single channel image, if input is rgb */
Mat getSingleChannelMatrix(const Mat frame);

/* returns area of frame where the face is */
Rect getFacePosition(const Mat frame);


/* returns area of user's left eye (user's left) */
Rect getLeftEyePosition(Mat & frame, Rect & headPosition);

/* returns area of user's right eye */
Rect getRightEyePosition(Mat & frame, Rect & headPosition);

};
