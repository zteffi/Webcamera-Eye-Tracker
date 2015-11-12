

#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>

#include "inputprocessing.h"

using namespace cv;
using namespace std;


InputProcessing::InputProcessing(): 
	faceCascade(CascadeClassifier()),
	leftEyeCascade(CascadeClassifier()),
	rightEyeCascade(CascadeClassifier())

{
	if (!faceCascade.load("haarcascade_frontalface_alt.xml")) {
		printf("--(!)File not found faceCascade\n"); exit(-11);
	}
	if (!leftEyeCascade.load("haarcascade_lefteye_2splits.xml")) {
		printf("--(!)File not found leftEyeCascade\n"); exit(-12);
	}
	if (!rightEyeCascade.load("haarcascade_righteye_2splits.xml")) {
		printf("--(!)File not found rightEyeCascade\n"); exit(-13);
	}

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


Rect InputProcessing::getFacePosition(const Mat frame) {
	// detect face
	vector<Rect> faces;
	equalizeHist(frame, frame);
	faceCascade.detectMultiScale(frame, faces, 1.1, 3, CV_HAAR_FIND_BIGGEST_OBJECT,
		Size(frame.rows*.2, frame.cols * .5));
	if (faces.size() < 1) {
		return Rect(0, 0, 0, 0);
	}
	return  faces[0];
}

Rect InputProcessing::getLeftEyePosition(Mat & frame, Rect & headPosition) {
return Rect(0, 0, 1, 1);
}
Rect InputProcessing::getRightEyePosition(Mat & frame, Rect & headPosition) {
return Rect(0, 0, 1, 1);
}

