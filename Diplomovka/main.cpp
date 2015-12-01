#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/features2d.hpp>
#include "inputprocessing.h"

#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

/** Global variables */

String windowName = "Demo";
bool DEBUG_MODE = true;
double alpha;
const int alpha_slider_max = 100;
int alpha_slider = 0;


/** Function variables */
Rect getLargestRect(vector<Rect> v);


void onTrackbar(int, void*) {
	alpha = 10*(double)alpha_slider / (double)alpha_slider_max;
}

int main(int, char) {
	long frameCount = 34; //skip glasses guy
	namedWindow(windowName, 1);
	InputProcessing ip(InputProcessing::INPUT_TYPE_CAMERA_INPUT);
	Mat frame, gray, red;
	
	while (true) {
		frame = ip.getNextFrame(frameCount);
		red = ip.getSingleChannelMatrix(frame);
		cvtColor(frame, gray, CV_BGR2GRAY);
		frameCount++;
		if (frame.rows == 0) {
			break;
		}
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
		leftEye.width/= 2;
		Rect rightEye = ip.getRightEyePosition(gray, face);
		rightEye.width /= 2;
		rightEye.x += rightEye.width;

		if (DEBUG_MODE) {
			rectangle(frame, leftEye, Scalar(45, 45, 200, 100));
			rectangle(frame, rightEye, Scalar(200, 200, 45, 100));
		}

		Mat im = Mat(red, leftEye);

		vector<Point2f> features;
	//	GaussianBlur(im, im, Size(3, 3), 0, 0);
		goodFeaturesToTrack(im, features, 8, .35, leftEye.height/16);
		
	
		if (DEBUG_MODE) {
			for (int i = 0; i < features.size(); i++)
			{
				 circle(frame, features[i] + Point2f(leftEye.x, leftEye.y), 1, Scalar(10,255,10), -1, 8, 0);
			}
		}
		int maxVal = INT_MIN, maxIndex = -1, i = 0;
		Point2f s(leftEye.width / 2, leftEye.height / 2);
		
		for (Point2f p : features) {
			int val = red.at<uchar>(p) - gray.at<uchar>(p);
			double dist = norm(Mat(p), Mat(s))/ norm(Mat(s));
			printf("%i - %f\n", val, dist);
			val -= alpha * dist;
			
			if (val > maxVal) {
				maxVal = val;
				maxIndex = i;
			}
			i++;
		}
		if (DEBUG_MODE && maxIndex >= 0) {
				circle(frame, features[maxIndex] + Point2f(leftEye.x, leftEye.y), 3, Scalar(10, 10, 255), -1, 8, 0);
		}

	/*
		/** Find eyes
		// shrink Regions of interest (ROI), in which eyes are detected
		Rect leftEye, rightEye; // left and right are from user's perspective
		Rect rEyeROI = Rect(faceROI.x + .2 * faceROI.width, faceROI.y + .3 * faceROI.height,
			.4 * faceROI.width, .3 *faceROI.height);
		Rect lEyeROI = Rect(faceROI.x + .5 * faceROI.width, faceROI.y + .3 * faceROI.height,
			.4 * faceROI.width, .3 *faceROI.height);
		if (DEBUG_MODE) {
			rectangle(frame, rEyeROI, Scalar(200, 20, 200));
			rectangle(frame, lEyeROI, Scalar(200, 200, 20));
		}
		
		// detect eyes using cascades
		vector<Rect> eyesUnfiltered;
		Mat rMat = Mat(equalizedGray, rEyeROI);
		// equalizeHist(rMat, rMat);
		eyesCascade.detectMultiScale(rMat, eyesUnfiltered, 1.05, 2, CV_HAAR_SCALE_IMAGE, Size(30, 30));
		if (eyesUnfiltered.size() > 0) {
			Rect rEye = getLargestRect(eyesUnfiltered);
			rEye.x += rEyeROI.x;
			rEye.y += rEyeROI.y;
			rectangle(frame, rEye, Scalar(255, 0, 0), 2);
			Mat im = Mat(gray, rEye);
			//imwrite("../right_eye.png", im);
			
			
		}
		Mat lMat = Mat(equalizedGray, lEyeROI);
		equalizeHist(lMat, lMat);
		eyesCascade.detectMultiScale(lMat, eyesUnfiltered, 1.05, 2, CV_HAAR_SCALE_IMAGE, Size(30, 30));
		if (eyesUnfiltered.size() > 0) {
			Rect lEye = getLargestRect(eyesUnfiltered);
			lEye.x += lEyeROI.x - 0.05 * lEyeROI.width;
			lEye.y += lEyeROI.y + 0.125 * lEyeROI.width;
			lEye.width /= 2.5;
			lEye.height /= 2;

			
			rectangle(frame, lEye, Scalar(255, 0, 0), 2);
			Mat im = Mat(gray, lEye);
			//imwrite("../left_eye.png", im);
			vector<Point2f> features;
			GaussianBlur(im, im, Size(3, 3), 1.5, 1.5);
			goodFeaturesToTrack(im, features, 1, .5, 1.1);
			RNG rng(12345);
			for (int i = 0; i < features.size(); i++)
			{
				circle(frame, features[i] + Point2f(lEye.x, lEye.y), 4, Scalar(rng.uniform(0, 255), rng.uniform(0, 255),
					rng.uniform(0, 255)), -1, 8, 0);
			}
			
		}

		*/


		//GaussianBlur(edges, edges, Size(7, 7), 1.5, 1.5);
		//Canny(edges, edges, 0, 30, 3);
		
		createTrackbar("LengthValue", windowName, &alpha_slider, alpha_slider_max, onTrackbar);

		imshow(windowName, frame);
		if (waitKey(7) == 27)
			break;
	}
	// the camera will be deinitialized automatically in VideoCapture destructor
	return 0;
}


Rect getLargestRect(vector<Rect> v) {
	Rect r = v[0];
	for (int i = 1; i < v.size(); i++) {
		if (v[i].area() > r.area())
			r = v[i];
	}
	return r;

}
