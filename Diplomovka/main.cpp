#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

/** Global variables */
String faceCascadeName = "haarcascade_frontalface_default.xml";
String eyesCascadeName = "haarcascade_eye.xml";
CascadeClassifier faceCascade;
CascadeClassifier eyesCascade;
String windowName = "Demo";
bool DEBUG_MODE = true;

double dWidth = 10;
double dHeight = 10;

/** Function variables */
Rect getLargestRect(vector<Rect> v);

int main(int, char) {
	VideoCapture cap(0);

	// Load cascades
	if (!faceCascade.load(faceCascadeName) || !eyesCascade.load(eyesCascadeName)) {
		printf("--(!)Error loading files\n"); return -1;
	};

	// Open the default camera
	if (!cap.isOpened()) {
		printf("--(!)Camera 0 not available\n"); return -2;
	}
	

	Mat equalizedGray;
	namedWindow(windowName, 1);

	String path = "C:\\Users\\stefa_000\\SkyDrive\\Documents\\MatfyzMgr\\Diplomovka\\face db\\BioID-FaceDatabase-V1.2\\";
	

	for (int f = 1228; f < 1520; f++) {
		Mat frame,  gray;
		
		//cap >> frame; // get a new frame from camera
		String fn = path + "BioID_" + to_string( f) + ".pgm";
		printf("%s", fn);
 		gray = imread(fn, CV_LOAD_IMAGE_GRAYSCALE);
		frame = gray;
		/*
		//cvtColor(frame, gray, CV_BGR2GRAY); // convert to equalizedGrayscale
		vector<Mat> channels;
		split(frame, channels);
		gray = channels[2]; // extract red channel, since it leaves highest contrast between iris and skin

		*/
		// detect face
		vector<Rect> faces;
		cv::equalizeHist(gray, equalizedGray);
		faceCascade.detectMultiScale(equalizedGray, faces, 1.1, 3, CV_HAAR_SCALE_IMAGE,
			Size(dWidth*.2, dHeight * .5));
		if (faces.size() < 1) {
			if (waitKey(30) == 27)
				break;
			imshow(windowName, frame);
			continue;
		}
		Rect faceROI = faces[0];
		if (DEBUG_MODE) {
			rectangle(frame, faceROI, Scalar(45, 200, 200, 100));
		}
		/** Find eyes*/
		// shrink Regions of interest (ROI), in which eyes are detected
		Rect leftEye, rightEye; // left and right are from user's perspective
		Rect rEyeROI = Rect(faceROI.x + .2 * faceROI.width, faceROI.y + .3 * faceROI.height,
			.25 * faceROI.width, .3 *faceROI.height);
		Rect lEyeROI = Rect(faceROI.x + .65 * faceROI.width, faceROI.y + .3 * faceROI.height,
			.25 * faceROI.width, .3 *faceROI.height);
		if (DEBUG_MODE) {
			rectangle(frame, rEyeROI, Scalar(200, 20, 200));
			rectangle(frame, lEyeROI, Scalar(200, 200, 20));
		}
		
		// detect eyes using cascades
		vector<Rect> eyesUnfiltered;
		Mat rMat = Mat(equalizedGray, rEyeROI);
		 equalizeHist(rMat, rMat);
		eyesCascade.detectMultiScale(rMat, eyesUnfiltered, 1.1, 3, CV_HAAR_SCALE_IMAGE, Size(5, 5));
		if (eyesUnfiltered.size() > 0) {
			Rect rEye = getLargestRect(eyesUnfiltered);
			rEye.x += rEyeROI.x;
			rEye.y += rEyeROI.y;
			rectangle(frame, rEye, Scalar(255, 0, 0), 2);
			Mat im = Mat(gray, rEye);
			//imwrite("../right_eye.png", im);
			
			
		}
		Mat lMat = Mat(equalizedGray, lEyeROI);
		cv::equalizeHist(lMat, lMat);
		eyesCascade.detectMultiScale(lMat, eyesUnfiltered, 1.05, 2, CV_HAAR_SCALE_IMAGE, Size(5, 5));
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

		

		//GaussianBlur(edges, edges, Size(7, 7), 1.5, 1.5);
		//Canny(edges, edges, 0, 30, 3);
		imshow(windowName, frame);
		if (waitKey(2000) == 32)
			continue;
		if (waitKey(30) == 27)
			break;
	}
	if (waitKey(10000) == 27)
		return 0;
	// the camera will be deinitialized automatically in VideoCapture destructor
	return 0;
}

Rect getLargestRect(vector<Rect> v) {
	if (v.size() == 0) {
		return Rect(0,0,0,0);
	}
	Rect r = v[0];
	for (int i = 1; i < v.size(); i++) {
		if (v[i].area() > r.area())
			r = v[i];
	}
	return r;
}
