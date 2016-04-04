#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>

using namespace cv;
using namespace std;


class InputProcessing {

	CascadeClassifier faceCascade;
	CascadeClassifier eyeCascadeGlasses;
	CascadeClassifier eyeCascade;
	int inputType;
	VideoCapture cap;
	

public:
	const int static INPUT_TYPE_CAMERA_INPUT = 1;
	const int static INPUT_TYPE_VIDEO_INPUT = 2;
	const int static INPUT_TYPE_BIOID_DB = 3;
	const int static INPUT_TYPE_GI4E_DB = 4;

	const int static GROUND_TRUTH_LEFT_OUTER_CORNER = 0;
	const int static GROUND_TRUTH_LEFT_CENTER = 1;
	const int static GROUND_TRUTH_LEFT_INNER_CORNER = 2;
	const int static GROUND_TRUTH_RIGHT_INNER_CORNER = 3;
	const int static GROUND_TRUTH_RIGHT_CENTER = 4;
	const int static GROUND_TRUTH_RIGHT_OUTER_CORNER = 5;

	const string WINDOW_NAME = "debug window";

	vector<vector<Point2f>> labels; // ground truth for features in GI4E db

	Mat prevRedFrame;
	unsigned long frameCount = 0;
	vector<Point2i> corners;


	bool DEBUG_MODE;

	InputProcessing(int inputType, bool DEBUG_MODE = false);

	/* returns next frame of for input type selected in constructor */
	Mat getNextFrame(unsigned long frameNum);

	/* returns red channel of image, if input is rgb */
	Mat getRedChannelMatrix(const Mat frame);

	/* returns area of frame where the face is */
	Rect getFacePosition(const Mat frame);

	/* returns area of user's left eye (user's left) */
	Rect getLeftEyePosition(Mat frame, Rect facePosition);

	Rect getRightEyePosition(Mat frame, Rect facePosition);

	/* returns position of user's inner eye corner */
	Point2i getLeftEyeCorner(Mat gray, Rect leftEye, Mat drawFrame = Mat(0,0,CV_8U));

	Point2i getRightEyeCorner(Mat gray, Rect leftEye, Mat drawFrame = Mat(0, 0, CV_8U));


	/* returns ground thruth for GI4E db. flag variable specifies feature point: GROUND_TRUTH_(SIDE)_(COMPONENT) */
	Point getGroundTruth(int frameCount, int flag);

	/*
		Eye centre described by:
		F. Timm and E. Barth, “Accurate eye centre localisation by means of gradients.,” in VISAPP, pp. 125–130, 2011.
	*/
	Point timm2011accurate(Mat frame, Rect eye);

	Point timm2011accurateTest(Mat frame, Rect eye);


	/* tim2012accurate with gradient descend */
	Point2f getEyeCenter(Mat frame, Rect eye, unsigned int stepCount = 200, float gamma = .1);

	/* same, but with eye corners as ROI boundries */
	Point2f getEyeCenter(Mat frame, Point leftCorner, Point rightCorner);

	/* returns minimal intensity in 4-neighbourhood as pupil center, used as a starting point for getEyeCenter */
	Point getPupilPointFromIntensity(Mat ROI, float shrinkFactor);

	/*saves feature positions + gaze coordinates into file
	returns true if features were found
	*/
	bool saveFeatures(ofstream & file, int x, int y);

	void processTrainingFile(ofstream & file);

	/* if input is live video stream, specifie camera number */
	void setCamera(int deviceNum);

	/* if input is a video file, set file */
	void setVideo(string file);

private:
	/* Finds center of a large area for getEyeCenter() */
	Point getApproximateEyeCenter(Mat frame, Rect eye);




};
