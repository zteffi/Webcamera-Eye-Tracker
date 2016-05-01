#pragma once
#pragma warning(disable:4996)
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <stdio.h>
#include <time.h>
#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/ml/ml.hpp"

//need old school libs for neural network due to bad documentation of openCV 3.0




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

		unsigned long frameCount = 0;


		bool DEBUG_MODE;

		InputProcessing(int inputType, bool DEBUG_MODE = false);

		/* returns next frame of for input type selected in constructor */
		Mat getNextFrame(unsigned long frameNum);

		/* returns red channel of image, if input is rgb */
		Mat getRedChannelMatrix(const Mat frame);

		/* returns area of frame where the face is */
		Rect getFacePosition(const Mat frame);

		/* returns area of user's left eye */
		Rect getLeftEyePosition(Mat frame, Rect facePosition);

		Rect getRightEyePosition(Mat frame, Rect facePosition);

		/* returns position of user's inner eye corner */
		Point2i getLeftEyeCorner(Mat gray, Rect leftEye);

		Point2i getRightEyeCorner(Mat gray, Rect leftEye);


		/* returns ground thruth for GI4E db. flag variable specifies feature point: GROUND_TRUTH_(SIDE)_(COMPONENT) */
		Point getGroundTruth(int frameCount, int flag);

	

		/*
		Eye centre described by:
		F. Timm and E. Barth, “Accurate eye centre localisation by means of gradients.,” in VISAPP, pp. 125–130, 2011.
		*/
		Point2d getEyeCenter(Mat frame, Rect eye);

		/* same, but with eye corners as ROI boundries */
		Point2d getEyeCenter(Mat frame, Point leftCorner, Point rightCorner);

		Point2d getEyeCenterTest(Mat frame, Rect eye);

		/* returns minimal intensity in 4-neighbourhood as pupil center, used as a starting point for getEyeCenter */
		Point getPupilPointFromIntensity(Mat ROI, double shrinkFactor);

		/*saves measure positions + gaze coordinates into file
		returns true if features were found
		*/
		bool saveMeasures(ofstream & file, int x, int y, Size screenSize);

		/*
			returns features as vector. If fails, returns empty vector
			Order:
			0:right outer,
			1:right center,
			2:right inner,
			3:left inner,
			4:left center,
			5:left outer,
		*/
		vector<Point> getFeatures(Mat gray);

		/*
			get measures feeded into NN which are
			0) ratio of x- dist between center and eye corners - right eye (user)
			1) left eye
			2) ratio of y- dist from inner corner to  eye center and x-dist between corners- right eye
			3) left eye
			4) ratio of eucl. distances between eye corners right and left
			5), 6) x, y coords of right inner corner
			7), 8) x, y coords of left inner corner

		*/
		vector<double> getDerivativeFeatures(vector<Point> features, Size frameSize);

		/*
		writes predictions into outputfile. If succes, returns 0, otherwise -1
		loadMatFromCSVFile(inputFile, Mat, 11, inputCout) returns 11 x inputCount matrix, last 2 cols are targets)
		trackfile 9x trackcount matrix
		outputfile 2x trackcount matrix
		*/
		int processTrainingFile(const char * inputFile, int inputCount, const char  * trackFile, int trackCount, const char  * outputFile);

		/*
		return 0 if csv loaded to mat, -1 otherwise
		*/
		int loadMatFromCSVFile(const char* filename, Mat & mat, int numAttr, int numLines);



	};


