
#include "training.h"


void drawFancyTarget(Mat dst, Mat target, Point loc) {
	int i, j;
	uchar * p;
	uchar * q;
	for (i = 0; i < target.rows; i++) {
		p = dst.ptr<uchar>(i + loc.y - target.rows / 2);
		q = target.ptr<uchar>(i);
		//3 for 3 color channels
		for (j = 0; j < target.rows * 3; j++) {
			p[j + 3 * loc.x - 3 * (target.rows / 2)] += q[j];
		}
	}
}





void cleanFancyTarget(Mat dst, Mat target, Point loc) {
	int i, j;
	uchar * p;
	uchar * q;
	for (i = 0; i < target.rows; i++) {
		p = dst.ptr<uchar>(i + loc.y - target.rows/2);
		q = target.ptr<uchar>(i);
		//3 for 3 color channels
		for (j = 0; j < target.rows * 3; j++) {
			p[j+3*loc.x -3*(target.rows/2)] -= q[j];
		}
	}
}

void drawTarget(Mat dst, Point loc) {
	circle(dst, loc, 1, Scalar(50, 50, 200),16);
}

void cleanTarget(Mat dst, Point loc) {
	circle(dst, loc, 1, Scalar(30, 30, 30),16);
}



void trainingPhase(InputProcessing ip, Size screenResolution) {
	Mat bg(screenResolution, CV_8UC(3), Scalar((char)30,30,30));
	cv::namedWindow(WINDOW_NAME, CV_WINDOW_NORMAL);
	setWindowProperty(WINDOW_NAME, CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
	
	Mat target = imread("img/target2.png");
	Mat waitMessage = imread("img/processing.png");
	Point targetLoc(target.cols, target.rows);
	ofstream file;
	file.open("train.data");
	
		
	//snake through training points
	for (int pointCol = 0; pointCol <= POINT_COLS; pointCol++) {
		for (int pointRow2 = 0; pointRow2 <= POINT_ROWS; pointRow2++) {
			//we move up in even columns 
			int pointRow = (pointCol % 2 == 0) ? pointRow2 : (POINT_ROWS - pointRow2);
			targetLoc.y = 32 + pointRow * (screenResolution.height -64)/ POINT_ROWS;
			drawFancyTarget(bg, target, targetLoc);
			cv::imshow(WINDOW_NAME, bg);
			//wait for user to look at the point
			if (waitKey(80) == 27) {
				exit(0);
			}
			int counter = 0;
			while (counter < FRAME_COUNT) {
				if (ip.saveFeatures(file, targetLoc.x, targetLoc.y, screenResolution)) {
					counter++;
				}
				//user can cancel
				if (waitKey(20) == 27) {
					exit(0);
				}
			}

			cleanFancyTarget(bg, target, targetLoc);
		}
		targetLoc.x += (screenResolution.width - 50) / POINT_COLS;
	}
	file.close();
	string messageName = "Message";
	cv::namedWindow(messageName, CV_WINDOW_NORMAL);
	cv::imshow(messageName, waitMessage);
	destroyWindow(messageName);
}