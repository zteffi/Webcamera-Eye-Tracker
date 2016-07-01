#include <windows.h>

#include "inputprocessing.h"

/*
	returns screen resolution scaled by the scaleFactor (scaleFactor = .5 means screen capture image
	will be 1/2 width and 1/2 height of the screen resolution) 
*/
Size getScreenRes(float scaleFactor = 1);

/*
	transforms windows' window handle to a Mat object
	source: http://stackoverflow.com/a/14167433/2962048
*/
Mat  hwnd2mat(HWND hwnd, float scaleFactor);
/*
captures both screen and webcam input
*/
long captureVids(InputProcessing ip, string fileName, float scaleFactor);
/*
processes and removes video capture data
*/
void processTrackingData(InputProcessing ip, string folderName, long lastFrameNum, const char * trackFile);

/*
writes predictions into outputfile and draws predicitons onto screencapture images returns success
loadMatFromCSVFile(inputFile, Mat, 11, inputCout) returns 11 x inputCount matrix, last 2 cols are targets)
trackfile 9x trackcount matrix
outputfile 2x trackcount matrix
*/
bool processOutput(InputProcessing ip, long trainCount, const char * trainFile, long trackCount, const char * trackFile, 
	const char * outputFile, string outputImagesFolderName, Size screenSize);